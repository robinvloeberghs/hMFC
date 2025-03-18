import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax import lax, vmap
from jax.nn import sigmoid
from jaxtyping import Float, Array
from tensorflow_probability.substrates import jax as tfp
from typing import Union

from hmfc.constants import A_MAX, SIGMASQ0
from hmfc.utils import convert_mean_to_std_ig_params

tfd = tfp.distributions
tfb = tfp.bijectors


class HierarchicalBernoulliLDS(eqx.Module):
    r"""
    Implementation of the model.
    """
    # Hyperparameters of the prior distributions
    log_lambda_mu : Float = 3.0              # rate of the exponential prior on mu (mean of inverse gamma prior for sigmasq of latent states)
    log_lambda_beta  : Float = 3.0           # rate of the exponential prior on beta (scale of inverse gamma prior for sigmasq of latent states)
    
    # Global parameters of the model
    w_0       : Float[Array, "num_inputs"]   # mean normal for input weights
    log_nu_w  : Float[Array, "num_inputs"]   # sd normal for input weights
    logit_a_0 : Float = 10.0                 # mean truncated normal for autoregressive coefficients (latent states) in unconstrained form (allows HMC)
    log_nu_a  : Float = -1.0                 # sd truncated normal for autoregressive coefficient (latent states) in unconstrained form
    log_nu_mu0 : Float = -1.0                # sd normal for mu0
    log_mu_sigmasq: Float = -2.3             # log of the mean of sigmasq (variance of latent states)
    log_beta_sigmasq: Float = -2.3           # log of the scale of sigmasq (variance of latent states)

    def __init__(self, 
                 num_inputs : int,
                 a_0 : float = 0.99, 
                 nu_a : float = 0.1, 
                 w_0: Union[float, Float[Array, "num_inputs"]] = 0.0, 
                 nu_w: Union[float, Float[Array, "num_inputs"]] = 1.0,
                 mu_sigmasq: float = 0.1, 
                 beta_sigmasq: float = 0.1, 
                 nu_mu0: float = 1.0, 
                 lambda_mu: float = 10., 
                 lambda_beta: float = 10.):
        # Set the hyperparameters
        self.log_lambda_mu = jnp.log(lambda_mu)
        self.log_lambda_beta = jnp.log(lambda_beta)
        
        # Set the global parameters
        self.logit_a_0 = jnp.log(a_0 / (1 - a_0))
        self.log_nu_a = jnp.log(nu_a)
        self.w_0 = w_0 if isinstance(w_0, jnp.ndarray) else jnp.full((num_inputs,), w_0)
        self.log_nu_w = jnp.log(nu_w) if isinstance(nu_w, jnp.ndarray) else jnp.full((num_inputs,), jnp.log(nu_w))
        self.log_nu_mu0 = jnp.log(nu_mu0)
        self.log_mu_sigmasq = jnp.log(mu_sigmasq)
        self.log_beta_sigmasq = jnp.log(beta_sigmasq)

    @property
    def num_inputs(self):
        return self.w_0.shape[0]
    
    @property
    def lambda_mu(self):
        return jnp.exp(self.log_lambda_mu)
    
    @property
    def lambda_beta(self):
        return jnp.exp(self.log_lambda_beta)
    
    @property
    def nu_w(self):
        return jnp.exp(self.log_nu_w)
    
    @property
    def a_0(self):
        return sigmoid(self.logit_a_0)
    
    @property
    def nu_a(self):
        return jnp.exp(self.log_nu_a)
    
    @property
    def nu_mu0(self):
        return jnp.exp(self.log_nu_mu0)

    @property
    def mu_sigmasq(self):
        return jnp.exp(self.log_mu_sigmasq)
    
    @property
    def beta_sigmasq(self):
        return jnp.exp(self.log_beta_sigmasq)

    def sample(self,
               key,
               inputs : Float[Array, "num_subjects num_trials num_inputs"]
               ):
        r"""
        Draw a sample from the generative model.
        """
        num_subjects, num_trials, num_inputs = inputs.shape
        assert num_inputs == self.w_0.shape[0]

        def _sample_one(key, u_i):
            k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)

            # Sample per trial parameters
            w_i = tfd.Normal(self.w_0, self.nu_w).sample(seed=k1)
        
            a_i = tfd.TruncatedNormal(self.a_0, self.nu_a, 0.0, A_MAX).sample(seed=k2)
            
            sigmasq_i = tfd.InverseGamma(
                *convert_mean_to_std_ig_params(self.mu_sigmasq, self.beta_sigmasq)
            ).sample(seed=k3)

            mu0_i = tfd.Normal(0, self.nu_mu0).sample(seed=k4)
            b_i = mu0_i * (1 - a_i)
 
            # Sample latent states starting at the stationary distribution
            # Stationary covariance is \sigma_0^2 = \sigma^2 / (1 - a^2)
            # but for simplicity, we assume the initial variance is 1.0
            x_i0 = tfd.Normal(mu0_i, jnp.sqrt(SIGMASQ0)).sample(seed=k5)
  
            def _step(x_it, key):
                x_itp1 = tfd.Normal(a_i * x_it + b_i, jnp.sqrt(sigmasq_i)).sample(seed=key)
                return x_itp1, x_it
            _, x_i = lax.scan(_step, x_i0, jr.split(k6, num_trials))

            # Sample emissions
            y_i = tfd.Bernoulli(x_i + u_i @ w_i).sample(seed=k7)
            return dict(a=a_i, mu0=mu0_i, w=w_i, sigmasq=sigmasq_i), x_i, y_i

        return vmap(_sample_one)(jr.split(key, num_subjects), inputs)


    def log_prob(self,
                 emissions : Float[Array, "num_subjects num_trials"],
                 masks : Float[Array, "num_subjects num_trials"],
                 states : Float[Array, "num_subjects num_trials"],
                 inputs : Float[Array, "num_subjects num_trials num_inputs"],
                 params : dict):
        def _single_lp(y_i, m_i, x_i, u_i, params_i):
            a_i = params_i["a"]
            mu0_i = params_i["mu0"]
            w_i = params_i["w"]
            sigmasq_i = params_i["sigmasq"]

            # Derive b_i from mu0_i and a_i
            b_i = mu0_i * (1 - a_i)

            # \log p(\theta_i | \eta)
            lp_i = tfd.TruncatedNormal(self.a_0, self.nu_a, 0.0, A_MAX).log_prob(a_i)
            lp_i += tfd.Normal(0.0, self.nu_mu0).log_prob(mu0_i)
            lp_i += tfd.Normal(self.w_0, self.nu_w).log_prob(w_i)
            lp_i += tfd.InverseGamma(
                *convert_mean_to_std_ig_params(self.mu_sigmasq, self.beta_sigmasq)
                ).log_prob(sigmasq_i)

            # \log p(x_i | \theta_i)
            lp_i += tfd.Normal(mu0_i, jnp.sqrt(SIGMASQ0)).log_prob(x_i[0])
            lp_i += tfd.Normal(a_i * x_i[:-1] + b_i, jnp.sqrt(sigmasq_i)).log_prob(x_i[1:]).sum()

            # \log p(y_i | x_i, u_i, \theta_i)
            lp_i += jnp.sum(m_i * tfd.Bernoulli(x_i + u_i @ w_i).log_prob(y_i)) # m_i is mask (if 0 then just sum 0)
            return lp_i

        return vmap(_single_lp)(emissions, masks, states, inputs, params).sum()
