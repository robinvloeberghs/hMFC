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
    log_lambda_mu : Float = 3.0               # rate of the exponential prior on mu (mean of inverse gamma prior for sigmasq of latent states)
    log_lambda_beta  : Float = 3.0            # rate of the exponential prior on beta (scale of inverse gamma prior for sigmasq of latent states)
    
    # Global parameters of the model
    mu_w       : Float[Array, "num_inputs"]   # mean normal for input weights
    log_sigma_w  : Float[Array, "num_inputs"] # sd normal for input weights
    logit_mu_a : Float = 10.0                 # mean truncated normal for autoregressive coefficients (latent states) in unconstrained form (allows HMC)
    log_sigma_a  : Float = -1.0               # sd truncated normal for autoregressive coefficient (latent states) in unconstrained form
    log_sigma_mu_x : Float = -1.0             # sd normal for mu_x
    log_mu_sigmasq: Float = -2.3              # log of the mean of sigmasq (variance of latent states)
    log_beta_sigmasq: Float = -2.3            # log of the scale of sigmasq (variance of latent states)

    def __init__(self, 
                 num_inputs : int,
                 mu_a : float = 0.99, 
                 sigma_a : float = 0.1, 
                 mu_w: Union[float, Float[Array, "num_inputs"]] = 0.0, 
                 sigma_w: Union[float, Float[Array, "num_inputs"]] = 1.0,
                 mu_sigmasq: float = 0.1, 
                 beta_sigmasq: float = 0.1, 
                 sigma_mu_x: float = 1.0, 
                 lambda_mu: float = 10., 
                 lambda_beta: float = 10.):
        
        # Set the hyperparameters
        self.log_lambda_mu = jnp.log(lambda_mu)
        self.log_lambda_beta = jnp.log(lambda_beta)
        
        # Set the global parameters
        self.logit_mu_a = jnp.log(mu_a / (1 - mu_a))
        self.log_sigma_a = jnp.log(sigma_a)
        self.mu_w = mu_w if isinstance(mu_w, jnp.ndarray) else jnp.full((num_inputs,), mu_w)
        self.log_sigma_w = jnp.log(sigma_w) if isinstance(sigma_w, jnp.ndarray) else jnp.full((num_inputs,), jnp.log(sigma_w))
        self.log_sigma_mu_x = jnp.log(sigma_mu_x)
        self.log_mu_sigmasq = jnp.log(mu_sigmasq)
        self.log_beta_sigmasq = jnp.log(beta_sigmasq)

    @property
    def num_inputs(self):
        return self.mu_w.shape[0]
    
    @property
    def lambda_mu(self):
        return jnp.exp(self.log_lambda_mu)
    
    @property
    def lambda_beta(self):
        return jnp.exp(self.log_lambda_beta)
    
    @property
    def sigma_w(self):
        return jnp.exp(self.log_sigma_w)
    
    @property
    def mu_a(self):
        return sigmoid(self.logit_mu_a)
    
    @property
    def sigma_a(self):
        return jnp.exp(self.log_sigma_a)
    
    @property
    def sigma_mu_x(self):
        return jnp.exp(self.log_sigma_mu_x)

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
        assert num_inputs == self.mu_w.shape[0]

        def _sample_one(key, u_i):
            k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)

            # Sample per trial parameters
            w_i = tfd.Normal(self.mu_w, self.sigma_w).sample(seed=k1)
        
            a_i = tfd.TruncatedNormal(self.mu_a, self.sigma_a, 0.0, A_MAX).sample(seed=k2)
            
            sigmasq_i = tfd.InverseGamma(
                *convert_mean_to_std_ig_params(self.mu_sigmasq, self.beta_sigmasq)
            ).sample(seed=k3)

            mu_x_i = tfd.Normal(0, self.sigma_mu_x).sample(seed=k4)
            b_i = mu_x_i * (1 - a_i)
 
            # Sample latent states starting at the stationary distribution
            # Stationary covariance is \sigma_0^2 = \sigma^2 / (1 - a^2)
            # but for simplicity, we assume the initial variance is 1.0
            x_it0 = tfd.Normal(mu_x_i, jnp.sqrt(SIGMASQ0)).sample(seed=k5)
  
            def _step(x_itp, key):
                x_itp1 = tfd.Normal(a_i * x_itp + b_i, jnp.sqrt(sigmasq_i)).sample(seed=key)
                return x_itp1, x_itp
            _, x_it = lax.scan(_step, x_it0, jr.split(k6, num_trials))

            # Sample emissions
            y_it = tfd.Bernoulli(x_it + u_i @ w_i).sample(seed=k7)
            return dict(a=a_i, mu_x=mu_x_i, w=w_i, sigmasq=sigmasq_i), x_it, y_it

        return vmap(_sample_one)(jr.split(key, num_subjects), inputs)


    def log_prob(self,
                 emissions : Float[Array, "num_subjects num_trials"],
                 masks : Float[Array, "num_subjects num_trials"],
                 states : Float[Array, "num_subjects num_trials"],
                 inputs : Float[Array, "num_subjects num_trials num_inputs"],
                 params : dict):
        def _single_lp(y_it, m_i, x_it, u_i, params_i):
            a_i = params_i["a"]
            mu_x_i = params_i["mu_x"]
            w_i = params_i["w"]
            sigmasq_i = params_i["sigmasq"]

            # Derive b_i from mu_x_i and a_i
            b_i = mu_x_i * (1 - a_i)

            # \log p(\theta_i | \eta)
            lp_i = tfd.TruncatedNormal(self.mu_a, self.sigma_a, 0.0, A_MAX).log_prob(a_i)
            lp_i += tfd.Normal(0.0, self.sigma_mu_x).log_prob(mu_x_i)
            lp_i += tfd.Normal(self.mu_w, self.sigma_w).log_prob(w_i)
            lp_i += tfd.InverseGamma(
                *convert_mean_to_std_ig_params(self.mu_sigmasq, self.beta_sigmasq)
                ).log_prob(sigmasq_i)

            # \log p(x_it | \theta_i)
            lp_i += tfd.Normal(mu_x_i, jnp.sqrt(SIGMASQ0)).log_prob(x_it[0])
            lp_i += tfd.Normal(a_i * x_it[:-1] + b_i, jnp.sqrt(sigmasq_i)).log_prob(x_it[1:]).sum()

            # \log p(y_it | x_it, u_i, \theta_i)
            lp_i += jnp.sum(m_i * tfd.Bernoulli(x_it + u_i @ w_i).log_prob(y_it)) # m_i is mask (if 0 then just sum 0)
            return lp_i

        return vmap(_single_lp)(emissions, masks, states, inputs, params).sum()
