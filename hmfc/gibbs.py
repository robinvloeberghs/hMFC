import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import lax, vmap
from jax.nn import sigmoid
from jaxtyping import Float, Array
from tensorflow_probability.substrates import jax as tfp

from hmfc.constants import A_MAX, NU_A_MAX, PG_TRUNC, SIGMASQ0
from hmfc.lds import lds_info_sample, _sample_info_gaussian
from hmfc.model import HierarchicalBernoulliLDS
from hmfc.utils import convert_mean_to_std_ig_params

tfd = tfp.distributions
tfb = tfp.bijectors

def gibbs_step_states(key,
                      emissions : Float[Array, "num_subjects num_trials"],
                      masks: Float[Array, "num_subjects num_trials"],
                      inputs : Float[Array, "num_subjects num_trials num_inputs"],
                      pg_samples : Float[Array, "num_subjects num_trials"],
                      params: dict):
    """
    Draw a sample of the latent states from their conditional distribution
    given emissions, inputs, auxiliary PG variables, and parameters.
    """
    N, T, D = inputs.shape
    def _sample_one(key, y_i, m_i, u_i, pg_i, params_i):
        w_i = params_i["w"]
        a_i = params_i["a"]
        mu0_i = params_i["mu0"]
        sigmasq_i = params_i["sigmasq"]

        # Compute b from mu0 and a
        b_i = mu0_i * (1 - a_i)

        # Compute the LDS natural params
        J_diag = (pg_i * m_i)                                   # (T,)
        J_diag = J_diag.at[0].add(1 / SIGMASQ0)
        J_diag = J_diag.at[:-1].add(a_i**2 / sigmasq_i)
        J_diag = J_diag.at[1:].add(1. / sigmasq_i)

        # lower diagonal blocks of precision matrix
        J_lower_diag = -a_i / sigmasq_i * jnp.ones(T - 1)       # (T-1,)

        # linear potential (precision-weighted mean h)
        h = (y_i - pg_i * (u_i @ w_i) - 0.5) * m_i              # (T,)

        # Incorporate the bias
        h = h.at[0].add(mu0_i / SIGMASQ0)
        h = h.at[:-1].add(-b_i * a_i / sigmasq_i)
        h = h.at[1:].add(b_i / sigmasq_i)

        # Run the information form sampling algorithm
        x_i = lds_info_sample(key,
                              J_diag[:, None, None],
                              J_lower_diag[:, None, None],
                              h[:, None])[:, 0]                 # (T,)

        return x_i

    return vmap(_sample_one)(jr.split(key, N),
                             emissions,
                             masks,
                             inputs,
                             pg_samples,
                             params)

def gibbs_step_local_params(key,
                            emissions : Float[Array, "num_subjects num_trials"],
                            masks: Float[Array, "num_subjects num_trials"],
                            states: Float[Array, "num_subjects num_trials"],
                            inputs : Float[Array, "num_subjects num_trials num_inputs"],
                            pg_samples : Float[Array, "num_subjects num_trials"],
                            params: dict,
                            model : HierarchicalBernoulliLDS,
                            ):
    r"""
    Perform one Gibbs step to update the local parameters.
    """
    num_subjects, num_trials, num_inputs = inputs.shape
    a_0 = sigmoid(model.logit_a_0)
    nu_a = jnp.exp(model.log_nu_a)
    w_0 = model.w_0
    nu_w = jnp.exp(model.log_nu_w)
    nu_mu0 = jnp.exp(model.log_nu_mu0)
    
    def _sample_one(key, y_i, m_i, x_i, u_i, pg_i, params_i):
        k1, k2, k3, k4 = jr.split(key, 4)

        # Gibbs sample the input weights
        J_w = 1.0 / nu_w**2 * jnp.eye(num_inputs)
        J_w += jnp.einsum('ti,tj,t,t->ij', u_i, u_i, m_i, pg_i)
        J_w = 0.5 * (J_w + J_w.T)
        h_w = w_0 / nu_w**2
        h_w += jnp.einsum('t,t,ti->i', y_i - pg_i * x_i - 0.5, m_i, u_i)
        w_i = _sample_info_gaussian(k1, J_w, h_w)

        # Gibbs sample the dynamics coefficient (given sigmasq_i, b_i, and rest)
        # TODO: Double check the conditional distribution of a_i | b_i
        a_i = params_i["a"]
        mu0_i = params_i["mu0"]
        b_i = mu0_i * (1 - a_i)
        sigmasq_i = params_i["sigmasq"]
        J_a = 1.0 / nu_a**2 + jnp.sum(m_i[1:] * x_i[:-1]**2) / sigmasq_i
        h_a = a_0 / nu_a**2 + jnp.sum(m_i[1:] * x_i[:-1] * (x_i[1:] - b_i)) / sigmasq_i
        a_i = tfd.TruncatedNormal(h_a / J_a, jnp.sqrt(1.0 / J_a), 0.0, A_MAX).sample(seed=k2)

        # Gibbs sample the bias term (given a_i and rest)
        # p(mu0 | a, x)
        # \propto N(mu0 | 0, nu_mu0^2) N(x_1 | mu0, 1) \prod_{t=1}^{T-1} N(x_{t+1} - a x_t|  mu0 (1 - a), \sigma^2)
        # \propto N(mu0 | 0, nu_mu0^2) N(x_1 | mu0, 1) \prod_{t=1}^{T-1} N((x_{t+1} - a x_t) / (1 - a) |  mu0 , \sigma^2 / (1 - a)^2)
        J_mu0 = 1/nu_mu0**2 + m_i[0] + jnp.sum(m_i[1:]) * (1 - a_i)**2 / sigmasq_i
        h_mu0 = 0 + m_i[0] * x_i[0] + jnp.sum(m_i[1:] * (x_i[1:] - a_i * x_i[:-1]) / (1 - a_i)) * (1 - a_i)**2 / sigmasq_i
        mu0_i = tfd.Normal(h_mu0 / J_mu0, jnp.sqrt(1.0 / J_mu0)).sample(seed=k3)
        b_i = mu0_i * (1 - a_i)

        # Gibbs sample the dynamics noise variance (given a_i and rest)
        alpha0, beta0 = convert_mean_to_std_ig_params(model.mu_sigmasq, model.beta_sigmasq)
        alpha_post = alpha0 + 0.5 * jnp.sum(m_i[1:])
        beta_post = beta0 + 0.5 * jnp.sum(m_i[1:] * (x_i[1:] - a_i * x_i[:-1] - b_i)**2)
        sigmasq_i = tfd.InverseGamma(alpha_post, beta_post).sample(seed=k4)
        return dict(a=a_i, mu0=mu0_i, w=w_i, sigmasq=sigmasq_i)

    return vmap(_sample_one)(jr.split(key, num_subjects),
                             emissions,
                             masks,
                             states,
                             inputs,
                             pg_samples,
                             params)

def random_walk_mh(key,
                   log_prob,
                   current_param,
                   proposal_variance,
                   num_steps=1
                   ):
    """
    Run Metropolis Hastings with symmetric Gaussian proposal distribution.
    This is called "Random Walk MH".

    accept_prob = min{1, q(x | x') / q(x' | x) * p(x') / p(x)}
    log(accept_prob) = min{0, log p(x') - p(x)}

    """
    def _step(carry, key):
        x, lp_x = carry
        k1, k2 = jr.split(key)
        prop_x = tfd.Normal(x, jnp.sqrt(proposal_variance)).sample(seed=k1)
        lp_prop_x = log_prob(prop_x)
        accept = jnp.log(tfd.Uniform(0, 1).sample(seed=k2)) < (lp_prop_x - lp_x) # log (ratio new and old value)
        new_x = jnp.where(accept, prop_x, x)
        new_lp_x = jnp.where(accept, lp_prop_x, lp_x)
        return (new_x, new_lp_x), None

    initial_carry = (current_param, log_prob(current_param))
    (x, _), _ = lax.scan(_step, initial_carry, jr.split(key, num_steps))
    return x

def _gibbs_step_global_weights(key,
                               model : HierarchicalBernoulliLDS,
                               params : dict):
    r"""
    Update the global params w_0, nu_w, nu_mu0 with Gibbs
    """
    k1, k2, k3 = jr.split(key, 3)

    # Update the global mean, w_0
    nu_w = jnp.exp(model.log_nu_w)
    ws = params["w"]
    N, D = ws.shape # N = number of subject, D = number of input variables
    w_0 = tfd.Normal(ws.mean(axis=0), nu_w / jnp.sqrt(N)).sample(seed=k1) # draw w_0 for each input variable
    model = eqx.tree_at(lambda m: m.w_0, model, w_0)

    # Update the global variance, nu_w^2
    nu_w = jnp.sqrt(tfd.InverseGamma(0.5 * N, 0.5 * jnp.sum((ws - w_0)**2, axis=0)).sample(seed=k2))    # returns (D,) samples of \nu_w
    nu_w = jnp.clip(nu_w, a_min=1e-4) # specify lower bound such that nu_w cannot go to zero

    model = eqx.tree_at(lambda m: m.log_nu_w, model, jnp.log(nu_w))

    # Update the global bias variance, nu_mu0^2
    mu0s = params["mu0"]
    nu_mu0 = jnp.sqrt(tfd.InverseGamma(0.5 * N, 0.5 * jnp.sum((mu0s - 0)**2, axis=0)).sample(seed=k3))
    nu_mu0 = jnp.clip(nu_mu0, a_min=1e-4) # specify lower bound such that nu_mu0 cannot go to zero

    model = eqx.tree_at(lambda m: m.log_nu_mu0, model, jnp.log(nu_mu0))

    return model


def _gibbs_step_global_ar(key,
                          model: HierarchicalBernoulliLDS,
                          params: dict,
                          proposal_variance: float=0.05**2,
                          num_steps: int=20):
    r"""
    Update the global params a_0, nu_a with RWMH
    """
    def _log_prob(logit_a_0):
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, A_MAX),
            tfb.Invert(tfb.Sigmoid()),
        ).log_prob(logit_a_0)

        lp += tfd.TruncatedNormal(sigmoid(logit_a_0), jnp.exp(model.log_nu_a),
                                  0.0, A_MAX).log_prob(params["a"]).sum()
        return lp

    logit_a_0 = random_walk_mh(key,
                               _log_prob,
                               model.logit_a_0,
                               proposal_variance,
                               num_steps)

    model = eqx.tree_at(lambda m: m.logit_a_0, model, logit_a_0)

    return model

def _gibbs_step_global_ar_var(key,
                              model: HierarchicalBernoulliLDS,
                              params: dict,
                              proposal_variance: float=0.05**2,
                              num_steps: int =20):
    r"""
    Update the global params a_0, nu_a with RWMH
    """

    def _log_prob(log_nu_a):
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, NU_A_MAX),
            tfb.Log(),
        ).log_prob(log_nu_a)

        # log likelihood: \sum_i log p(a_i | a_0, \nu_a^2)
        lp += tfd.TruncatedNormal(sigmoid(model.logit_a_0), jnp.exp(log_nu_a),
                                  0.0, A_MAX).log_prob(params["a"]).sum()
        return lp

    log_nu_a = random_walk_mh(key,
                              _log_prob,
                              model.log_nu_a,
                              proposal_variance,
                              num_steps)

    model = eqx.tree_at(lambda m: m.log_nu_a, model, log_nu_a)
    return model

def _gibbs_step_global_mu_sigmasq(key,
                                  model: HierarchicalBernoulliLDS,
                                  params: dict,
                                  proposal_variance_mu: float=0.1**2,
                                  num_steps_mu: int=20):
    r"""
    Update mean of inverse gamma for sigmasq (mu_sigmasq) with RWMH 
    """

    def _log_prob_mu_sigmasq(log_mu):

        lp = tfd.TransformedDistribution(
            tfd.Gamma(2.0, 2.0 * model.lambda_mu),
            tfb.Log(),
        ).log_prob(log_mu)

        alpha, beta = convert_mean_to_std_ig_params(jnp.exp(log_mu), model.beta_sigmasq)
        lp += tfd.InverseGamma(alpha, beta).log_prob(params["sigmasq"]).sum()
        return lp

    log_mu_sigmasq = random_walk_mh(key,
                                   _log_prob_mu_sigmasq,
                                   model.log_mu_sigmasq,
                                   proposal_variance_mu,
                                   num_steps_mu)

    model = eqx.tree_at(lambda m: m.log_mu_sigmasq, model, log_mu_sigmasq)
    return model

def _gibbs_step_global_beta_sigmasq(key,
                                    model: HierarchicalBernoulliLDS,
                                    params: dict,
                                    proposal_variance_beta: float=0.1,
                                    num_steps_beta: int=20):
    r"""
    Update beta of inverse gamma for sigmasq with RWMH

    TODO: implement Gibbs step for beta_sigmasq (it should have a gamma conditional)
    """

    def _log_prob_beta(log_beta):

        lp = tfd.TransformedDistribution(
            tfd.Exponential(model.lambda_beta),
            tfb.Log(),
        ).log_prob(log_beta)

        alpha, beta = convert_mean_to_std_ig_params(model.mu_sigmasq, jnp.exp(log_beta))

        lp += tfd.InverseGamma(alpha, jnp.exp(log_beta)).log_prob(params["sigmasq"]).sum()
        return lp

    log_beta = random_walk_mh(key,
                              _log_prob_beta,
                              model.log_beta_sigmasq,
                              proposal_variance_beta,
                              num_steps_beta)

    model = eqx.tree_at(lambda m: m.log_beta_sigmasq, model, log_beta)
    return model

def gibbs_step_global_params(key,
                             model : HierarchicalBernoulliLDS,
                             params : dict,
                             update_global_weights: bool=True,
                             update_global_ar: bool=True,
                             update_global_ar_var: bool=True,
                             update_global_mu_sigmasq: bool=True,
                             update_global_beta_sigmasq: bool=True):
    k1, k2, k3, k4, k5 = jr.split(key, 5)
    if update_global_weights: model = _gibbs_step_global_weights(k1, model, params)
    if update_global_ar: model = _gibbs_step_global_ar(k2, model, params)
    if update_global_ar_var: model = _gibbs_step_global_ar_var(k3, model, params)
    if update_global_mu_sigmasq: model = _gibbs_step_global_mu_sigmasq(k4, model, params)
    if update_global_beta_sigmasq: model = _gibbs_step_global_beta_sigmasq(k5, model, params)
    return model

def _pg_sample(key, b, c):
    '''pg(b,c) =
    1/(2pi)^2\sum_k=1^\infinity \dfrac{g_k}{(k-1/2)^2+c^2/(4pi^2)}
    where g_k ~ Ga(b,1)'''
    gammas = jr.gamma(key, b, shape=(PG_TRUNC,))
    scaling = 1 / (4 * jnp.pi ** 2 * (jnp.arange(1, PG_TRUNC + 1) - 1 / 2) ** 2 + c ** 2)
    pg = 2 * jnp.sum(gammas * scaling)
    return jnp.clip(pg, 1e-2, jnp.inf)


def gibbs_step_pg(key,
                  states: Float[Array, "num_subjects num_trials"],
                  inputs : Float[Array, "num_subjects num_trials num_inputs"],
                  params: dict,
                  ):

    num_subjects, num_trials, _ = inputs.shape

    def _sample_one(key, x_i, u_i, w_i):
        psi_i = x_i + u_i @ w_i
        return vmap(_pg_sample)(jr.split(key, num_trials),
                                jnp.ones(num_trials),
                                psi_i)

    return vmap(_sample_one)(jr.split(key, num_subjects),
                             states,
                             inputs,
                             params["w"])

@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def gibbs_step(key,
               emissions : Float[Array, "num_subjects num_trials"],
               masks: Float[Array, "num_subjects num_trials"],
               states: Float[Array, "num_subjects num_trials"],
               inputs : Float[Array, "num_subjects num_trials num_inputs"],
               params: dict,
               model : HierarchicalBernoulliLDS,
               update_global_weights: bool=True,
               update_global_ar: bool=True,
               update_global_ar_var: bool=True,
               update_global_mu_sigmasq: bool=True,
               update_global_beta_sigmasq: bool=True
               ):
    k1, k2, k3, k4 = jr.split(key, 4)

    # 0. Evaluate log joint probability
    lp = model.log_prob(emissions, masks, states, inputs, params)

    # 1. Sample PG auxiliary variables
    pg_samples = gibbs_step_pg(k1, states, inputs, params)

    # 2. Sample local params
    params = gibbs_step_local_params(k2, emissions, masks, states, inputs, pg_samples, params, model)

    # 3. Sample new latent states
    states = gibbs_step_states(k3, emissions, masks, inputs, pg_samples, params)

    # 4. Sample new global params
    model = gibbs_step_global_params(k4, model, params,
                                     update_global_weights=update_global_weights,
                                     update_global_ar=update_global_ar,
                                     update_global_ar_var=update_global_ar_var,
                                     update_global_mu_sigmasq=update_global_mu_sigmasq,
                                     update_global_beta_sigmasq=update_global_beta_sigmasq)

    return lp, states, params, model