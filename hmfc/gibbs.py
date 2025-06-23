import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import lax, vmap
from jax.nn import sigmoid
from jaxtyping import Float, Array
from tensorflow_probability.substrates import jax as tfp

from hmfc.constants import A_MAX, SIGMA_A_MIN, SIGMA_A_MAX, PG_TRUNC, SIGMASQ0
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
    def _sample_one(key, y_it, m_i, u_i, pg_i, params_i):
        w_i = params_i["w"]
        a_i = params_i["a"]
        mu_x_i = params_i["mu_x"]
        sigmasq_i = params_i["sigmasq"]

        # Compute b from mu_x and a
        b_i = mu_x_i * (1 - a_i)

        # Compute the LDS natural params
        J_diag = (pg_i * m_i)                                   # (T,)
        J_diag = J_diag.at[0].add(1 / SIGMASQ0)
        J_diag = J_diag.at[:-1].add(a_i**2 / sigmasq_i)
        J_diag = J_diag.at[1:].add(1. / sigmasq_i)

        # lower diagonal blocks of precision matrix
        J_lower_diag = -a_i / sigmasq_i * jnp.ones(T - 1)       # (T-1,)

        # linear potential (precision-weighted mean h)
        h = (y_it - pg_i * (u_i @ w_i) - 0.5) * m_i              # (T,)

        # Incorporate the bias
        h = h.at[0].add(mu_x_i / SIGMASQ0)
        h = h.at[:-1].add(-b_i * a_i / sigmasq_i)
        h = h.at[1:].add(b_i / sigmasq_i)

        # Run the information form sampling algorithm
        x_it = lds_info_sample(key,
                               J_diag[:, None, None],
                               J_lower_diag[:, None, None],
                               h[:, None])[:, 0]                 # (T,)

        return x_it

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
    mu_a = model.mu_a
    sigma_a = model.sigma_a
    mu_w = model.mu_w
    sigma_w = model.sigma_w
    sigma_mu_x = model.sigma_mu_x
    
    def _sample_one(key, y_it, m_i, x_it, u_i, pg_i, params_i):
        k1, k2, k3, k4 = jr.split(key, 4)

        w_i = params_i["w"]
        a_i = params_i["a"]
        mu_i = params_i["mu_x"]
        sigmasq_i = params_i["sigmasq"]

        # Gibbs sample the input weights
        J_w = 1.0 / sigma_w**2 * jnp.eye(num_inputs)
        J_w += jnp.einsum('ti,tj,t,t->ij', u_i, u_i, m_i, pg_i)
        J_w = 0.5 * (J_w + J_w.T)
        h_w = mu_w / sigma_w**2
        h_w += jnp.einsum('t,t,ti->i', y_it - pg_i * x_it - 0.5, m_i, u_i)
        w_i = _sample_info_gaussian(k1, J_w, h_w)

        # Gibbs sample the dynamics coefficient (given sigmasq_i, b_i, and rest)
        dx_it = x_it - mu_i
        J_a = 1.0 / sigma_a**2 + jnp.sum(m_i[1:] * dx_it[:-1]**2) / sigmasq_i
        h_a = mu_a / sigma_a**2 + jnp.sum(m_i[1:] * dx_it[:-1] * dx_it[1:]) / sigmasq_i
        a_i = tfd.TruncatedNormal(h_a / J_a, jnp.sqrt(1.0 / J_a), 0.0, A_MAX).sample(seed=k2)

        # Gibbs sample the bias term (given a_i and rest)
        J_mu_x = 1 / sigma_mu_x**2 
        J_mu_x += m_i[0] * 1 / SIGMASQ0
        J_mu_x += jnp.sum(m_i[1:]) * (1 - a_i)**2 / sigmasq_i
        h_mu_x = m_i[0] * x_it[0]
        h_mu_x += jnp.sum(m_i[1:] * (x_it[1:] - a_i * x_it[:-1]) * (1 - a_i)) / sigmasq_i
        mu_i = tfd.Normal(h_mu_x / J_mu_x, jnp.sqrt(1.0 / J_mu_x)).sample(seed=k3)

        # Gibbs sample the dynamics noise variance (given a_i and rest)
        alpha0, beta0 = convert_mean_to_std_ig_params(model.mu_sigmasq, model.beta_sigmasq)
        alpha_post = alpha0 + 0.5 * jnp.sum(m_i[1:])
        beta_post = beta0 + 0.5 * jnp.sum(m_i[1:] * (x_it[1:] - a_i * x_it[:-1] - (1 - a_i) * mu_i)**2)
        sigmasq_i = tfd.InverseGamma(alpha_post, beta_post).sample(seed=k4)
        return dict(a=a_i, mu_x=mu_i, w=w_i, sigmasq=sigmasq_i)

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
    Update the global params mu_w with Gibbs
    """
    
    sigma_w = jnp.exp(model.log_sigma_w)
    ws = params["w"]
    N, D = ws.shape # N = number of subject, D = number of input variables
    mu_w = tfd.Normal(ws.mean(axis=0), sigma_w / jnp.sqrt(N)).sample(seed=key) # returns (D,) samples of \mu_w
    
    model = eqx.tree_at(lambda m: m.mu_w, model, mu_w)

    return model

def _gibbs_step_global_weights_var(key,
                                   model : HierarchicalBernoulliLDS,
                                   params : dict):
    r"""
    Update the global params sigmasq_w with Gibbs
    """

    sigma_w = jnp.exp(model.log_sigma_w)
    mu_w = model.mu_w
    ws = params["w"]
    N, D = ws.shape # N = number of subject, D = number of input variables
    sigma_w = jnp.sqrt(tfd.InverseGamma(2.1 + 0.5 * N, 1.1 + 0.5 * jnp.sum((ws - mu_w)**2, axis=0)).sample(seed=key))    # returns (D,) samples of \sigma_w
    sigma_w = jnp.clip(sigma_w, a_min=1e-4) # specify lower bound such that sigma_w cannot go to zero

    model = eqx.tree_at(lambda m: m.log_sigma_w, model, jnp.log(sigma_w))

    return model

def _gibbs_step_global_bias_var(key,
                               model : HierarchicalBernoulliLDS,
                               params : dict):
    r"""
    Update the global params sigmasq_mu_x with Gibbs
    """

    mu_xs = params["mu_x"]
    N, = mu_xs.shape
    sigma_mu_x = jnp.sqrt(tfd.InverseGamma(2.1 + 0.5 * N, 1.1 + 0.5 * jnp.sum((mu_xs - 0)**2, axis=0)).sample(seed=key))
    sigma_mu_x = jnp.clip(sigma_mu_x, a_min=1e-4) # specify lower bound such that sigma_mu_x cannot go to zero

    model = eqx.tree_at(lambda m: m.log_sigma_mu_x, model, jnp.log(sigma_mu_x))

    return model

def _gibbs_step_global_ar(key,
                          model: HierarchicalBernoulliLDS,
                          params: dict,
                          proposal_variance: float=0.1**2,
                          num_steps: int=20):
    r"""
    Update the global params mu_a with RWMH
    """
    def _log_prob(logit_mu_a):
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, A_MAX),
            tfb.Invert(tfb.Sigmoid()),
        ).log_prob(logit_mu_a)

        lp += tfd.TruncatedNormal(sigmoid(logit_mu_a), jnp.exp(model.log_sigma_a),
                                  0.0, A_MAX).log_prob(params["a"]).sum()
        return lp

    logit_mu_a = random_walk_mh(key,
                               _log_prob,
                               model.logit_mu_a,
                               proposal_variance,
                               num_steps)

    model = eqx.tree_at(lambda m: m.logit_mu_a, model, logit_mu_a)

    return model

def _gibbs_step_global_ar_var(key,
                              model: HierarchicalBernoulliLDS,
                              params: dict,
                              proposal_variance: float=0.1**2,
                              num_steps: int=20):
    r"""
    Update the global params sigma_a with RWMH
    """

    def _log_prob(log_sigma_a):
        lp = tfd.TransformedDistribution(
            tfd.Uniform(SIGMA_A_MIN, SIGMA_A_MAX),
            tfb.Log(),
        ).log_prob(log_sigma_a)

        # log likelihood: \sum_i log p(a_i | mu_a, \sigma_a^2)
        lp += tfd.TruncatedNormal(sigmoid(model.logit_mu_a), jnp.exp(log_sigma_a),
                                  0.0, A_MAX).log_prob(params["a"]).sum()
        return lp

    log_sigma_a = random_walk_mh(key,
                              _log_prob,
                              model.log_sigma_a,
                              proposal_variance,
                              num_steps)

    model = eqx.tree_at(lambda m: m.log_sigma_a, model, log_sigma_a)
    
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
                                    proposal_variance_beta: float=0.1**2,
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
                             update_global_weights_var: bool=True,
                             update_global_bias_var: bool=True,
                             update_global_ar: bool=True,
                             update_global_ar_var: bool=True,
                             update_global_mu_sigmasq: bool=True,
                             update_global_beta_sigmasq: bool=True):
    
    k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)
    
    if update_global_weights: model = _gibbs_step_global_weights(k1, model, params)
    if update_global_weights_var: model = _gibbs_step_global_weights_var(k2, model, params)
    if update_global_bias_var: model = _gibbs_step_global_bias_var(k3, model, params)
    if update_global_ar: model = _gibbs_step_global_ar(k4, model, params)
    if update_global_ar_var: model = _gibbs_step_global_ar_var(k5, model, params)
    if update_global_mu_sigmasq: model = _gibbs_step_global_mu_sigmasq(k6, model, params)
    if update_global_beta_sigmasq: model = _gibbs_step_global_beta_sigmasq(k7, model, params)
        
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

    def _sample_one(key, x_it, u_i, w_i):
        psi_i = x_it + u_i @ w_i
        return vmap(_pg_sample)(jr.split(key, num_trials),
                                jnp.ones(num_trials),
                                psi_i)

    return vmap(_sample_one)(jr.split(key, num_subjects),
                             states,
                             inputs,
                             params["w"])

@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12, 13))
def gibbs_step(key,
               emissions : Float[Array, "num_subjects num_trials"],
               masks: Float[Array, "num_subjects num_trials"],
               states: Float[Array, "num_subjects num_trials"],
               inputs : Float[Array, "num_subjects num_trials num_inputs"],
               params: dict,
               model : HierarchicalBernoulliLDS,
               update_global_weights: bool=True,
               update_global_weights_var: bool=True,
               update_global_bias_var: bool=True,
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
                                     update_global_weights_var=update_global_weights_var,
                                     update_global_bias_var=update_global_bias_var,
                                     update_global_ar=update_global_ar,
                                     update_global_ar_var=update_global_ar_var,
                                     update_global_mu_sigmasq=update_global_mu_sigmasq,
                                     update_global_beta_sigmasq=update_global_beta_sigmasq)

    return lp, states, params, model