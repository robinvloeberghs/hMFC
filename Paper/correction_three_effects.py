# -*- coding: utf-8 -*-
"""
@author: Robin Vloeberghs
"""

"""
In this code we'll show through simulations how fluctuations in decision criterion
can generate apparent sequential effects and affect the psychometric slope and d',
both popular measures used to quantify sensitivity. Next, we'll demonstrate that
these biases/misestimates can be ameliorated by estimating the criterion fluctuations with hMFC.

Note: the script below contains an adjusted version of hmfc to be able to not estimate criterion fluctuations
(to create the plots shown in the paper). This is done in a rather hacky way so don't use this version of the model
to your own data. Instead, use hmfc.py. If you don't want to estimate criterion fluctuations, use mixed models.


# Simulating data with or without criterion fluctuations
- With:
    as usual (see hmfc.py)
- Without:
    _sample function returns zeros for the latent states + true model has hyperparameter for a and sigma_sq
    such that sampled values for both parameters are very close to zero.
    

# Estimating criterion fluctuations or not
- If estimated:
    as usual (see hmfc.py)
- If not estimated: 
    Init model has hyperparameters for a and sigma_sq (a0, nu_a, alpha, and beta) such that sampled values for both parameters are very close to zero.
    We don't update the a0, nu_a, alpha, and beta by removing these steps in the _gibbs_steps_global.
    In gibbs_step_states we don't estimate the latent state but instead return a sequence of zeros.
    
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from fastprogress import progress_bar
from jax import jit, lax, vmap
from jax.nn import sigmoid
from jaxtyping import Float, Array
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import pandas as pd
import numpy as np
import tensorflow as tf
import dill



class HierarchicalBernoulliLDS(eqx.Module):
    """
    Implementation of the model.
    """
    # Parameters of prior distributions
    w_0       : Float[Array, "num_inputs"]   # mean normal for input weights
    log_nu_w  : Float[Array, "num_inputs"]   # sd normal for input weights
    logit_a_0 : Float = 10.0                 # mean truncated normal for autoregressive coefficients (latent states) in unconstrained form (allows HMC)
    log_nu_a  : Float = -1.0                 # sd truncated normal for autoregressive coefficient (latent states) in unconstrained form
    log_alpha : Float = 1.0                  # shape of inverse gamma prior for sigmasq of latent states
    log_beta  : Float = 1.0                  # scale of inverse gamma prior for sigmasq of latent states

    def __init__(self, a_0, nu_a, w_0, nu_w, alpha, beta):
        self.logit_a_0 = jax.scipy.special.logit(a_0)
        self.log_nu_a = jnp.log(nu_a)
        self.w_0 = w_0
        self.log_nu_w = jnp.log(nu_w)
        self.log_alpha = jnp.log(alpha)
        self.log_beta = jnp.log(beta)

    def sample(self,
               key,
               inputs : Float[Array, "num_subjects num_trials num_inputs"],
               criterion_fluctuations = True
               ):
        """
        Draw a sample from the generative model.
        """
        num_subjects, num_trials, num_inputs = inputs.shape
        assert num_inputs == self.w_0.shape[0]

        def _sample_one(key, u_i):
            k1, k2, k3, k4, k5, k6 = jr.split(key, 6)

            # Sample per trial parameters
            a_i = tfd.TruncatedNormal(sigmoid(self.logit_a_0),
                                      jnp.exp(self.log_nu_a),
                                      0.0, 1.0).sample(seed=k1)

            w_i = tfd.Normal(self.w_0, jnp.exp(self.log_nu_w)).sample(seed=k2)

            sigmasq_i = tfd.InverseGamma(jnp.exp(self.log_alpha),
                                          jnp.exp(self.log_beta)).sample(seed=k3)

            # Sample latent states
            x_i = jnp.zeros(num_trials) # in case no criterion fluctuations are simulated (i.e. fixed criterion)

            if criterion_fluctuations: # if criterion fluctuations are simulated (i.e. fluctuations in criterion)
              x_i0 = tfd.Normal(0, 1.0).sample(seed=k4)
              def _step(x_it, key):
                  x_itp1 = tfd.Normal(a_i * x_it, jnp.sqrt(sigmasq_i)).sample(seed=key)
                  return x_itp1, x_it
              _, x_i = lax.scan(_step, x_i0, jr.split(k5, num_trials))

              # Ensure the latent states are centered
              x_i -= x_i.mean()

            # Sample emissions
            y_i = tfd.Bernoulli(x_i + u_i @ w_i).sample(seed=k6)
            return dict(a=a_i, w=w_i, sigmasq=sigmasq_i), x_i, y_i

        return vmap(_sample_one)(jr.split(key, num_subjects), inputs)

    def log_prob(self,
                 emissions : Float[Array, "num_subjects num_trials"],
                 masks : Float[Array, "num_subjects num_trials"],
                 states : Float[Array, "num_subjects num_trials"],
                 inputs : Float[Array, "num_subjects num_trials num_inputs"],
                 params : dict):
        lp = 0.0

        def _single_lp(y_i, m_i, x_i, u_i, params_i):
            a_i = params_i["a"]
            w_i = params_i["w"]
            sigma_i = jnp.sqrt(params_i["sigmasq"])

            # \log p(\theta_i | \eta)
            lp_i = tfd.TruncatedNormal(sigmoid(self.logit_a_0),
                                       jnp.exp(self.log_nu_a),
                                       0.0, 1.0).log_prob(a_i)
            lp_i += tfd.Normal(self.w_0, jnp.exp(self.log_nu_w)).log_prob(w_i)
            lp_i += tfd.InverseGamma(jnp.exp(self.log_alpha),
                                     jnp.exp(self.log_beta)).log_prob(sigma_i**2)

            # \log p(x_i | \theta_i)
            lp_i += tfd.Normal(0, 1.0).log_prob(x_i[0])
            lp_i += tfd.Normal(a_i * x_i[:-1], sigma_i).log_prob(x_i[1:]).sum()

            # \log p(y_i | x_i, u_i, \theta_i)
            lp_i += jnp.sum(m_i * tfd.Bernoulli(x_i + u_i @ w_i).log_prob(y_i)) # m_i is mask (if 0 then just sum 0)
            return lp_i

        return vmap(_single_lp)(emissions, masks, states, inputs, params).sum()


# Helper functions for information form message passing
from jax.scipy.linalg import solve_triangular

def lds_info_filter(J_diag, J_lower_diag, h):
    """ 
    Information form filtering for a linear Gaussian dynamical system.
    """
    # extract dimensions
    num_trials, dim, _ = J_diag.shape

    # Pad the L's with one extra set of zeros for the last predict step
    J_lower_diag_pad = jnp.concatenate((J_lower_diag, jnp.zeros((1, dim, dim))), axis=0)

    def marginalize(carry, t):
        Jp, hp, lp = carry

        # Condition
        Jc = J_diag[t] + Jp
        hc = h[t] + hp

        # Predict (using Cholesky)
        sqrt_Jc = jnp.linalg.cholesky(Jc)
        trm1 = solve_triangular(sqrt_Jc, hc, lower=True)
        trm2 = solve_triangular(sqrt_Jc, J_lower_diag_pad[t].T, lower=True)
        log_Z = 0.5 * dim * jnp.log(2 * jnp.pi)
        log_Z += -jnp.sum(jnp.log(jnp.diag(sqrt_Jc)))  # sum these terms only to get approx log|J|
        log_Z += 0.5 * jnp.dot(trm1.T, trm1)
        Jp = -jnp.dot(trm2.T, trm2)
        hp = -jnp.dot(trm2.T, trm1)

        # Alternative predict step:
        # log_Z = 0.5 * dim * np.log(2 * np.pi)
        # log_Z += -0.5 * np.linalg.slogdet(Jc)[1]
        # log_Z += 0.5 * np.dot(hc, np.linalg.solve(Jc, hc))
        # Jp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, J_lower_diag_pad[t].T))
        # hp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, hc))

        new_carry = Jp, hp, lp + log_Z
        return new_carry, (Jc, hc)

    # Initialize
    Jp0 = jnp.zeros((dim, dim))
    hp0 = jnp.zeros((dim,))
    (_, _, log_Z), (filtered_Js, filtered_hs) = \
        lax.scan(marginalize, (Jp0, hp0, 0), jnp.arange(num_trials))
    return log_Z, filtered_Js, filtered_hs


def _sample_info_gaussian(key, J, h, sample_shape=()):
    # TODO: avoid inversion.
    # see https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/util/stats.py#L117-L122
    # L = np.linalg.cholesky(J)
    # x = np.random.randn(h.shape[0])
    # return scipy.linalg.solve_triangular(L,x,lower=True,trans='T') \
    #     + dpotrs(L,h,lower=True)[0]
    cov = jnp.linalg.inv(J)
    loc = jnp.einsum("...ij,...j->...i", cov, h)
    return tfp.distributions.MultivariateNormalFullCovariance(
        loc=loc, covariance_matrix=cov).sample(sample_shape=sample_shape, seed=key)


def lds_info_sample(key, J_diag, J_lower_diag, h):
        log_Z, filtered_Js, filtered_hs = lds_info_filter(J_diag, J_lower_diag, h)

        def _step(carry, inpt):
            x_next = carry
            key, Jf, hf, L = inpt

            # Condition on the next observation
            Jc = Jf
            hc = hf - jnp.einsum('i,ij->j', x_next, L)

            # Split the key
            x = _sample_info_gaussian(key, Jc, hc)
            return x, x

        # Initialize with sample of last timestep and sample in reverse
        keys = jr.split(key, num_trials)
        x_T = _sample_info_gaussian(keys[-1], filtered_Js[-1], filtered_hs[-1])
        args = (keys[:-1], filtered_Js[:-1], filtered_hs[:-1], J_lower_diag)
        _, x = lax.scan(_step, x_T, args, reverse=True)

        # Append the last sample
        return jnp.vstack((x, x_T))


def gibbs_step_states_without_criterion_fluctuations_estimation(key,
                      emissions : Float[Array, "num_subjects num_trials"],
                      masks: Float[Array, "num_subjects num_trials"],
                      inputs : Float[Array, "num_subjects num_trials num_inputs"],
                      pg_samples : Float[Array, "num_subjects num_trials"],
                      params: dict):
    """
    Return a sequence of zeros as states.
    """
    N, T, D = inputs.shape
    def _sample_one(key, y_i, m_i, u_i, pg_i, params_i):
        x_i = jnp.zeros(num_trials)
        return x_i

    return vmap(_sample_one)(jr.split(key, num_subjects),
                             emissions,
                             masks,
                             inputs,
                             pg_samples,
                             params)


def gibbs_step_states_with_criterion_fluctuations_estimation(key,
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
        a_i = params_i["a"]
        w_i = params_i["w"]
        sigmasq_i = params_i["sigmasq"]

        # Compute the LDS natural params
        J_diag = (pg_i * m_i)                                   # (T,)
        J_diag = J_diag.at[0].add(1.0)
        J_diag = J_diag.at[:-1].add(a_i**2 / sigmasq_i)
        J_diag = J_diag.at[1:].add(1. / sigmasq_i)

        # lower diagonal blocks of precision matrix
        J_lower_diag = -a_i / sigmasq_i * jnp.ones(T - 1)       # (T-1,)

        # linear potential (precision-weighted mean h)
        h = (y_i - pg_i * (u_i @ w_i) - 0.5) * m_i              # (T,)

        # Run the information form sampling algorithm
        x_i = lds_info_sample(key,
                              J_diag[:, None, None],
                              J_lower_diag[:, None, None],
                              h[:, None])[:, 0]                 # (T,)

        # Enforce the mean-zero constraint
        # Note: this is a bit hacky. Really we should sample from the conditional
        # distribution given the mean-zero constraint, but that renders all the
        # time steps dependent. I don't know if there's an efficient way to
        # sample the joint distribution of x_i in less than O(T^3) time.
        return x_i - x_i.mean()

    return vmap(_sample_one)(jr.split(key, num_subjects),
                             emissions,
                             masks,
                             inputs,
                             pg_samples,
                             params)


"""
Sampling the parameters
"""
def gibbs_step_local_params(key,
                            emissions : Float[Array, "num_subjects num_trials"],
                            masks: Float[Array, "num_subjects num_trials"],
                            states: Float[Array, "num_subjects num_trials"],
                            inputs : Float[Array, "num_subjects num_trials num_inputs"],
                            pg_samples : Float[Array, "num_subjects num_trials"],
                            params: dict,
                            model : HierarchicalBernoulliLDS,
                            ):
    """
    Perform one Gibbs step to update the local parameters.
    """
    num_subjects, num_trials, num_inputs = inputs.shape
    a_0 = sigmoid(model.logit_a_0)
    nu_a = jnp.exp(model.log_nu_a)
    w_0 = model.w_0
    nu_w = jnp.exp(model.log_nu_w)

    def _sample_one(key, y_i, m_i, x_i, u_i, pg_i, params_i):
        k1, k2, k3 = jr.split(key, 3)

        # Gibbs sample the input weights
        J_w = 1.0 / nu_w**2 * jnp.eye(num_inputs)
        J_w += jnp.einsum('ti,tj,t,t->ij', u_i, u_i, m_i, pg_i)
        J_w = 0.5 * (J_w + J_w.T)
        h_w = w_0 / nu_w**2
        h_w += jnp.einsum('t,t,ti->i', y_i - pg_i * x_i - 0.5, m_i, u_i)
        w_i = _sample_info_gaussian(k1, J_w, h_w)

        # Gibbs sample the dynamics coefficient (given sigmasq_i and rest)
        sigmasq_i = params_i["sigmasq"]
        J_a = 1.0 / nu_a**2 + jnp.sum(m_i[:-1] * x_i[:-1]**2) / sigmasq_i
        h_a = a_0 / nu_a**2 + jnp.sum(m_i[:-1] * x_i[:-1] * x_i[1:]) / sigmasq_i
        a_i = tfd.TruncatedNormal(h_a / J_a, jnp.sqrt(1.0 / J_a), 0.0, 1.0).sample(seed=k2)

        # Gibbs sample the dynamics noise variance (given a_i and rest)
        alpha = jnp.exp(model.log_alpha) + 0.5 * jnp.sum(m_i)
        beta = jnp.exp(model.log_beta) + 0.5 * jnp.sum(m_i[1:] * (x_i[1:] - a_i * x_i[:-1])**2)
        sigmasq_i = tfd.InverseGamma(alpha, beta).sample(seed=k3)

        return dict(a=a_i, w=w_i, sigmasq=sigmasq_i)

    return vmap(_sample_one)(jr.split(key, num_subjects),
                             emissions,
                             masks,
                             states,
                             inputs,
                             pg_samples,
                             params)


"""
Sample the global parameters using Gibbs
"""
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
    """
    Update the global params w_0, nu_w with Gibbs
    """
    k1, k2 = jr.split(key)

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

    return model


def _gibbs_step_global_ar(key,
                          model : HierarchicalBernoulliLDS,
                          params : dict,
                          proposal_variance=0.05**2,
                          num_steps=20):
    """
    Update the global params a_0, nu_a with RWMH
    """
    def _log_prob(logit_a_0):
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, 1),
            tfb.Invert(tfb.Sigmoid()),
        ).log_prob(logit_a_0)

        lp += tfd.TruncatedNormal(sigmoid(logit_a_0), jnp.exp(model.log_nu_a),
                                  0.0, 1.0).log_prob(params["a"]).sum()
        return lp

    logit_a_0 = random_walk_mh(key,
                               _log_prob,
                               model.logit_a_0,
                               proposal_variance,
                               num_steps)

    model = eqx.tree_at(lambda m: m.logit_a_0, model, logit_a_0)

    return model


def _gibbs_step_global_ar_var(key,
                              model : HierarchicalBernoulliLDS,
                              params : dict,
                              proposal_variance=0.05**2,
                              num_steps=20,
                              max_nu_a=0.2):
    """
    Update the global params a_0, nu_a with RWMH
    """

    def _log_prob(log_nu_a):
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, max_nu_a),
            tfb.Log(),
        ).log_prob(log_nu_a)

        # log likelihood: \sum_i log p(a_i | a_0, \nu_a^2)
        lp += tfd.TruncatedNormal(sigmoid(model.logit_a_0), jnp.exp(log_nu_a),
                                  0.0, 1.0).log_prob(params["a"]).sum()
        return lp

    log_nu_a = random_walk_mh(key,
                              _log_prob,
                              model.log_nu_a,
                              proposal_variance,
                              num_steps)

    model = eqx.tree_at(lambda m: m.log_nu_a, model, log_nu_a)
    return model


def _gibbs_step_global_sigmasq_alpha(key,
                                     model: HierarchicalBernoulliLDS,
                                     params: dict,
                                     proposal_variance_alpha=0.1,
                                     num_steps_alpha=20):
    """
    Update alpha of inverse gamma for sigmasq with RWMH
    """

    def _log_prob_alpha(log_alpha):
        alpha = jnp.exp(log_alpha)
        beta = jnp.exp(model.log_beta)
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, 75),  # Adjust the upper bound as needed
            tfb.Log(),
        ).log_prob(log_alpha)

        lp += tfd.InverseGamma(alpha, beta).log_prob(params["sigmasq"]).sum()
        return lp

    log_alpha = random_walk_mh(key,
                               _log_prob_alpha,
                               model.log_alpha,
                               proposal_variance_alpha,
                               num_steps_alpha)

    model = eqx.tree_at(lambda m: m.log_alpha, model, log_alpha)
    return model


def _gibbs_step_global_sigmasq_beta(key,
                                    model: HierarchicalBernoulliLDS,
                                    params: dict,
                                    proposal_variance_beta=0.1,
                                    num_steps_beta=20):
    """
    Update beta of inverse gamma for sigmasq with RWMH
    """

    def _log_prob_beta(log_beta):
        alpha = jnp.exp(model.log_alpha)
        beta = jnp.exp(log_beta)
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, 50),  # Adjust the upper bound as needed
            tfb.Log(),
        ).log_prob(log_beta)

        lp += tfd.InverseGamma(alpha, beta).log_prob(params["sigmasq"]).sum()
        return lp

    log_beta = random_walk_mh(key,
                              _log_prob_beta,
                              model.log_beta,
                              proposal_variance_beta,
                              num_steps_beta)

    model = eqx.tree_at(lambda m: m.log_beta, model, log_beta)
    return model


def gibbs_step_global_params_with_criterion_fluctuations_estimation(key,
                                                       model : HierarchicalBernoulliLDS,
                                                       params : dict):
    k1, k2, k3, k4, k5 = jr.split(key, 5)
    model = _gibbs_step_global_weights(k1, model, params)
    model = _gibbs_step_global_ar(k2, model, params)
    model = _gibbs_step_global_ar_var(k3, model, params)
    model = _gibbs_step_global_sigmasq_alpha(k4, model, params)
    model = _gibbs_step_global_sigmasq_beta(k5, model, params)

    return model


def gibbs_step_global_params_without_criterion_fluctuations_estimation(key,
                                                          model : HierarchicalBernoulliLDS,
                                                          params : dict):

    model = _gibbs_step_global_weights(key, model, params)

    return model


def _pg_sample(key, b, c, trunc=200):
    '''pg(b,c) =
    1/(2pi)^2\sum_k=1^\infinity \dfrac{g_k}{(k-1/2)^2+c^2/(4pi^2)}
    where g_k ~ Ga(b,1)'''
    gammas = jr.gamma(key, b, shape=(trunc,))
    scaling = 1 / (4 * jnp.pi ** 2 * (jnp.arange(1, trunc + 1) - 1 / 2) ** 2 + c ** 2)
    pg = 2 * jnp.sum(gammas * scaling)
    return jnp.clip(pg, 1e-2, jnp.inf)


def gibbs_step_pg(key,
                  states: Float[Array, "num_subjects num_trials"],
                  inputs : Float[Array, "num_subjects num_trials num_inputs"],
                  params: dict,
                  ):

    num_subjects, num_trials, num_inputs = inputs.shape

    def _sample_one(key, x_i, u_i, w_i):
        psi_i = x_i + u_i @ w_i
        return vmap(_pg_sample)(jr.split(key, num_trials),
                                jnp.ones(num_trials),
                                psi_i)

    return vmap(_sample_one)(jr.split(key, num_subjects),
                             states,
                             inputs,
                             params["w"])


"""
Put all the steps together
"""

@jit
def gibbs_step_with_criterion_fluctuations_estimation(key,
               emissions : Float[Array, "num_subjects num_trials"],
               masks: Float[Array, "num_subjects num_trials"],
               states: Float[Array, "num_subjects num_trials"],
               inputs : Float[Array, "num_subjects num_trials num_inputs"],
               params: dict,
               model : HierarchicalBernoulliLDS):
    k1, k2, k3, k4 = jr.split(key, 4)

    # 0. Evaluate log joint probability
    lp = model.log_prob(emissions, masks, states, inputs, params)

    # 1. Sample PG auxiliary variables
    pg_samples = gibbs_step_pg(k1, states, inputs, params)

    # 2. Sample local params
    params = gibbs_step_local_params(k2, emissions, masks, states, inputs, pg_samples, params, model)

    # 3. Sample new latent states
    states = gibbs_step_states_with_criterion_fluctuations_estimation(k3, emissions, masks, inputs, pg_samples, params)

    # 4. Sample new global params (encapsulated in the model)
    model = gibbs_step_global_params_with_criterion_fluctuations_estimation(k4, model, params)

    return lp, states, params, model


@jit
def gibbs_step_without_criterion_fluctuations_estimation(key,
               emissions : Float[Array, "num_subjects num_trials"],
               masks: Float[Array, "num_subjects num_trials"],
               states: Float[Array, "num_subjects num_trials"],
               inputs : Float[Array, "num_subjects num_trials num_inputs"],
               params: dict,
               model : HierarchicalBernoulliLDS):
    k1, k2, k3, k4 = jr.split(key, 4)

    # 0. Evaluate log joint probability
    lp = model.log_prob(emissions, masks, states, inputs, params)

    # 1. Sample PG auxiliary variables
    pg_samples = gibbs_step_pg(k1, states, inputs, params)

    # 2. Sample local params
    params = gibbs_step_local_params(k2, emissions, masks, states, inputs, pg_samples, params, model)

    # 3. Sample new latent states
    states = gibbs_step_states_without_criterion_fluctuations_estimation(k3, emissions, masks, inputs, pg_samples, params)

    # 4. Sample new global params
    model = gibbs_step_global_params_without_criterion_fluctuations_estimation(k4, model, params)

    return lp, states, params, model




"""
Set seaborn style for plotting figures
"""

sns.set(style="ticks", context="paper",
        font="Arial",
        rc={"font.size": 24,
            "axes.titlesize": 23,
            "axes.labelsize": 23,
            "lines.linewidth": 1.5,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "axes.spines.right": False,
            "axes.spines.top": False
            })




"""
Illustration AR(1) parameters
"""

n_trials = 500

def generate_criterion_fluctuations(key, a, sigmasq):

    """
    Simulate criterion fluctuations as an AR(1) process
    """

    k1, k2 = jr.split(key, 2)

    x_i0 = tfd.Normal(0, 1.0).sample(seed=k1)
    def _step(x_it, key):
        x_itp1 = tfd.Normal(a * x_it, jnp.sqrt(sigmasq)).sample(seed=key)
        return x_itp1, x_it
    _, x_i = lax.scan(_step, x_i0, jr.split(k2, n_trials))

    return x_i


# Generate a dataset
key = jr.PRNGKey(0)

criterion_fluctuations_HH = generate_criterion_fluctuations(key, .99, .2)
criterion_fluctuations_HL = generate_criterion_fluctuations(key, .99, .05)
criterion_fluctuations_LH = generate_criterion_fluctuations(key, .90, .2)
criterion_fluctuations_LL = generate_criterion_fluctuations(key, .90, .05)

condition = np.repeat(["$a$ = 0.99, $\sigma^2$= 0.20", "$a$ = 0.99, $\sigma^2$= 0.05", "$a$ = 0.90, $\sigma^2$= 0.20", "$a$ = 0.90, $\sigma^2$= 0.05"], n_trials)
index = np.tile(np.arange(1, n_trials+1), 4) # repeat for number of conditions (4)
criterion_fluctuations = jnp.concatenate((criterion_fluctuations_HH, criterion_fluctuations_HL, criterion_fluctuations_LH, criterion_fluctuations_LL)) # make sure order is same as condition variable
df_model = pd.DataFrame({"index": index, "criterion_fluctuations": criterion_fluctuations, "condition": condition})


# Plot criterion trajectories
colors = ["#83aff0","#4779c4","#3c649f","#2c456b"]
plt.figure(figsize=(6, 6), dpi=600)
plot = sns.relplot(data=df_model, x="index", y="criterion_fluctuations", col="condition", col_wrap=2, kind="line", height=3, aspect=1.5)

for ax, color in zip(plot.axes.flatten(), colors):
    for line in ax.lines:
        line.set_color(color)

plot.set_axis_labels("Trial $t$", "Criterion $x_t$", size=18)
plot.set_titles(col_template="{col_name}", size=18, y=0.9)
plot.set(xticks=np.arange(0, 501, 100))
plot.set_xticklabels(np.arange(0, 501, 100), size=15)
plot.set(yticks=np.arange(-6, 6.01, 2))
plot.set_yticklabels(np.arange(-6, 6.01, 2), size=15)

plt.savefig("illustration_ar_paramss.png", bbox_inches='tight', dpi=600)




"""
Initialize global parameters before we start with correction of the three effects:
    1) underestimation d'
    2) underestimation psychometric slope
    3) apparent sequential effects
"""

key = jr.PRNGKey(1)

num_inputs = 2 # intercept and slope
num_subjects = 50
num_trials = 5000

w0 = jnp.array([0.0, 1.25]) # intercept and slope
nu_w = jnp.array([0.1, 0.1])


# Define true model
# H is high, L is low, first letter is for a, second letter for sigmasq
# so true_model_HH has high values for a and high values for sigmasq
true_model_fixed = HierarchicalBernoulliLDS(0.0, 1e-5, w0, nu_w, 10.0, 1e-5) # for fixed condition
true_model_HH = HierarchicalBernoulliLDS(0.98, 0.005, w0, nu_w, 50.0, 50.0 * 0.2) # for criterion fluctuations
true_model_HL = HierarchicalBernoulliLDS(0.98, 0.005, w0, nu_w, 50.0, 50.0 * 0.1) # for criterion fluctuations
true_model_LH = HierarchicalBernoulliLDS(0.95, 0.005, w0, nu_w, 50.0, 50.0 * 0.2) # for criterion fluctuations
true_model_LL = HierarchicalBernoulliLDS(0.95, 0.005, w0, nu_w, 50.0, 50.0 * 0.1) # for criterion fluctuations




"""
Criterion fluctuations and d'
"""

# Simulate data (with only 1 evidence level)
k1, k2 = jr.split(key)

inputs_dprime = tfd.Normal(0, 1).sample(sample_shape=(num_subjects, num_trials, num_inputs), seed=k1)
inputs_dprime = inputs_dprime.at[:, :, 0].set(1.0) # for intercept


# Sample params, states, and emissions from true model
true_params_fixed_dprime, true_states_fixed_dprime, emissions_fixed_dprime = true_model_fixed.sample(key, inputs_dprime, criterion_fluctuations=False) # fixed criterion of 0, no fluctuations
true_params_HH_dprime, true_states_HH_dprime, emissions_HH_dprime = true_model_HH.sample(key, inputs_dprime, criterion_fluctuations=True)
true_params_HL_dprime, true_states_HL_dprime, emissions_HL_dprime = true_model_HL.sample(key, inputs_dprime, criterion_fluctuations=True)
true_params_LH_dprime, true_states_LH_dprime, emissions_LH_dprime = true_model_LH.sample(key, inputs_dprime, criterion_fluctuations=True)
true_params_LL_dprime, true_states_LL_dprime, emissions_LL_dprime = true_model_LL.sample(key, inputs_dprime, criterion_fluctuations=True)




"""
# Calculate d prime and criterion using standard SDT calculation
"""

def dprime_fnc(resp, cresp, crit=False):

    """
    Function to estimate d'
    """

    n = resp.shape[0]

    hit = np.zeros(n)
    hit[(cresp == 1) & (resp == 1)] = 1

    fa = np.zeros(n)
    fa[(cresp == 0) & (resp == 1)] = 1

    phit = np.sum(hit) / np.sum(cresp)
    pfa = np.sum(fa) / np.sum(cresp == 0)

    phit = 0.01 if phit == 0 else 0.99 if phit == 1 else phit
    pfa = 0.01 if pfa == 0 else 0.99 if pfa == 1 else pfa

    if crit:
        return -0.5 * (norm.ppf(phit) + norm.ppf(pfa))
    else:
        return norm.ppf(phit) - norm.ppf(pfa)


def calculate_dprime_for_each_subject(inputs, emissions):

    """
    Calculate d prime and criterion for each subject in a dataset
    """

    all_d_prime = []
    all_c_prime = []

    for subj in range(num_subjects):
      cresp = np.where(inputs[subj,:,1] < 0, 0, 1) # check correct response by comparing evidence to 0
      resp = emissions[subj,:]

      all_d_prime.append(dprime_fnc(resp, cresp, crit=False))
      all_c_prime.append(dprime_fnc(resp, cresp, crit=True))

    return all_d_prime, all_c_prime


# Calculate d' and c'
# inputs are the same for all conditions
d_prime_fixed, c_prime_fixed = calculate_dprime_for_each_subject(inputs_dprime, emissions_fixed_dprime)
d_prime_LL, c_prime_LL = calculate_dprime_for_each_subject(inputs_dprime, emissions_LL_dprime)
d_prime_LH, c_prime_LH = calculate_dprime_for_each_subject(inputs_dprime, emissions_LH_dprime)
d_prime_HL, c_prime_HL = calculate_dprime_for_each_subject(inputs_dprime, emissions_HL_dprime)
d_prime_HH, c_prime_HH = calculate_dprime_for_each_subject(inputs_dprime, emissions_HH_dprime)


d_estimate = np.concatenate((d_prime_fixed, d_prime_LL, d_prime_LH, d_prime_HL, d_prime_HH))
c_estimate = np.concatenate((c_prime_fixed, c_prime_LL, c_prime_LH, c_prime_HL, c_prime_HH))
condition = np.repeat(["  $a$=0\n$\sigma^2$=0","  $a$=0.95\n$\sigma^2$=0.10", "  $a$=0.95\n$\sigma^2$=0.20", "  $a$=0.98\n$\sigma^2$=0.10", "  $a$=0.98\n$\sigma^2$=0.20"], num_subjects)

# Create a dataframe for plotting
df_dprime = pd.DataFrame({"d_estimate": d_estimate, "c_estimate": c_estimate, "condition": condition})




"""
# Plot d' over conditions
"""

# Note that generative d prime is absent since we simulate data from the model and not SDT directly
colors = ["black","#2c456b","#3c649f", "#4779c4", "#83aff0"]

plt.figure(figsize=(7, 6), dpi=600)
sns.stripplot(
    data=df_dprime, x="condition",hue="condition", y="d_estimate",
    alpha=.3, palette=colors
)
sns.pointplot(
    data=df_dprime, x="condition",hue="condition", y="d_estimate",
    linestyle="none", errorbar=None,
    markersize=6, markeredgewidth=3,palette=colors
)

plt.axhline(y=1.085, color='red', linestyle='--')
plt.xlabel('Condition')
plt.ylabel("Estimated d'")
plt.legend('', frameon=False)

plt.savefig("underestimation_dprimee.png", bbox_inches='tight', dpi=600)




"""
# Plot criterion over conditions
"""

colors = ["black","#2c456b","#3c649f", "#4779c4", "#83aff0"]

sns.stripplot(
    data=df_dprime, x="condition",hue="condition", y="c_estimate",
    alpha=.3, palette=colors
)
sns.pointplot(
    data=df_dprime, x="condition",hue="condition", y="c_estimate",
    linestyle="none", errorbar=None,
    markersize=6, markeredgewidth=3,palette=colors
)

plt.axhline(y=0, color='red', linestyle='--', label='y = 1')
plt.xlabel('Condition', size=18)
plt.ylabel("Estimated criterion", size=18)
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylim(-.3,.3)
plt.legend('', frameon=False)

plt.savefig("estimation_criterion.jpeg", bbox_inches='tight', dpi=600)




"""
Criterion fluctuations and psychometric slope
"""

# This is the data we'll fit with the model later on

# Simulate data (with multiple evidence levels for psychometric slope)
k1, k2, k3, k4, k5, k6, k7, k8, k9 = jr.split(key, 9)

inputs0 = tfd.Normal(-4, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k1)
inputs1 = tfd.Normal(-3, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k2)
inputs2 = tfd.Normal(-2, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k3)
inputs3 = tfd.Normal(-1, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k4)
inputs4 = tfd.Normal(0, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k5)
inputs5 = tfd.Normal(1, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k6)
inputs6 = tfd.Normal(2, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k7)
inputs7 = tfd.Normal(3, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k8)
inputs8 = tfd.Normal(4, 1).sample(sample_shape=(num_subjects, num_trials//9, num_inputs), seed=k9)

# Create stimulus category variable
stimulus_category = jnp.tile(jnp.repeat(jnp.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]), num_trials//9), num_subjects)

stimulus_category = stimulus_category.reshape([num_subjects,(num_trials//9)*9])

merged_inputs = jnp.array(tf.concat([inputs0, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8], axis=1))
inputs = merged_inputs.at[:, :, 0].set(1.0) # for intercept

# Shuffle order (both are shuffled in the same way)
stimulus_category = jr.permutation(jr.PRNGKey(0), stimulus_category, axis=1)
inputs = jr.permutation(jr.PRNGKey(0), inputs, axis=1)


# Sample params, states, and emissions from true model
true_params_fixed, true_states_fixed, emissions_fixed = true_model_fixed.sample(key, inputs, criterion_fluctuations=False) # for fixed condition
true_params_HH, true_states_HH, emissions_HH = true_model_HH.sample(key, inputs, criterion_fluctuations=True)
true_params_HL, true_states_HL, emissions_HL = true_model_HL.sample(key, inputs, criterion_fluctuations=True)
true_params_LH, true_states_LH, emissions_LH = true_model_LH.sample(key, inputs, criterion_fluctuations=True)
true_params_LL, true_states_LL, emissions_LL = true_model_LL.sample(key, inputs, criterion_fluctuations=True)



def calculation_psychometric(emissions):
    """
    Calculate the mean response for each stimulus category within each subject.
    Then take the average over subjects to obtain the mean response for the
    psychometric function.
    """
    mean_resp_all_subj = []
    for subj in range(num_subjects):
        temp_df = pd.DataFrame({"resp" : emissions[subj],"stimulus_category": stimulus_category[subj], "evidence": inputs[:,:,1][subj]})

        mean_resp = temp_df.groupby("stimulus_category")["resp"].mean().reset_index()
        mean_resp_all_subj.append(mean_resp["resp"])

    mean_resp_over_subj = np.mean(mean_resp_all_subj, axis=0) # take average p(resp) over subjects for each evidence level

    return mean_resp_over_subj



# Create dataframe for plotting
mean_resp_fixed = calculation_psychometric(emissions_fixed)
mean_resp_LL = calculation_psychometric(emissions_LL)
mean_resp_LH = calculation_psychometric(emissions_LH)
mean_resp_HL = calculation_psychometric(emissions_HL)
mean_resp_HH = calculation_psychometric(emissions_HH)

mean_resp = np.concatenate((mean_resp_fixed, mean_resp_LL, mean_resp_LH, mean_resp_HL, mean_resp_HH))

evidence = np.tile([-4, -3, -2, -1, 0, 1, 2, 3, 4], 5)  # Repeat number of parameter combinations (5)
condition = np.repeat([
    "$a$=0, $\sigma^2$=0",
    "$a$=0.95, $\sigma^2$=0.10",
    "$a$=0.95, $\sigma^2$=0.20",
    "$a$=0.98, $\sigma^2$=0.10",
    "$a$=0.98, $\sigma^2$=0.20"], 9)  # Repeat for the number of evidence levels (9)

df_psychometric = pd.DataFrame({"evidence": evidence, "condition": condition, "mean_resp": mean_resp})
df_psychometric["condition"] = pd.Categorical(df_psychometric["condition"], categories=[
    "$a$=0.98, $\sigma^2$=0.20",
    "$a$=0.98, $\sigma^2$=0.10",
    "$a$=0.95, $\sigma^2$=0.20",
    "$a$=0.95, $\sigma^2$=0.10",
    "$a$=0, $\sigma^2$=0"])


# Plotting
colors = ["#83aff0","#4779c4","#3c649f","#2c456b","black"]
plt.figure(figsize=(8, 6), dpi=600)
ax = df_psychometric.groupby(["evidence", "condition"])["mean_resp"].mean().unstack().plot(
    style='o-', markerfacecolor='white', markersize=5, color=colors)

ax.set_xlim(-4.5, 4.5)
ax.set_xlabel("Signed evidence", size=18)
ax.set_ylabel("P(response A)", size=18)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

# Reversing the order of legend labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), title="Condition", loc="upper left", fontsize=10)
ax.grid(False)


plt.savefig("underestimation_psychometric_slopee.png", bbox_inches='tight', dpi=600)



# Plotting (zoomed in psychometric function)
colors = ["#83aff0","#4779c4","#3c649f","#2c456b","black"]

p_psychometric = (
    df_psychometric.groupby(["evidence", "condition"])["mean_resp"]
    .mean()
    .unstack()
    .plot(style='o-', markerfacecolor='white', markersize=10, linewidth=2, color=colors)
)

p_psychometric.set_xlim(1.5, 2.5)
p_psychometric.set_ylim(.7, .95)

p_psychometric.set_xlabel("Signed evidence", size=20)
p_psychometric.set_ylabel("P(response A)", size=25)
p_psychometric.tick_params(axis='x', labelsize=17)
p_psychometric.tick_params(axis='y', labelsize=23)
p_psychometric.legend([]) # drop legend

p_psychometric.grid(False)

plt.savefig("underestimation_psychometric_slope_zoomed_in.png", bbox_inches='tight', dpi=600)




"""
Criterion fluctuations and P(repeat response)
"""

# Add previous response to input variable
def add_prev_resp(inputs, emissions):
    # Create previous response variable by shifting responses (emissions)
    # Add a 0 as the first element in each row (could be better but adding a 0 for everybody with 5000 trials won't matter much)
    prev_resp = jnp.insert(emissions, 0, 0, axis=1)

    # Delete the last element in each row
    prev_resp = jnp.delete(prev_resp, -1, axis=1)
    prev_resp = jnp.where(prev_resp == 0, -1, 1) # change 0 to -1

    # Add prev resp to inputs
    inputs_with_prevresp = jnp.concatenate((inputs, prev_resp[..., jnp.newaxis]), axis=2)

    return inputs_with_prevresp


inputs_fixed = add_prev_resp(inputs, emissions_fixed) # structure output: [intercept, slope evidence, previous response]
inputs_HH = add_prev_resp(inputs, emissions_HH)
inputs_HL = add_prev_resp(inputs, emissions_HL)
inputs_LH = add_prev_resp(inputs, emissions_LH)
inputs_LL = add_prev_resp(inputs, emissions_LL)



def calculate_p_repeat(inputs, emissions):
  # Check whether previous response == current response
  repeat_response = inputs[:,:,2] == jnp.where(emissions == 0, -1, 1)

  # Sum and divide by num_trials to get probability of repeating a response
  p_repeat_response = repeat_response.sum(axis=1)/num_trials

  # Calculate the average probability over subjects
  mean = jnp.mean(p_repeat_response)

  return p_repeat_response, mean


# Calculate P(repeat response) for every condition
p_rep_HH, mean_p_rep_HH = calculate_p_repeat(inputs_HH, emissions_HH)
p_rep_HL, mean_p_rep_HL = calculate_p_repeat(inputs_HL, emissions_HL)
p_rep_LH, mean_p_rep_LH = calculate_p_repeat(inputs_LH, emissions_LH)
p_rep_LL, mean_p_rep_LL = calculate_p_repeat(inputs_LL, emissions_LL)
p_rep_fixed, mean_p_rep_fixed = calculate_p_repeat(inputs_fixed, emissions_fixed)



# Create a dataframe for plotting
p_repeat = np.concatenate((p_rep_fixed, p_rep_LL, p_rep_LH, p_rep_HL, p_rep_HH))
condition = np.repeat([" $a$ = 0\n$\sigma^2$= 0"," $a$ = 0.95\n$\sigma^2$= 0.10", " $a$ = 0.95\n$\sigma^2$= 0.20", " $a$ = 0.98\n$\sigma^2$= 0.10", " $a$ = 0.98\n$\sigma^2$= 0.20"], num_subjects)

df_sequential = pd.DataFrame({"p_repeat": p_repeat, "condition": condition})


# Plotting
colors = ["black","#2c456b","#3c649f", "#4779c4", "#83aff0"]
plt.figure(figsize=(8, 6), dpi=600)
sns.stripplot(
    data=df_sequential, x="condition",hue="condition", y="p_repeat",
    alpha=.3, palette=colors
)
sns.pointplot(
    data=df_sequential, x="condition",hue="condition", y="p_repeat",
    linestyle="none", errorbar=None,
    markersize=6, markeredgewidth=3,palette=colors
)


plt.axhline(y=.5, color='red', linestyle='--', label='y = 1')
plt.xlabel('Condition', fontsize=18)
plt.ylabel("P(repeat response)", fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.legend('', frameon=False)

plt.savefig("repetition_effectss.png", bbox_inches='tight', dpi=600)




"""
Fit model
"""

# criterion_fluctuations_estimated = True: criterion fluctuations are estimated -> hyperparams for a and sigmasq are estimated and initialized
# criterion_fluctuations_estimated = False: criterion fluctuations are NOT estimated -> hyperparams for a and sigmasq are not estimated and set to close to zero

# Create mask variable
masks = jnp.ones_like(emissions_fixed) # is same for all conditions

# Update num_inputs since prev_resp is added as additional variable to input
num_inputs_with_prevresp = num_inputs + 1

def fit_model(inputs, emissions, key, criterion_fluctuations_estimated=True):

    # Initial values for hyperparameters
    init_nu_w = jnp.repeat(0.1, num_inputs_with_prevresp)
    init_w = jnp.zeros(num_inputs_with_prevresp)

    # If no criterion fluctuations are estimated
    model = HierarchicalBernoulliLDS(0.0, 1e-4, init_w, init_nu_w, 10.0, 1e-4)

    # If criterion fluctuations are estimated
    if criterion_fluctuations_estimated:
      model = HierarchicalBernoulliLDS(0.90, 0.1, init_w, init_nu_w, 20.0, 20.0 * 0.2**2)

    params, states, _ = model.sample(key, inputs)
    masks = jnp.ones_like(emissions)


    # Fit model
    lps = []

    posterior_samples_a0 = []
    posterior_samples_nu_a0 = []
    posterior_samples_w0 = []
    posterior_samples_nu_w0 = []
    posterior_samples_alpha = []
    posterior_samples_beta = []
        
    posterior_samples_w = []
    posterior_samples_a = []
    posterior_samples_sigmasq = []
    
    posterior_samples_states = []
    posterior_samples_states_squared = []


    for itr in progress_bar(range(num_iters)):
        print(itr)
        this_key, key = jr.split(key)

        if criterion_fluctuations_estimated:
          lp, states, params, model = gibbs_step_with_criterion_fluctuations_estimation(this_key, emissions, masks, states, inputs, params, model)
        else:
          lp, states, params, model = gibbs_step_without_criterion_fluctuations_estimation(this_key, emissions, masks, states, inputs, params, model)

        lps.append(lp)

        posterior_samples_a0.append(sigmoid(model.logit_a_0))
        posterior_samples_nu_a0.append(jnp.exp(model.log_nu_a))
        posterior_samples_w0.append(model.w_0)
        posterior_samples_nu_w0.append(jnp.exp(model.log_nu_w))
        posterior_samples_alpha.append(jnp.exp(model.log_alpha))
        posterior_samples_beta.append(jnp.exp(model.log_beta))
    
        posterior_samples_w.append(params['w'])
        posterior_samples_a.append(params['a'])
        posterior_samples_sigmasq.append(params['sigmasq'])
            
        if itr == burn_in:
            states_sum = states
            states_sum_squared = states
        elif itr > burn_in:
            states_sum += states
            states_sum_squared += states**2 # for calculation standard deviation later
            


    lps = jnp.stack(lps)
    
    posterior_samples_a0 = jnp.stack(posterior_samples_a0)
    posterior_samples_nu_a0 = jnp.stack(posterior_samples_nu_a0)
    posterior_samples_w0 = jnp.stack(posterior_samples_w0)
    posterior_samples_nu_w0 = jnp.stack(posterior_samples_nu_w0)
    posterior_samples_alpha = jnp.stack(posterior_samples_alpha)
    posterior_samples_beta = jnp.stack(posterior_samples_beta)
    
    posterior_samples_w = jnp.stack(posterior_samples_w)
    posterior_samples_a = jnp.stack(posterior_samples_a)
    posterior_samples_sigmasq = jnp.stack(posterior_samples_sigmasq)
    
    posterior_samples_states = jnp.stack(states_sum) 
    posterior_samples_states_squared = jnp.stack(states_sum_squared)
    

    return lps, states, params, posterior_samples_a0, posterior_samples_nu_a0, posterior_samples_w0, posterior_samples_nu_w0, posterior_samples_alpha, posterior_samples_beta, posterior_samples_w, posterior_samples_a, posterior_samples_sigmasq, posterior_samples_states, posterior_samples_states_squared

num_trials = (num_trials//9)*9 # 5000 trials can't be divided by 9

burn_in = 250
num_iters = 1250

key = jr.PRNGKey(0)

lps_fixed, inf_states_fixed, inf_params_fixed, posterior_a0_fixed, posterior_nu_a0_fixed, posterior_w_0_fixed, posterior_nu_w0_fixed, posterior_alpha_fixed, posterior_beta_fixed, posterior_w_fixed, posterior_a_fixed, posterior_sigmasq_fixed, posterior_states_fixed, posterior_states_squared_fixed = fit_model(inputs_fixed, emissions_fixed, key, criterion_fluctuations_estimated=True)
lps_LL, inf_states_LL, inf_params_LL, posterior_a0_LL, posterior_nu_a0_LL, posterior_w0_LL, posterior_nu_w0_LL, posterior_alpha_LL, posterior_beta_LL, posterior_w_LL, posterior_a_LL, posterior_sigmasq_LL, posterior_states_LL, posterior_states_squared_LL = fit_model(inputs_LL, emissions_LL, key, criterion_fluctuations_estimated=True)
lps_LH, inf_states_LH, inf_params_LH, posterior_a0_LH, posterior_nu_a0_LH, posterior_w0_LH, posterior_nu_w0_LH, posterior_alpha_LH, posterior_beta_LH, posterior_w_LH, posterior_a_LH, posterior_sigmasq_LH, posterior_states_LH, posterior_states_squared_LH = fit_model(inputs_LH, emissions_LH, key, criterion_fluctuations_estimated=True)
lps_HL, inf_states_HL, inf_params_HL, posterior_a0_HL, posterior_nu_a0_HL, posterior_w0_HL, posterior_nu_w0_HL, posterior_alpha_HL, posterior_beta_HL, posterior_w_HL, posterior_a_HL, posterior_sigmasq_HL, posterior_states_HL, posterior_states_squared_HL = fit_model(inputs_HL, emissions_HL, key, criterion_fluctuations_estimated=True)
lps_HH, inf_states_HH, inf_params_HH, posterior_a0_HH, posterior_nu_a0_HH, posterior_w0_HH, posterior_nu_w0_HH, posterior_alpha_HH, posterior_beta_HH, posterior_w_HH, posterior_a_HH, posterior_sigmasq_HH, posterior_states_HH, posterior_states_squared_HH = fit_model(inputs_HH, emissions_HH, key, criterion_fluctuations_estimated=True)

lps_fixed_no_est_CF, inf_states_fixed_no_est_CF, inf_params_fixed_no_est_CF, posterior_a0_fixed_no_est_CF, posterior_nu_a0_fixed_no_est_CF, posterior_w0_fixed_no_est_CF, posterior_nu_w0_fixed_no_est_CF, posterior_alpha_fixed_no_est_CF, posterior_beta_fixed_no_est_CF, posterior_w_fixed_no_est_CF, posterior_a_fixed_no_est_CF, posterior_sigmasq_fixed_no_est_CF, posterior_states_fixed_no_est_CF, posterior_states_squared_fixed_no_est_CF = fit_model(inputs_fixed, emissions_fixed, key, criterion_fluctuations_estimated=False)
lps_LL_no_est_CF, inf_states_LL_no_est_CF, inf_params_LL_no_est_CF, posterior_a0_LL_no_est_CF, posterior_nu_a0_LL_no_est_CF, posterior_w0_LL_no_est_CF, posterior_nu_w0_LL_no_est_CF, posterior_alpha_LL_no_est_CF, posterior_beta_LL_no_est_CF, posterior_w_LL_no_est_CF, posterior_a_LL_no_est_CF, posterior_sigmasq_LL_no_est_CF, posterior_states_LL_no_est_CF, posterior_states_squared_LL_no_est_CF = fit_model(inputs_LL, emissions_LL, key, criterion_fluctuations_estimated=False)
lps_LH_no_est_CF, inf_states_LH_no_est_CF, inf_params_LH_no_est_CF, posterior_a0_LH_no_est_CF, posterior_nu_a0_LH_no_est_CF, posterior_w0_LH_no_est_CF, posterior_nu_w0_LH_no_est_CF, posterior_alpha_LH_no_est_CF, posterior_beta_LH_no_est_CF, posterior_w_LH_no_est_CF, posterior_a_LH_no_est_CF, posterior_sigmasq_LH_no_est_CF, posterior_states_LH_no_est_CF, posterior_states_squared_LH_no_est_CF = fit_model(inputs_LH, emissions_LH, key, criterion_fluctuations_estimated=False)
lps_HL_no_est_CF, inf_states_HL_no_est_CF, inf_params_HL_no_est_CF, posterior_a0_HL_no_est_CF, posterior_nu_a0_HL_no_est_CF, posterior_w0_HL_no_est_CF, posterior_nu_w0_HL_no_est_CF, posterior_alpha_HL_no_est_CF, posterior_beta_HL_no_est_CF, posterior_w_HL_no_est_CF, posterior_a_HL_no_est_CF, posterior_sigmasq_HL_no_est_CF, posterior_states_HL_no_est_CF, posterior_states_squared_HL_no_est_CF = fit_model(inputs_HL, emissions_HL, key, criterion_fluctuations_estimated=False)
lps_HH_no_est_CF, inf_states_HH_no_est_CF, inf_params_HH_no_est_CF, posterior_a0_HH_no_est_CF, posterior_nu_a0_HH_no_est_CF, posterior_w0_HH_no_est_CF, posterior_nu_w0_HH_no_est_CF, posterior_alpha_HH_no_est_CF, posterior_beta_HH_no_est_CF, posterior_w_HH_no_est_CF, posterior_a_HH_no_est_CF, posterior_sigmasq_HH_no_est_CF, posterior_states_HH_no_est_CF, posterior_states_squared_HH_no_est_CF = fit_model(inputs_HH, emissions_HH, key, criterion_fluctuations_estimated=False)




"""
Save environment (if ran on a cluster computer)
"""

file_name = '/vsc-hard-mounts/leuven-data/343/vsc34314/correction_effects/data_correction_effects.dil'
list_of_variable_names = (
    "lps_fixed", "inf_states_fixed", "inf_params_fixed", "posterior_a0_fixed", "posterior_nu_a0_fixed", "posterior_w_0_fixed", "posterior_nu_w0_fixed", "posterior_alpha_fixed", "posterior_beta_fixed", "posterior_w_fixed", "posterior_a_fixed", "posterior_sigmasq_fixed", "posterior_states_fixed", "posterior_states_squared_fixed",
    "lps_LL", "inf_states_LL", "inf_params_LL", "posterior_a0_LL", "posterior_nu_a0_LL", "posterior_w0_LL", "posterior_nu_w0_LL", "posterior_alpha_LL", "posterior_beta_LL", "posterior_w_LL", "posterior_a_LL", "posterior_sigmasq_LL", "posterior_states_LL", "posterior_states_squared_LL",
    "lps_LH", "inf_states_LH", "inf_params_LH", "posterior_a0_LH", "posterior_nu_a0_LH", "posterior_w0_LH", "posterior_nu_w0_LH", "posterior_alpha_LH", "posterior_beta_LH", "posterior_w_LH", "posterior_a_LH", "posterior_sigmasq_LH", "posterior_states_LH", "posterior_states_squared_LH",
    "lps_HL", "inf_states_HL", "inf_params_HL", "posterior_a0_HL", "posterior_nu_a0_HL", "posterior_w0_HL", "posterior_nu_w0_HL", "posterior_alpha_HL", "posterior_beta_HL", "posterior_w_HL", "posterior_a_HL", "posterior_sigmasq_HL", "posterior_states_HL", "posterior_states_squared_HL",
    "lps_HH", "inf_states_HH", "inf_params_HH", "posterior_a0_HH", "posterior_nu_a0_HH", "posterior_w0_HH", "posterior_nu_w0_HH", "posterior_alpha_HH", "posterior_beta_HH", "posterior_w_HH", "posterior_a_HH", "posterior_sigmasq_HH", "posterior_states_HH", "posterior_states_squared_HH",
    "lps_fixed_no_est_CF", "inf_states_fixed_no_est_CF", "inf_params_fixed_no_est_CF", "posterior_a0_fixed_no_est_CF", "posterior_nu_a0_fixed_no_est_CF", "posterior_w0_fixed_no_est_CF", "posterior_nu_w0_fixed_no_est_CF", "posterior_alpha_fixed_no_est_CF", "posterior_beta_fixed_no_est_CF", "posterior_w_fixed_no_est_CF", "posterior_a_fixed_no_est_CF", "posterior_sigmasq_fixed_no_est_CF", "posterior_states_fixed_no_est_CF", "posterior_states_squared_fixed_no_est_CF",
    "lps_LL_no_est_CF", "inf_states_LL_no_est_CF", "inf_params_LL_no_est_CF", "posterior_a0_LL_no_est_CF", "posterior_nu_a0_LL_no_est_CF", "posterior_w0_LL_no_est_CF", "posterior_nu_w0_LL_no_est_CF", "posterior_alpha_LL_no_est_CF", "posterior_beta_LL_no_est_CF", "posterior_w_LL_no_est_CF", "posterior_a_LL_no_est_CF", "posterior_sigmasq_LL_no_est_CF", "posterior_states_LL_no_est_CF", "posterior_states_squared_LL_no_est_CF",
    "lps_LH_no_est_CF", "inf_states_LH_no_est_CF", "inf_params_LH_no_est_CF", "posterior_a0_LH_no_est_CF", "posterior_nu_a0_LH_no_est_CF", "posterior_w0_LH_no_est_CF", "posterior_nu_w0_LH_no_est_CF", "posterior_alpha_LH_no_est_CF", "posterior_beta_LH_no_est_CF", "posterior_w_LH_no_est_CF", "posterior_a_LH_no_est_CF", "posterior_sigmasq_LH_no_est_CF", "posterior_states_LH_no_est_CF", "posterior_states_squared_LH_no_est_CF",
    "lps_HL_no_est_CF", "inf_states_HL_no_est_CF", "inf_params_HL_no_est_CF", "posterior_a0_HL_no_est_CF", "posterior_nu_a0_HL_no_est_CF", "posterior_w0_HL_no_est_CF", "posterior_nu_w0_HL_no_est_CF", "posterior_alpha_HL_no_est_CF", "posterior_beta_HL_no_est_CF", "posterior_w_HL_no_est_CF", "posterior_a_HL_no_est_CF", "posterior_sigmasq_HL_no_est_CF", "posterior_states_HL_no_est_CF", "posterior_states_squared_HL_no_est_CF",
    "lps_HH_no_est_CF", "inf_states_HH_no_est_CF", "inf_params_HH_no_est_CF", "posterior_a0_HH_no_est_CF", "posterior_nu_a0_HH_no_est_CF", "posterior_w0_HH_no_est_CF", "posterior_nu_w0_HH_no_est_CF", "posterior_alpha_HH_no_est_CF", "posterior_beta_HH_no_est_CF", "posterior_w_HH_no_est_CF", "posterior_a_HH_no_est_CF", "posterior_sigmasq_HH_no_est_CF", "posterior_states_HH_no_est_CF", "posterior_states_squared_HH_no_est_CF",
    "true_params_fixed", "true_states_fixed", "emissions_fixed", "inputs_fixed",
    "true_params_HH", "true_states_HH", "emissions_HH", "inputs_HH",
    "true_params_HL", "true_states_HL", "emissions_HL", "inputs_HL",
    "true_params_LH", "true_states_LH", "emissions_LH", "inputs_LH",
    "true_params_LL", "true_states_LL", "emissions_LL", "inputs_LL",
    "num_iters", "num_subjects", "num_trials", "w0", "nu_w"
    )

with open(file_name, 'wb') as file:
    dill.dump(list_of_variable_names, file)  # Store all the names first
    for variable_name in list_of_variable_names:
        dill.dump(eval(variable_name), file) # Store the objects themselves



"""
Load in environment (if ran on a cluster computer)
"""

file_name = "C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/hLDS/hLDS Gibbs/Correction effects/data_correction_effects.dil"
g = globals()
with open(file_name,'rb') as file:
    list_of_variable_names = dill.load(file)  # Get the names of stored objects
    for variable_name in list_of_variable_names:
        g[variable_name] = dill.load(file)    # Get the objects themselves




"""
Create dataframe for plotting posteriors
"""
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)

burn_in = 750  # high burn_in but otherwise weird shaped posteriors for prev. resp (doesn't change the story, only gives a nicer plot)

posterior_evidence_CF = np.concatenate((posterior_w_0_fixed[:,1][burn_in:], posterior_w0_LL[:,1][burn_in:], posterior_w0_LH[:,1][burn_in:], posterior_w0_HL[:,1][burn_in:], posterior_w0_HH[:,1][burn_in:]))
posterior_evidence_noCF = np.concatenate((posterior_w0_fixed_no_est_CF[:,1][burn_in:], posterior_w0_LL_no_est_CF[:,1][burn_in:], posterior_w0_LH_no_est_CF[:,1][burn_in:], posterior_w0_HL_no_est_CF[:,1][burn_in:], posterior_w0_HH_no_est_CF[:,1][burn_in:]))
posterior_evidence = np.concatenate((posterior_evidence_CF,posterior_evidence_noCF))

posterior_prevresp_CF = np.concatenate((posterior_w_0_fixed[:,2][burn_in:], posterior_w0_LL[:,2][burn_in:], posterior_w0_LH[:,2][burn_in:], posterior_w0_HL[:,2][burn_in:], posterior_w0_HH[:,2][burn_in:]))
posterior_prevresp_noCF = np.concatenate((posterior_w0_fixed_no_est_CF[:,2][burn_in:], posterior_w0_LL_no_est_CF[:,2][burn_in:], posterior_w0_LH_no_est_CF[:,2][burn_in:], posterior_w0_HL_no_est_CF[:,2][burn_in:], posterior_w0_HH_no_est_CF[:,2][burn_in:]))
posterior_prevresp = np.concatenate((posterior_prevresp_CF,posterior_prevresp_noCF))



condition = np.tile(np.repeat([" $a$ = 0\n$\sigma^2$= 0"," $a$ = 0.95\n$\sigma^2$= 0.10", " $a$ = 0.95\n$\sigma^2$= 0.20", " $a$ = 0.98\n$\sigma^2$= 0.10", " $a$ = 0.98\n$\sigma^2$= 0.20"], len(posterior_w_0_fixed[:,1][burn_in:])),2)
criterion_estimated = np.repeat(["Criterion estimated","Criterion not estimated"], 5*len(posterior_w_0_fixed[:,1][burn_in:]))

df_posterior = pd.DataFrame({"posterior_evidence": posterior_evidence, "posterior_prevresp": posterior_prevresp, "condition": condition, "criterion_estimated" : criterion_estimated})




""" 
Plotting posteriors psychometric slope 
"""

plt.figure(dpi=600)
sns.violinplot(data=df_posterior, x="condition", y="posterior_evidence", hue="criterion_estimated",
               split=True, inner=None)
plt.axhline(y=1.25, color='red', linestyle='--')
plt.xlabel('Condition', fontsize=15)
plt.ylabel(r"$\beta_{stimulus}$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(title='',loc='lower left')
plt.show()




""" 
Plotting posteriors previous response 
"""

plt.figure(dpi=600)
plt.axhline(y=0, color='red', linestyle='--')
sns.violinplot(data=df_posterior, x="condition", y="posterior_prevresp", hue="criterion_estimated",
               split=True, inner=None, fill=True)
plt.xlabel('Condition', fontsize=15)
plt.ylabel(r"$\beta_{previous \,response}$", fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(title='',loc='upper left')
plt.show()




"""
Create dataframe 
"""
subject_est_evidence_CF = np.concatenate((jnp.mean(posterior_w_fixed[:burn_in,:,1], axis=0), jnp.mean(posterior_w_LL[:burn_in,:,1], axis=0), jnp.mean(posterior_w_LH[:burn_in,:,1], axis=0), jnp.mean(posterior_w_HL[:burn_in,:,1], axis=0), jnp.mean(posterior_w_HH[:burn_in,:,1], axis=0)))
subject_est_evidence_noCF = np.concatenate((jnp.mean(posterior_w_fixed_no_est_CF[:burn_in,:,1], axis=0), jnp.mean(posterior_w_LL_no_est_CF[:burn_in,:,1], axis=0), jnp.mean(posterior_w_LH_no_est_CF[:burn_in,:,1], axis=0), jnp.mean(posterior_w_HL_no_est_CF[:burn_in,:,1], axis=0), jnp.mean(posterior_w_HH_no_est_CF[:burn_in,:,1], axis=0)))
subject_est_evidence = np.concatenate((subject_est_evidence_CF,subject_est_evidence_noCF))

subject_est_prevresp_CF = np.concatenate((jnp.mean(posterior_w_fixed[:burn_in,:,2], axis=0), jnp.mean(posterior_w_LL[:burn_in,:,2], axis=0), jnp.mean(posterior_w_LH[:burn_in,:,2], axis=0), jnp.mean(posterior_w_HL[:burn_in,:,2], axis=0), jnp.mean(posterior_w_HH[:burn_in,:,2], axis=0)))
subject_est_prevresp_noCF = np.concatenate((jnp.mean(posterior_w_fixed_no_est_CF[:burn_in,:,2], axis=0), jnp.mean(posterior_w_LL_no_est_CF[:burn_in,:,2], axis=0), jnp.mean(posterior_w_LH_no_est_CF[:burn_in,:,2], axis=0), jnp.mean(posterior_w_HL_no_est_CF[:burn_in,:,2], axis=0), jnp.mean(posterior_w_HH_no_est_CF[:burn_in,:,2], axis=0)))
subject_est_prevresp = np.concatenate((subject_est_prevresp_CF,subject_est_prevresp_noCF))

condition = np.tile(np.repeat([" $a$ = 0\n$\sigma^2$= 0"," $a$ = 0.95\n$\sigma^2$= 0.10", " $a$ = 0.95\n$\sigma^2$= 0.20", " $a$ = 0.98\n$\sigma^2$= 0.10", " $a$ = 0.98\n$\sigma^2$= 0.20"], len(jnp.mean(posterior_w_fixed[:burn_in,:,1], axis=0))),2)
criterion_estimated = np.repeat(["Criterion estimated","Criterion not estimated"], 5*len(jnp.mean(posterior_w_fixed[:burn_in,:,1], axis=0)))

df_subject_est = pd.DataFrame({"subject_est_evidence": subject_est_evidence, "subject_est_prevresp": subject_est_prevresp, "condition": condition, "criterion_estimated" : criterion_estimated})




"""
Plotting individual estimates for slope (can't get the same color mapping as figure 1c)
"""

colors = ["black","#2c456b","#3c649f", "#4779c4", "#83aff0"]

sns.stripplot(
    data=df_subject_est, x="condition", hue="criterion_estimated", y="subject_est_evidence",
    alpha=.3, palette=colors, dodge=True
)

plt.axhline(y=1.25, color='red', linestyle='--')
plt.xlabel('Condition', fontsize=15)
plt.ylabel(r"$\beta_{stimulus}$", fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.legend(title='',loc='lower left')
plt.show()



"""
Plotting individual estimates for previous response (can't get the same color mapping as figure 1c)
"""

sns.stripplot(
    data=df_subject_est, x="condition", hue="criterion_estimated", y="subject_est_prevresp",
    alpha=.3, palette=colors, dodge=True
)

plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Condition', fontsize=15)
plt.ylabel(r"$\beta_{previous \,response}$", fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.legend(title='',loc='upper left')
plt.show()




