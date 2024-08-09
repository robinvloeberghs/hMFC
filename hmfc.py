# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:10:19 2024

@author: u0141056
"""



import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt
import seaborn as sns
import dill

from scipy.stats import pearsonr, spearmanr
from fastprogress import progress_bar
from jax import grad, jit, lax, vmap
from jax.nn import sigmoid
from jaxtyping import Float, Array, Int
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# to make bridge with simulation script
model_parameters = {
    'burn_in' : None,
    'true_model' : None,
    'true_a' : None,
    'true_nu_a' : None,
    'true_alpha' : None,
    'true_beta' : None,
    'true_w0' : None,
    'true_nu_w' : None,
    'num_iters' : None,
    'num_inputs' : None,
    'num_subjects' : None,
    'num_trials' : None,
    'num_datasets' : None
    }


class HierarchicalBernoulliLDS(eqx.Module):
    r"""
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
               inputs : Float[Array, "num_subjects num_trials num_inputs"]
               ):
        r"""
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

            # Choose sigma_sq as such that the states have a stationary distribution with variance 2^2
            #range_state = 2.0**2
            #sigmasq_i = (range_state*(1 - a_i**2))

            sigmasq_i = tfd.InverseGamma(jnp.exp(self.log_alpha),
                                         jnp.exp(self.log_beta)).sample(seed=k3)

            # Sample latent states
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



# @title Helper functions for information form message passing
from jax.scipy.linalg import solve_triangular

def lds_info_filter(J_diag, J_lower_diag, h):
    """ Information form filtering for a linear Gaussian dynamical system.
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

            # Sample the gaussian
            x = _sample_info_gaussian(key, Jc, hc)
            return x, x

        # Initialize with sample of last timestep and sample in reverse
        keys = jr.split(key, model_parameters['num_trials'])
        x_T = _sample_info_gaussian(keys[-1], filtered_Js[-1], filtered_hs[-1])
        args = (keys[:-1], filtered_Js[:-1], filtered_hs[:-1], J_lower_diag)
        _, x = lax.scan(_step, x_T, args, reverse=True)

        # Append the last sample
        return jnp.vstack((x, x_T))

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

    return vmap(_sample_one)(jr.split(key, model_parameters['num_subjects']),
                             emissions,
                             masks,
                             inputs,
                             pg_samples,
                             params)

"""### Sampling the parameters
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
    r"""
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

"""### Sample the global parameters using Gibbs
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
    r"""
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
    r"""
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
    r"""
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
    r"""
    Update alpha of inverse gamma for sigmasq with RWMH
    """

    def _log_prob_alpha(log_alpha):
        alpha = jnp.exp(log_alpha)
        beta = jnp.exp(model.log_beta)
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, 10),  # Adjust the upper bound as needed
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
    r"""
    Update beta of inverse gamma for sigmasq with RWMH
    """

    def _log_prob_beta(log_beta):
        alpha = jnp.exp(model.log_alpha)
        beta = jnp.exp(log_beta)
        lp = tfd.TransformedDistribution(
            tfd.Uniform(0, 10),  # Adjust the upper bound as needed
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



def gibbs_step_global_params(key,
                             model : HierarchicalBernoulliLDS,
                             params : dict):
    k1, k2, k3, k4,k5 = jr.split(key, 5)
    model = _gibbs_step_global_weights(k1, model, params)
    model = _gibbs_step_global_ar(k2, model, params)
    model = _gibbs_step_global_ar_var(k3, model, params)
    model = _gibbs_step_global_sigmasq_alpha(k4, model, params)
    model = _gibbs_step_global_sigmasq_beta(k5, model, params)

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

"""### Put all the steps together"""

@jit
def gibbs_step(key,
               emissions : Float[Array, "num_subjects num_trials"],
               masks: Float[Array, "num_subjects num_trials"],
               states: Float[Array, "num_subjects num_trials"],
               inputs : Float[Array, "num_subjects num_trials num_inputs"],
               params: dict,
               model : HierarchicalBernoulliLDS
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
    model = gibbs_step_global_params(k4, model, params)

    return lp, states, params, model

    
    
def simulate_one_dataset(key):
    global model_parameters
    # Sample inputs
    inputs = tfd.Normal(0, 1).sample(sample_shape=(model_parameters['num_subjects'], model_parameters['num_trials'], model_parameters['num_inputs']), seed=key)
    inputs = inputs.at[:, :, 0].set(1.0)
    
    # Sample true params, states, and emissions
    true_params, true_states, emissions = model_parameters['true_model'].sample(key, inputs)
    masks = jnp.ones_like(emissions)
    
    """
    Problem: if absolute value of true_states for a subject exceeds crit_value, then sigmoid becomes saturated.
    This is very hard for the model to recover so we'll resample the true_states for these problematic subjects with
    their original true_params. If we can't find abs(true_states) > 10 within max_iter iterations, we resample a_i and sigmasq_i.
    """
    crit_value = 9
    max_iter = 50
    
    
    bad_subjects = jnp.where(jnp.max(abs(true_states),axis=1) > crit_value)[0] # check for each subject the absolute true_states exceed crit_value
    
    if bad_subjects.size > 0:
        print('abs(true states) > crit_value detected')
        for subj in bad_subjects:
            print(subj)
            # Call the true params and input from bad subject
            x_i = true_states[subj]
            u_i = inputs[subj]
            
            w_i = true_params['w'][subj]
            a_i = true_params['a'][subj]
            sigmasq_i = true_params['sigmasq'][subj]
            
            
            # Resample true_states and emissions from true params and inputs
            i = 0
            
            while jnp.any(jnp.abs(x_i) > crit_value):
                
                key = jr.PRNGKey(i)
                k1, k2, k3, k4, k5 = jr.split(key,5)
                
                
                # Sample latent states
                x_i0 = tfd.Normal(0, 1.0).sample(seed=k1)
                def _step(x_it, key):
                    x_itp1 = tfd.Normal(a_i * x_it, jnp.sqrt(sigmasq_i)).sample(seed=key)
                    return x_itp1, x_it
                _, x_i = lax.scan(_step, x_i0, jr.split(k2, model_parameters['num_trials']))
                
                # Ensure the latent states are centered
                x_i -= x_i.mean()
                
                # Sample emissions
                y_i = tfd.Bernoulli(x_i + u_i @ w_i).sample(seed=k3)
                
                i=i+1 # change seed for key
                
                if i == max_iter: # if stuck in a loop, resample parameters
                    # sample new params
                    print('new params sampled!')
                    a_i = tfd.TruncatedNormal(model_parameters['true_a'],
                                              model_parameters['true_nu_a'],
                                              0.0, 1.0).sample(seed=k4)
        
                    sigmasq_i = tfd.InverseGamma(model_parameters['true_alpha'],
                                                 model_parameters['true_beta']).sample(seed=k5)
                    
                    
                    # save new params
                    true_params['a'] = true_params['a'].at[subj].set(a_i)
                    true_params['sigmasq'] = true_params['sigmasq'].at[subj].set(sigmasq_i)
                
            # Save new true_states and emissions

            true_states = true_states.at[subj].set(x_i)
            emissions = emissions.at[subj].set(y_i)
        

    return true_params, true_states, emissions, inputs, masks

    
def simulate_and_fit_model(keys):
    key = jr.PRNGKey(keys)

    # Sample dataset
    true_params, true_states, emissions, inputs, masks = simulate_one_dataset(key)

    # Initialize model
    init_nu_w = jnp.repeat(1.25, model_parameters['num_inputs'])
    model = HierarchicalBernoulliLDS(0.90, 0.1, jnp.zeros(model_parameters['num_inputs']), init_nu_w, 3.0, 3.0 * 0.3**2)
    params, states, _ = model.sample(key, inputs)

    # Fit model
    lps = []
    
    posterior_samples_a0 = []
    posterior_samples_nu_a0 = []
    posterior_samples_w0 = []
    posterior_samples_nu_w0 = []
    posterior_samples_alpha = []
    posterior_samples_beta = []

    posterior_samples_a = []
    posterior_samples_sigmasq = []
    posterior_samples_w = []
    
    posterior_samples_states = []

    for itr in progress_bar(range(model_parameters['num_iters'])):
        this_key, key = jr.split(key)
        lp, states, params, model = gibbs_step(this_key, emissions, masks, states, inputs, params, model)
        lps.append(lp)
    
    
        posterior_samples_a0.append(sigmoid(model.logit_a_0))
        posterior_samples_nu_a0.append(jnp.exp(model.log_nu_a))
        posterior_samples_w0.append(model.w_0)
        posterior_samples_nu_w0.append(jnp.exp(model.log_nu_w))
        posterior_samples_alpha.append(jnp.exp(model.log_alpha))
        posterior_samples_beta.append(jnp.exp(model.log_beta))

        posterior_samples_a.append(params['a'])
        posterior_samples_sigmasq.append(params['sigmasq'])
        posterior_samples_w.append(params['w'])

        if itr == model_parameters['burn_in']:
            states_sum = states
            states_sum_squared = states
        elif itr > model_parameters['burn_in']:
            states_sum += states
            states_sum_squared += states**2 # for calculation standard deviation later
   

    posterior_samples_a0 = jnp.stack(posterior_samples_a0)
    posterior_samples_nu_a0 = jnp.stack(posterior_samples_nu_a0)
    posterior_samples_w_0 = jnp.stack(posterior_samples_w0)
    posterior_samples_nu_w0 = jnp.stack(posterior_samples_nu_w0)
    posterior_samples_alpha = jnp.stack(posterior_samples_alpha)
    posterior_samples_beta = jnp.stack(posterior_samples_beta)    

    posterior_samples_a = jnp.stack(posterior_samples_a)
    posterior_samples_sigmasq = jnp.stack(posterior_samples_sigmasq)
    posterior_samples_w = jnp.stack(posterior_samples_w)

    #posterior_samples_states = jnp.stack(posterior_samples_states) # could result in internal memory issues (jnp.concatenate) with large number of iterations
    posterior_samples_states = jnp.stack(states_sum) 
    posterior_samples_states_squared = jnp.stack(states_sum_squared)


    return true_params, true_states, params, states, emissions, inputs, lps, posterior_samples_a0, posterior_samples_nu_a0, posterior_samples_w_0, posterior_samples_nu_w0, posterior_samples_alpha, posterior_samples_beta, posterior_samples_a, posterior_samples_sigmasq, posterior_samples_w, posterior_samples_states, posterior_samples_states_squared  


