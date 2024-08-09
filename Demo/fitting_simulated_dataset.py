# -*- coding: utf-8 -*-
"""
@author: Robin Vloeberghs
"""


"""
Fitting hMFC to a simulated dataset
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
import numpy as np

from hmfc import *




""" Specify some variables and the true value for the hierarchical (global/group) parameters
   
    num_iters: number of iterations for the estimation procedure
    num_inputs: number of input variables
    num_subjects: number of subjects
    num_trials: number of trials per subject
   
    true_a0: true mean of truncated normal for AR coefficient
    true_nu_a: true standard deviation of truncated normal for AR coefficient
    true_w0: true means of normal distributions for input variables
    true_nu_w: true standard deviations of normal distributions for input variables
    true_alpha: true alpha of inverse gamma for sigma_sq
    true_beta: true beta of inverse gamma for sigma_sq
"""

num_subjects = 50
num_trials = 500
num_iters = 100
num_inputs = 4 
burn_in = 0


true_a0 = 0.99
true_nu_a0 = 0.025
true_w0 = jnp.array([0.0, 0.2, -0.3, 0.6])
true_nu_w0 = jnp.array([1.0, 1.0, 1.0, 1.0])
true_alpha = 5.0
true_beta = 5.0 * 0.3**2


# Create our true model
true_model = HierarchicalBernoulliLDS(true_a0, true_nu_a0, true_w0, true_nu_w0, true_alpha, true_beta)


# Make bridge such that hmfc.py can use these variables
model_parameters['num_iters'] = num_iters
model_parameters['num_inputs'] = num_inputs
model_parameters['num_subjects'] = num_subjects
model_parameters['num_trials'] = num_trials
model_parameters['true_a'] = true_a0
model_parameters['true_nu_a'] = true_nu_a0
model_parameters['true_w0'] = true_w0
model_parameters['true_nu_w'] = true_nu_w0
model_parameters['true_alpha'] = true_alpha
model_parameters['true_beta'] = true_beta
model_parameters['true_model'] = true_model



""" Simulate one dataset
    
    true_params: per-subject (local) true parameter values for a_i, sigma_sq_i, w_i, nu_w_i 
                 sampled from distributions with true hierarchical parameters specified earlier
    true_states: true criterion trajectories for all subjects
    emissions: simulated responses (0 or 1) based on true_params, true_states, and inputs
    inputs: represent the input variables, sampled from standard normal (see hmfc.py)
    masks: not needed here, only needed when subjects differ in number of trials (see more details: fitting_empirical_dataset)
    
    simulate_one_dataset: samples true_params from hierarchical (global/group) distributions for each subject
                          generates a true criterion trajectory according to AR(1) model with per-subject true a_i and sigmasq_i
                          samples inputs from standard normal distributions
                          weighted inputs variables + criterion determine trial-by-trial Bernoulli logodds which
                          allows to generate binary responses (emissions)
"""

# Sample dataset
key = jr.PRNGKey(0)
true_params, true_states, emissions, inputs, masks = simulate_one_dataset(key)


# Initialize model
init_a0 = 0.90
init_nu_a0 = 0.1
init_w0 = jnp.zeros(num_inputs)
init_nu_w0 = jnp.repeat(1.0, num_inputs)
init_alpha = 3.0
init_beta = 3.0 * 0.3**2

model = HierarchicalBernoulliLDS(init_a0, init_nu_a0, init_w0, init_nu_w0, init_alpha, init_beta)
params, states, _ = model.sample(key, inputs)



""" Fit model to dataset

    Iteratively runs the blocked Gibbs sampling algorithm
    On each iteration, it saves the posterior samples for the global and local parameters, and the states (criterion trajectory)
"""
# TODO: make this code more efficient

lps = [] # log probability

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

for itr in progress_bar(range(num_iters)):
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
    posterior_samples_states.append(states)


posterior_samples_a0 = jnp.stack(posterior_samples_a0)
posterior_samples_nu_a0 = jnp.stack(posterior_samples_nu_a0)
posterior_samples_w0 = jnp.stack(posterior_samples_w0)
posterior_samples_nu_w0 = jnp.stack(posterior_samples_nu_w0)
posterior_samples_alpha = jnp.stack(posterior_samples_alpha)
posterior_samples_beta = jnp.stack(posterior_samples_beta)    

posterior_samples_a = jnp.stack(posterior_samples_a)
posterior_samples_sigmasq = jnp.stack(posterior_samples_sigmasq)
posterior_samples_w = jnp.stack(posterior_samples_w)

posterior_samples_states = jnp.stack(posterior_samples_states)



""" Set seaborn style for plotting figures
"""

sns.set(style="ticks", context="paper",
        font="Arial",
        rc={"font.size": 25,
            "axes.titlesize": 25,
            "axes.labelsize": 25,
            "lines.linewidth": 1.5,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "axes.spines.right": False,
            "axes.spines.top": False
            })



""" Check log joint probability to assess convergence and determine number of burn-in iterations
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(jnp.stack(lps)/emissions.size) # normalized log joint prob
plt.xlabel("Iteration")
plt.ylabel("Log joint probability")
plt.ylim()
plt.show()



""" Estimated criterion fluctuations
"""

# calculate posterior mean and std over iterations for each trial without burn_in trials
posterior_samples_states_mean = jnp.mean(posterior_samples_states[:,:,burn_in:], axis=0) 
posterior_samples_states_std = jnp.std(posterior_samples_states[:,:,burn_in:], axis=0)

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_estimated_states.pdf') as pdf:
  for subject in range(num_subjects):
      
      plt.figure(figsize=(8, 6), dpi=600)
      plt.plot(true_states[subject], label="True states")
      plt.plot(posterior_samples_states_mean[subject], label="Inferred states")
      plt.fill_between(jnp.arange(num_trials), #95% CI
                  posterior_samples_states_mean[subject] - 2 * posterior_samples_states_std[subject],
                  posterior_samples_states_mean[subject] + 2 * posterior_samples_states_std[subject], color='r', alpha=0.25)
      plt.xlabel("Trial")
      plt.ylabel("Criterion $x_t$")
      plt.title(r"Subject {:d}".format(subject))
      plt.legend(loc='lower right')
      r, p = spearmanr(true_states[subject], posterior_samples_states_mean[subject])
      plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.90), xycoords='axes fraction')

      pdf.savefig()
      plt.close()



""" Recovery criterion fluctuations 
"""

# Correlation between inferred and fitted criterion trajectories
correlations_inferred_and_fitted_states = jnp.array([spearmanr(true_states[subject], posterior_samples_states_mean[subject])[0] for subject in range(num_subjects)])

mean_value = jnp.mean(correlations_inferred_and_fitted_states)
median_value = jnp.median(correlations_inferred_and_fitted_states)

plt.figure(figsize=(8, 6), dpi=600)
sns.histplot(correlations_inferred_and_fitted_states, bins=25)
plt.annotate(f'Mean: {mean_value:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=18)
plt.annotate(f'Median: {median_value:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=18)
plt.xlabel('Spearman correlation generative and fitted criterion', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xlim(0, 1)
plt.show()



""" Compare true and inferred estimates of a_i averaged over iterations (minus the burn-in)
"""

# Compute posterior mean of a for each subject over iterations
posterior_mean_a = jnp.mean(posterior_samples_a[burn_in:], axis=0)

a_min = min(true_params['a'].min(), posterior_mean_a.min())
a_max = max(true_params['a'].max(), posterior_mean_a.max())

plt.figure(figsize=(8, 6), dpi=600)
plt.scatter(true_params['a'], posterior_mean_a)
plt.plot([a_min, a_max], [a_min, a_max], '-k')
plt.xlabel("True $a$")
plt.ylabel("Inferred $a$")
plt.gca().set_aspect(1.0)
r, p = spearmanr(true_params['a'], posterior_mean_a)
plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction')
plt.show()



""" Compare true and inferred estimates of sigma_sq averaged over iterations (minus the burn-in)
"""

posterior_mean_sigmasq = jnp.mean(posterior_samples_sigmasq[burn_in:], axis=0)
  
sigmasq_min = min(true_params['sigmasq'].min(), posterior_mean_sigmasq.min())
sigmasq_max = max(true_params['sigmasq'].max(), posterior_mean_sigmasq.max())

plt.figure(figsize=(8, 6), dpi=600)
plt.scatter(true_params['sigmasq'], posterior_mean_sigmasq)
plt.plot([sigmasq_min, sigmasq_max], [sigmasq_min, sigmasq_max], '-k')
plt.xlabel("True $\sigma^2$")
plt.ylabel("Inferred $\sigma^2$")
plt.gca().set_aspect(1.0)
r, p = spearmanr(true_params['sigmasq'], posterior_mean_sigmasq)
plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction')
plt.show()



""" Compare true and inferred estimates of w averaged over iterations (minus the burn-in)
"""
    
w_min = min(true_params["w"].min(), jnp.mean(posterior_samples_w[burn_in:,:,:], axis=0).min())
w_max = max(true_params["w"].max(), jnp.mean(posterior_samples_w[burn_in:,:,:], axis=0).max())

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)

for d, ax in enumerate(axs):
    ax.scatter(true_params["w"][:, d], jnp.mean(posterior_samples_w[burn_in:,:,d], axis=0))
    ax.set_xlabel(r"True $w_{:d}$".format(d), size=20)
    ax.set_ylabel(r"Inferred $w_{:d}$".format(d), size=20)
    ax.plot([w_min, w_max], [w_min, w_max], '-k')
    ax.set_aspect(1.0)
    r, p = spearmanr(true_params["w"][:, d], jnp.mean(posterior_samples_w[burn_in:,:,d], axis=0))
    ax.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=18)



# ================================================================== #
# GLOBAL (HIERARCHICAL) PARAMETERS
# ================================================================== #

""" Posterior distributions of w0 (mean of normal distribution for w_i)
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

mean_true_per_subject_w = jnp.mean(true_params['w'], axis=0)
for d, ax in enumerate(axs):
    ax.hist(posterior_samples_w0[burn_in:, d], bins=30)
    ax.axvline(x=true_w0[d], color='black', linewidth=3)
    ax.axvline(x=mean_true_per_subject_w[d], color='red', linewidth=3)
    
    if d == 0:
        ax.set_ylabel("Count", fontsize=15)
    ax.set_xlabel(r"$w_{:d}$".format(d), fontsize=15)
    
fig.legend([r"True $w$", r'Mean true per-subject $w_i$'])
plt.show()
      
  

""" Posterior distributions of w0 (mean of normal distribution for w_i) shown over time
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

mean_true_per_subject_w = jnp.mean(true_params['w'], axis=0)
for d, ax in enumerate(axs):
    ax.plot(posterior_samples_w0[burn_in:, d])
    ax.axhline(y=true_w0[d], color='black')
    ax.axhline(y=mean_true_per_subject_w[d], color='red')
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel(r"$w_{:d}$".format(d), fontsize=15)
    
fig.legend([Line2D([0], [0], color='black'), Line2D([0], [0], color='red')],[r"True $w$", r'Mean true per-subject $w_i$']) # fix to have black line in legend (without it is blue)
plt.show()



""" Posterior distributions of nu_w0 (variance of normal distribution for w_i)
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

sd_true_per_subject_w = jnp.std(true_params['w'], axis=0)
for d, ax in enumerate(axs):
    ax.hist(posterior_samples_nu_w0[burn_in:, d], bins=30)
    ax.axvline(x=true_nu_w0[d], color='black')
    ax.axvline(x=sd_true_per_subject_w[d], color='red')
    if d == 0:
        ax.set_ylabel("Count", fontsize=15)
    ax.set_xlabel(r"$\nu_{{w_{:d}}}$".format(d), fontsize=15)

fig.legend([r"True $\nu_w$", r'Std true per-subject $w_i$'])
plt.show()



""" Posterior distributions of nu_w0 (variance of normal distribution for w_i) shown over time
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

sd_true_per_subject_w = jnp.std(true_params['w'], axis=0)
for d, ax in enumerate(axs):
    ax.plot(posterior_samples_nu_w0[burn_in:, d])
    ax.axhline(y=true_nu_w0[d], color='black')
    ax.axhline(y=sd_true_per_subject_w[d], color='red')
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel(r"$\nu_{{w_{:d}}}$".format(d), fontsize=15)

fig.legend([Line2D([0], [0], color='black'), Line2D([0], [0], color='red')],[r"True $\nu_w$", r'Std true per-subject $w_i$']) # fix to have black line in legend (without it is blue)
plt.show()



""" Posterior distributions of a0 (mean of truncated normal for a_i)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_a0[burn_in:], bins=30)
plt.axvline(true_a0, color='black', label="True $a_0$")
plt.axvline(jnp.mean(true_params['a']), color='red', label="Mean true per-subject $a_i$")
plt.xlabel("$a_0$")
plt.ylabel("Count")
plt.legend(loc='upper right')
plt.suptitle("Posterior distributions", fontsize=16)
plt.show()
    


""" Posterior distributions of a0 (mean of truncated normal for a_i) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_a0[burn_in:])
plt.axhline(true_a0, color='black', label="True $a_0$")
plt.axhline(jnp.mean(true_params['a']), color='red', label="Mean true per-subject $a_i$")
plt.xlabel("Iteration")
plt.ylabel("$a_0$")
plt.legend(loc='upper right')
plt.show()



""" Posterior distributions of nu_a0 (standard deviation of truncated normal for a_i)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_nu_a0[burn_in:], bins=30)
plt.axvline(true_nu_a0, color='black', label=r"True $\nu_a$")
plt.axvline(jnp.std(true_params['a']), color='red', label="Std true per-subject $a_i$")
plt.ylabel("Count")
plt.xlabel(r"$\nu_a$")
plt.legend(loc='upper right')
plt.show()


""" Posterior distributions of nu_a0 (standard deviation of truncated normal for a_i) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_nu_a0[burn_in:])
plt.axhline(true_nu_a0, color='black', label=r"True $\nu_a$")
plt.axhline(jnp.std(true_params['a']), color='red', label="Std true per-subject $a_i$")
plt.xlabel("Iteration")
plt.ylabel(r"$\nu_a$")
plt.legend(loc='upper right')
plt.show()



""" Posterior distributions of alpha (shape parameter of inverse gamma for sigmasq)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_alpha[burn_in:], bins=30)
plt.axvline(true_alpha, color='black', label=r"True $\alpha$")
plt.ylabel("Count")
plt.xlabel(r"$\alpha$")
plt.legend(loc='upper right')
plt.show()


""" Posterior distributions of alpha (shape parameter of inverse gamma for sigmasq) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_alpha[burn_in:])
plt.axhline(true_alpha, color='black', label=r"True $\alpha$")
plt.xlabel("Iteration")
plt.ylabel(r"$\alpha$")
plt.legend(loc='upper right')
plt.show()



""" Posterior distributions of beta (scale parameter of inverse gamma for sigmasq)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_beta[burn_in:], bins=30)
plt.axvline(true_beta, color='black', label=r"True $\beta$")
plt.ylabel("Count")
plt.xlabel(r"$\beta$")
plt.legend(loc='upper right')
plt.show()
    


""" Posterior distributions of beta (scale parameter of inverse gamma for sigmasq) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_beta[burn_in:])
plt.axhline(true_beta, color='black', label=r"True $\beta$")
plt.xlabel("Iteration")
plt.ylabel(r"$\beta$")
plt.legend(loc='upper right')
plt.show()
