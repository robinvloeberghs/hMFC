# -*- coding: utf-8 -*-
"""
@author: Robin Vloeberghs
"""


"""
Fitting hMFC to an empirical dataset
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

from hmfc import * # make sure hmfc.py is located in current working directory


""" Load in dataset

    Should be in long format. Following columns have to be present:
        'subj': indicating subject number
        'resp': indicating the responses or emissions (should be 0 and 1)
        all the input variables you want to include as predictor in the model (e.g. stimulus, previous response,...)
    
"""

# TODO: provide example df?
data = pd.read_csv('YOUR_DF_HERE.csv')

# Indicate number of input variables that will be used in the model (intercept should be added as well)
# For example: intercept, stimulus, previous resp, previous stimulus -> num_input = 4
num_inputs = 4 



""" Put dataset in correct data structure
    
    We also have to create a 'mask' variable. This variable helps the model to deal with varying numbers of trials per subject.
    The masks variable indicates which data points in the emissions (responses) array are valid and which are padded 
    (i.e., added to make all subjects have the same number of trials). 
    
"""

num_trials_per_subject = jnp.array(data.groupby('subj').size())
max_num_trials = data.groupby('subj').size().max()  # Find the maximum number of observations
num_trials = max_num_trials

inputs, emissions, masks = [], [], []

for i in np.unique(data.subj):
    df = data[data.subj == i]
    
    """ Adjust the names of the variables below according to your dataset
    """
    
    intercept = jnp.repeat(1, len(df)) # 1 everywhere for model to know this is an intercept
    evidence = jnp.array(df.evidence)  # scaled between -1 and 1
    prevevidence = jnp.array(df.prev_evidence)  # -1 left, 1 right
    prevresp = jnp.array(df.prev_resp)  # -1 left, 1 right

    resp = jnp.array(df.resp)

    inputs_subj = jnp.vstack([intercept, evidence, prevevidence, prevresp]).T
    emissions_subj = resp
    
    # Create masking variable
    masks_subj = jnp.ones_like(emissions_subj)

    # Check if the subject has fewer observations than the maximum
    if df.shape[0] < max_num_trials:
        # Calculate the number of observations to fill
        num_to_fill = max_num_trials - df.shape[0]
        # Create a matrix of zeros to fill in the missing observations
        zero_input = jnp.zeros((num_to_fill, num_inputs))
        zero_emissions = jnp.zeros(num_to_fill)

        # Append the zero-filled observations
        inputs_subj = jnp.vstack([inputs_subj, zero_input])
        emissions_subj = jnp.concatenate((emissions_subj, zero_emissions))
        masks_subj = jnp.concatenate((masks_subj, zero_emissions))

    # Append the results to the output lists
    inputs.append(inputs_subj)
    emissions.append(emissions_subj)
    masks.append(masks_subj)

# Convert the lists to arrays
inputs = jnp.array(inputs)
emissions = jnp.array(emissions)
masks = jnp.array(masks)



""" Initialize some variables

    num_iters: number of iterations for the estimation procedure
    num_trials: number of trials per subject
    num_subjects: number of subjects
    num_inputs: number of input variables (specified earlier)

"""

num_iters = 10
num_trials = max_num_trials # set to number trials of subject with most trials, masking takes care of the other subjects
num_subjects = len(inputs)



# Make bridge such that hmfc.py can use these variables
model_parameters['num_iters'] = num_iters
model_parameters['num_inputs'] = num_inputs
model_parameters['num_subjects'] = num_subjects
model_parameters['num_trials'] = num_trials



""" Initialize the model

    init_a0: initial mean of truncated normal for AR coefficient
    init_nu_a: initial standard deviation of truncated normal for AR coefficient
    init_w0: initial means of normal distributions for input variables
    init_nu_w: initial standard deviations of normal distributions for input variables
    init_alpha: initial alpha of inverse gamma for sigma_sq
    init_beta: initial beta of inverse gamma for sigma_sq
"""

init_a0 = 0.90
init_nu_a0 = 0.1
init_w0 = jnp.zeros(num_inputs)
init_nu_w0 = jnp.repeat(1.0, num_inputs)
init_alpha = 3.0
init_beta = 3.0 * 0.3**2

key = jr.PRNGKey(0)

model = HierarchicalBernoulliLDS(init_a0, init_nu_a0, init_w0, init_nu_w0, init_alpha, init_beta)
params, states, _ = model.sample(key, inputs) # sample initial per-subject parameters and states (criterion trajectory)



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
        rc={"axes.titlesize": 20,
            "axes.labelsize": 20,
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



burn_in = 0 # number of iterations to discard



# ================================================================== #
# GLOBAL (HIERARCHICAL) PARAMETERS
# ================================================================== #


""" Posterior distributions of w0 (mean of normal distribution for w_i)
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

for d, ax in enumerate(axs):
    ax.hist(posterior_samples_w0[burn_in:, d], bins=30)
    ax.axvline(x=0, color='red', linewidth=3)
    if d == 0:
        ax.set_ylabel("Count", fontsize=15)
    ax.set_xlabel(r"$w_{:d}$".format(d), fontsize=15)
    
plt.show()
      

""" Posterior distributions of w0 (mean of normal distribution for w_i) shown over time
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

for d, ax in enumerate(axs):
    ax.plot(posterior_samples_w0[burn_in:, d])
    ax.axhline(y=0, color='red')
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel(r"$w_{:d}$".format(d), fontsize=15)
    
plt.show()




""" Posterior distributions of nu_w0 (variance of normal distribution for w_i)
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

for d, ax in enumerate(axs):
    ax.hist(posterior_samples_nu_w0[burn_in:, d], bins=30)
    if d == 0:
        ax.set_ylabel("Count", fontsize=15)
    ax.set_xlabel(r"$\nu_{{w_{:d}}}$".format(d), fontsize=15)

plt.show()


""" Posterior distributions of nu_w0 (variance of normal distribution for w_i) shown over time
"""

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
fig.suptitle("Posterior distributions", fontsize=16)

for d, ax in enumerate(axs):
    ax.plot(posterior_samples_nu_w0[burn_in:, d])
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel(r"$\nu_{{w_{:d}}}$".format(d), fontsize=15)

plt.show()




""" Posterior distributions of a0 (mean of truncated normal for a_i)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_a0[burn_in:], bins=30)
plt.xlabel("$a_0$")
plt.ylabel("Count")
plt.title("Posterior distributions", fontsize=16)
plt.show()
    

""" Posterior distributions of a0 (mean of truncated normal for a_i) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_a0[burn_in:])
plt.xlabel("Iteration")
plt.ylabel("$a_0$")
plt.show()




""" Posterior distributions of nu_a0 (standard deviation of truncated normal for a_i)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_nu_a0[burn_in:], bins=30)
plt.ylabel("Count")
plt.xlabel(r"$\nu_a$")
plt.show()


""" Posterior distributions of nu_a0 (standard deviation of truncated normal for a_i) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_nu_a0[burn_in:])
plt.xlabel("Iteration")
plt.ylabel(r"$\nu_a$")
plt.show()




""" Posterior distributions of alpha (shape parameter of inverse gamma for sigmasq)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_alpha[burn_in:], bins=30)
plt.ylabel("Count")
plt.xlabel(r"$\alpha$")
plt.show()


""" Posterior distributions of alpha (shape parameter of inverse gamma for sigmasq) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_alpha[burn_in:])
plt.xlabel("Iteration")
plt.ylabel(r"$\alpha$")
plt.show()




""" Posterior distributions of beta (scale parameter of inverse gamma for sigmasq)
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.hist(posterior_samples_beta[burn_in:], bins=30)
plt.ylabel("Count")
plt.xlabel(r"$\beta$")
plt.show()
    

""" Posterior distributions of beta (scale parameter of inverse gamma for sigmasq) shown over time
"""

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_beta[burn_in:])
plt.xlabel("Iteration")
plt.ylabel(r"$\beta$")
plt.show()


# ================================================================== #
# LOCAL (PER-SUBJECT) PARAMETERS
# ================================================================== #


""" Local parameters (averaged over iterations; posterior means per subject)
"""

# Posterior mean per subject
fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
for d, ax in enumerate(axs):
    ax.hist(jnp.mean(posterior_samples_w[burn_in:,:,d], axis=0), bins=50)
    ax.set_xlabel("Value")
    if d == 0: ax.set_ylabel("Count")
    ax.set_title(r"$w_{:d}$".format(d))
    if d == 0: ax.set_title("$w_0$ (intercept)")
plt.show()


plt.figure(figsize=(8, 6), dpi=600)
plt.hist(jnp.mean(posterior_samples_a[burn_in:], axis=0), bins=50)
plt.xlabel(r"$a$")
plt.ylabel("Count")
plt.title(r"$a$ for each subject (averaged over iterations)")
plt.show()


plt.figure(figsize=(8, 6), dpi=600)
plt.hist(jnp.mean(posterior_samples_sigmasq[burn_in:], axis=0), bins=50)
plt.xlabel(r"$\sigma^2$")
plt.ylabel("Count")
plt.title(r"$\sigma^2$ for each subject (averaged over iterations)")
plt.show()




""" Local parameters (over iterations)

    Can be used to assess quirky behavior (e.g., parameter estimates keep going down over iterations)
"""

# Each line represents a subject
fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
for d, ax in enumerate(axs):
    ax.plot(posterior_samples_w[burn_in:,:,d])
    ax.set_xlabel("Iteration")
    if d == 0: ax.set_ylabel("Value (all subjects)")
    ax.set_title(r"$w_{:d}$".format(d))
    if d == 0: ax.set_title("$w_0$ (intercept)")
plt.show()


plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_a[burn_in:])
plt.xlabel("Iteration")
plt.ylabel(r"$a$ (all subjects)")
plt.show()


plt.figure(figsize=(8, 6), dpi=600)
plt.plot(posterior_samples_sigmasq[burn_in:])
plt.xlabel("Iteration")
plt.ylabel(r"$\sigma^2$ (all subjects)")
plt.show()



""" Estimated criterion fluctuations
    
    On each trial the posterior mean with 95% credible interval is shown
    The red vertical line indicates where the trials of the subject stop
"""

# calculate posterior mean and std over iterations for each trial without burn_in trials
posterior_samples_states_mean = jnp.mean(posterior_samples_states[:,:,burn_in:], axis=0) 
posterior_samples_states_std = jnp.std(posterior_samples_states[:,:,burn_in:], axis=0)

with PdfPages("estimated_criterion_fluctuations.pdf") as pdf:
  for subject in range(num_subjects):
      
      trials_subject = num_trials_per_subject[subject]
      plt.figure(figsize=(8, 6), dpi=600)
      plt.plot(posterior_samples_states_mean[subject,:trials_subject], label="Inferred states")
      plt.fill_between(jnp.arange(trials_subject), #95% CI
                  posterior_samples_states_mean[subject,:trials_subject] - 2 * posterior_samples_states_std[subject,:trials_subject],
                  posterior_samples_states_mean[subject,:trials_subject] + 2 * posterior_samples_states_std[subject,:trials_subject], color='r', alpha=0.25)
      plt.xlabel("Trial")
      plt.ylabel("Criterion $x_t$")
      plt.title(r"Subject {:d}".format(subject))
      plt.annotate(r'$a$ = {:.3f}'.format(jnp.mean(posterior_samples_a[burn_in:, subject])), xy=(0.05, 0.9), xycoords='axes fraction')
      plt.annotate(r'$\sigma^2$ = {:.3f}'.format(jnp.mean(posterior_samples_sigmasq[burn_in:, subject])), xy=(0.05, 0.85), xycoords='axes fraction')

      pdf.savefig()
      plt.close()





""" Save the estimated criterion fluctuations
"""
estimated_criterion_fluctuations = []

for subject in range(num_subjects):
  estimated_criterion_fluctuations.append(jnp.mean(posterior_samples_states[burn_in:,subject,:num_trials_per_subject[subject]], axis=0))

estimated_criterion_fluctuations = jnp.concatenate(estimated_criterion_fluctuations)

df_estimated_criterion_fluctuations = pd.DataFrame({'subj':data.subj,
                                                   'estimated_criterion_fluctuations':estimated_criterion_fluctuations})

df_estimated_criterion_fluctuations.to_csv('estimated_criterion_fluctuations.csv', index=False)