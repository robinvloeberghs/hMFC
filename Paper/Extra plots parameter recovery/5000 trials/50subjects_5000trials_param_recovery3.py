# -*- coding: utf-8 -*-
"""
@author: Robin Vloeberghs
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
import numpy as np

from hmfc import *



"""
Parameter recovery with multiple datasets
"""


burn_in = 250
num_iters = 1250
num_inputs = 4
num_subjects = 50
num_trials = 5000
num_datasets = 10

true_a = 0.99
true_nu_a = 0.025
true_alpha = 5.0
true_beta = 5.0 * 0.3**2
true_w0 = jnp.array([0.0, 0.2, -0.3, 0.6])
true_nu_w = jnp.array([1.0, 1.0, 1.0, 1.0])

true_model = HierarchicalBernoulliLDS(true_a, true_nu_a, true_w0, true_nu_w, true_alpha, true_beta)


# so if we import definitions from hmfc.py then the global variables specified above aren't read it
# this is weird because usually you would haul in your definitions which then can use the
# variables specified in your environment. Turns out that it doesn't work that way in Python.
# We have to bridge the variables to hmfc.py like below 
model_parameters['burn_in'] = burn_in
model_parameters['num_iters'] = num_iters
model_parameters['num_inputs'] = num_inputs
model_parameters['num_subjects'] = num_subjects
model_parameters['num_trials'] = num_trials
model_parameters['num_datasets'] = num_datasets

model_parameters['true_a'] = true_a
model_parameters['true_nu_a'] = true_nu_a
model_parameters['true_alpha'] = true_alpha
model_parameters['true_beta'] = true_beta
model_parameters['true_w0'] = true_w0
model_parameters['true_nu_w'] = true_nu_w

model_parameters['true_model'] = true_model





all_true_params = []
all_true_states = []
all_inf_params = []
all_inf_states = []
all_emissions = []
all_inputs = []
all_lps = []
all_posterior_samples_a0 = []
all_posterior_samples_nu_a0 = []
all_posterior_samples_w_0 = []
all_posterior_samples_nu_w0 = []
all_posterior_samples_alpha = []
all_posterior_samples_beta = []

all_posterior_samples_a = []
all_posterior_samples_sigmasq = []
all_posterior_samples_w = []

all_posterior_samples_states = []
all_posterior_samples_states_sd = [] # standard deviation


seed = jnp.arange(num_datasets) + 20 # to make sure we don't have the same datasets as other file

# simulate and fit datasets
# note: not possible to do this with vmap (makes it run in parallel) due pre-allocation of memory by vmap function leading to memory issues
for i in seed:
  true_params, true_states, params, states, emissions, inputs, lps, posterior_samples_a0, posterior_samples_nu_a0, posterior_samples_w_0, posterior_samples_nu_w0, posterior_samples_alpha, posterior_samples_beta, posterior_samples_a, posterior_samples_sigmasq, posterior_samples_w, posterior_samples_states, posterior_samples_states_squared = simulate_and_fit_model(i)
  
  all_true_params.append(true_params)
  all_true_states.append(true_states)
  all_inf_params.append(params)
  all_inf_states.append(states)
  all_emissions.append(emissions)
  all_inputs.append(inputs)
  all_lps.append(lps)

  all_posterior_samples_a0.append(posterior_samples_a0)
  all_posterior_samples_nu_a0.append(posterior_samples_nu_a0)
  all_posterior_samples_w_0.append(posterior_samples_w_0)
  all_posterior_samples_nu_w0.append(posterior_samples_nu_w0)
  all_posterior_samples_alpha.append(posterior_samples_alpha)
  all_posterior_samples_beta.append(posterior_samples_beta)
  
  all_posterior_samples_a.append(posterior_samples_a)
  all_posterior_samples_sigmasq.append(posterior_samples_sigmasq)
  all_posterior_samples_w.append(posterior_samples_w)
   

  # due to memory issues we cannot save the states for all iterations (below)
  # solution: we take the average states over iterations (without burn-in) and save this
  # posterior_samples_states is summed up states after burn_in (see hmfc.py), below we divide by num_iters to obtain average
  # disadvantage: we already have to specify burn_in here without looking at the joint log prob trajectory
  mean_states = posterior_samples_states / (num_iters - burn_in)  # take average states over all iterations without burn_in
  sd_states = jnp.sqrt(posterior_samples_states_squared/(num_iters - burn_in)-mean_states**2) # chiastic formula for standard deviation


  all_posterior_samples_states.append(mean_states)
  all_posterior_samples_states_sd.append(sd_states)




"""
Write some stuff away
"""

# FOR RECOVERY LOCAL PARAMETERS PLOT
average_recovery_states_all_datasets = []
median_recovery_states_all_datasets = []

for dataset in range(num_datasets):

  correlations_inferred_and_fitted_states = []
  
  for trial in range(num_subjects):
      r, _= spearmanr(all_true_states[dataset][trial], all_posterior_samples_states[dataset][trial])
      correlations_inferred_and_fitted_states.append(r)
  
  correlations_inferred_and_fitted_states = jnp.array(correlations_inferred_and_fitted_states)
  
  mean_value = jnp.mean(correlations_inferred_and_fitted_states)
  median_value = jnp.median(correlations_inferred_and_fitted_states)
  
  average_recovery_states_all_datasets.append(mean_value)
  median_recovery_states_all_datasets.append(median_value)

average_recovery_states_all_datasets = jnp.stack(average_recovery_states_all_datasets)
median_recovery_states_all_datasets = jnp.stack(median_recovery_states_all_datasets)


all_r_w1 = []
all_r_w2 = []
all_r_w3 = []
all_r_w4 = []
all_r_a = []
all_r_sigmasq = []


for dataset in range(num_datasets):

    generative_w1 = all_true_params[dataset]['w'][:,0]
    generative_w2 = all_true_params[dataset]['w'][:,1]
    generative_w3 = all_true_params[dataset]['w'][:,2]
    generative_w4 = all_true_params[dataset]['w'][:,3]
    generative_a = all_true_params[dataset]['a']
    generative_sigmasq = all_true_params[dataset]['sigmasq']

    
    fitted_w1 = jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,0], axis=0)
    fitted_w2 = jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,1], axis=0)
    fitted_w3 = jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,2], axis=0)
    fitted_w4 = jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,3], axis=0)
    fitted_a = jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0)
    fitted_sigmasq = jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0)


    r_w1, p_w1 = spearmanr(generative_w1, fitted_w1)
    r_w2, p_w2 = spearmanr(generative_w2, fitted_w2)
    r_w3, p_w3 = spearmanr(generative_w3, fitted_w3)
    r_w4, p_w4 = spearmanr(generative_w4, fitted_w4)
    r_a, p_a = spearmanr(generative_a, fitted_a)
    r_sigmasq, p_sigmasq = spearmanr(generative_sigmasq, fitted_sigmasq)


    all_r_w1.append(r_w1)
    all_r_w2.append(r_w2)
    all_r_w3.append(r_w3)
    all_r_w4.append(r_w4)
    all_r_a.append(r_a)
    all_r_sigmasq.append(r_sigmasq)


df_recovery_local = pd.DataFrame({
    'correlation_w0': (all_r_w1),
    'correlation_w1': (all_r_w2),
    'correlation_w2': (all_r_w3),
    'correlation_w3': (all_r_w4),
    'correlation_a': (all_r_a),
    'correlation_sigmasq': (all_r_sigmasq),
    'mean_correlation_states': (average_recovery_states_all_datasets),
    'median_correlation_states': (median_recovery_states_all_datasets)
})


df_recovery_local.to_csv(f'{num_trials}trials_{num_subjects}subjects_recovery_localparams3.csv', index=False)





# FOR HEATMAP PLOT
all_correlations_inferred_and_fitted_states = []
generative_a = []
generative_sigmasq = []
estimated_a = []
estimated_sigmasq = []

for dataset in range(num_datasets):

  generative_a.append(all_true_params[dataset]['a'])
  generative_sigmasq.append(all_true_params[dataset]['sigmasq'])
  
  estimated_a.append(all_inf_params[dataset]['a'])
  estimated_sigmasq.append(all_inf_params[dataset]['sigmasq'])
  
  correlations_inferred_and_fitted_states = []
  
  for trial in range(num_subjects):
      r, _= spearmanr(all_true_states[dataset][trial], all_posterior_samples_states[dataset][trial])
      correlations_inferred_and_fitted_states.append(r)
  
  all_correlations_inferred_and_fitted_states.extend(correlations_inferred_and_fitted_states)
  

generative_a = jnp.hstack(generative_a)
generative_sigmasq = jnp.hstack(generative_sigmasq)
estimated_a = jnp.hstack(estimated_a)
estimated_sigmasq = jnp.hstack(estimated_sigmasq)



df_recovery_criterion_heatmap = pd.DataFrame({
    'correlation': (all_correlations_inferred_and_fitted_states),
    'generative_a': (generative_a),
    'estimated_a': (estimated_a),
    'generative_sigmasq': (generative_sigmasq),
    'estimated_sigmasq': (estimated_sigmasq)
})


df_recovery_criterion_heatmap.to_csv(f'{num_trials}trials_{num_subjects}subjects_recovery_criterion_heatmap3.csv', index=False)




# FOR POSTERIOR MEANS PLOT
# Calculate posterior mean for each dataset

posterior_mean_alpha = []
posterior_mean_beta = []

posterior_mean_a0 = []
posterior_mean_nu_a0 = []

posterior_mean_w0 = []
posterior_mean_w1 = []
posterior_mean_w2 = []
posterior_mean_w3 = []

posterior_mean_nu_w0 = []
posterior_mean_nu_w1 = []
posterior_mean_nu_w2 = []
posterior_mean_nu_w3 = []


posterior_var_alpha = []
posterior_var_beta = []

posterior_var_a0 = []
posterior_var_nu_a0 = []

posterior_var_w0 = []
posterior_var_w1 = []
posterior_var_w2 = []
posterior_var_w3 = []

posterior_var_nu_w0 = []
posterior_var_nu_w1 = []
posterior_var_nu_w2 = []
posterior_var_nu_w3 = []

posterior_q25_alpha = []
posterior_q975_alpha = []

posterior_q25_beta = []
posterior_q975_beta = []

posterior_q25_a0 = []
posterior_q975_a0 = []

posterior_q25_nu_a0 = []
posterior_q975_nu_a0 = []

posterior_q25_w0 = []
posterior_q975_w0 = []
posterior_q25_w1 = []
posterior_q975_w1 = []
posterior_q25_w2 = []
posterior_q975_w2 = []
posterior_q25_w3 = []
posterior_q975_w3 = []

posterior_q25_nu_w0 = []
posterior_q975_nu_w0 = []
posterior_q25_nu_w1 = []
posterior_q975_nu_w1 = []
posterior_q25_nu_w2 = []
posterior_q975_nu_w2 = []
posterior_q25_nu_w3 = []
posterior_q975_nu_w3 = []

# per dataset calculate the mean over each subject's true parameter
mean_subject_true_a = []
sd_subject_true_a = []
mean_subject_true_w0 = []
mean_subject_true_w1 = []
mean_subject_true_w2 = []
mean_subject_true_w3 = []
sd_subject_true_w0 = []
sd_subject_true_w1 = []
sd_subject_true_w2 = []
sd_subject_true_w3 = []


for dataset in range(num_datasets):
    
    posterior_mean_alpha.append(jnp.mean(all_posterior_samples_alpha[dataset][burn_in:]))
    posterior_mean_beta.append(jnp.mean(all_posterior_samples_beta[dataset][burn_in:]))

    posterior_mean_a0.append(jnp.mean(all_posterior_samples_a0[dataset][burn_in:]))
    posterior_mean_nu_a0.append(jnp.mean(all_posterior_samples_nu_a0[dataset][burn_in:]))
    
    posterior_mean_w0.append(jnp.mean(all_posterior_samples_w_0[dataset][burn_in:,0]))
    posterior_mean_w1.append(jnp.mean(all_posterior_samples_w_0[dataset][burn_in:,1]))
    posterior_mean_w2.append(jnp.mean(all_posterior_samples_w_0[dataset][burn_in:,2]))
    posterior_mean_w3.append(jnp.mean(all_posterior_samples_w_0[dataset][burn_in:,3]))

    posterior_mean_nu_w0.append(jnp.mean(all_posterior_samples_nu_w0[dataset][burn_in:,0]))
    posterior_mean_nu_w1.append(jnp.mean(all_posterior_samples_nu_w0[dataset][burn_in:,1]))
    posterior_mean_nu_w2.append(jnp.mean(all_posterior_samples_nu_w0[dataset][burn_in:,2]))
    posterior_mean_nu_w3.append(jnp.mean(all_posterior_samples_nu_w0[dataset][burn_in:,3]))
    
    
    posterior_var_alpha.append(jnp.var(all_posterior_samples_alpha[dataset][burn_in:]))
    posterior_var_beta.append(jnp.var(all_posterior_samples_beta[dataset][burn_in:]))

    posterior_var_a0.append(jnp.var(all_posterior_samples_a0[dataset][burn_in:]))
    posterior_var_nu_a0.append(jnp.var(all_posterior_samples_nu_a0[dataset][burn_in:]))
    
    posterior_var_w0.append(jnp.var(all_posterior_samples_w_0[dataset][burn_in:,0]))
    posterior_var_w1.append(jnp.var(all_posterior_samples_w_0[dataset][burn_in:,1]))
    posterior_var_w2.append(jnp.var(all_posterior_samples_w_0[dataset][burn_in:,2]))
    posterior_var_w3.append(jnp.var(all_posterior_samples_w_0[dataset][burn_in:,3]))

    posterior_var_nu_w0.append(jnp.var(all_posterior_samples_nu_w0[dataset][burn_in:,0]))
    posterior_var_nu_w1.append(jnp.var(all_posterior_samples_nu_w0[dataset][burn_in:,1]))
    posterior_var_nu_w2.append(jnp.var(all_posterior_samples_nu_w0[dataset][burn_in:,2]))
    posterior_var_nu_w3.append(jnp.var(all_posterior_samples_nu_w0[dataset][burn_in:,3]))
    
    posterior_q25_alpha.append(np.percentile(all_posterior_samples_alpha[dataset][burn_in:], 2.5))
    posterior_q975_alpha.append(np.percentile(all_posterior_samples_alpha[dataset][burn_in:], 97.5))
    
    posterior_q25_beta.append(np.percentile(all_posterior_samples_beta[dataset][burn_in:], 2.5))
    posterior_q975_beta.append(np.percentile(all_posterior_samples_beta[dataset][burn_in:], 97.5))
    
    posterior_q25_a0.append(np.percentile(all_posterior_samples_a0[dataset][burn_in:], 2.5))
    posterior_q975_a0.append(np.percentile(all_posterior_samples_a0[dataset][burn_in:], 97.5))
    
    posterior_q25_nu_a0.append(np.percentile(all_posterior_samples_nu_a0[dataset][burn_in:], 2.5))
    posterior_q975_nu_a0.append(np.percentile(all_posterior_samples_nu_a0[dataset][burn_in:], 97.5))
    
    posterior_q25_w0.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,0], 2.5))
    posterior_q975_w0.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,0], 97.5))
    posterior_q25_w1.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,1], 2.5))
    posterior_q975_w1.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,1], 97.5))
    posterior_q25_w2.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,2], 2.5))
    posterior_q975_w2.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,2], 97.5))
    posterior_q25_w3.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,3], 2.5))
    posterior_q975_w3.append(np.percentile(all_posterior_samples_w_0[dataset][burn_in:,3], 97.5))
    
    posterior_q25_nu_w0.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,0], 2.5))
    posterior_q975_nu_w0.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,0], 97.5))
    posterior_q25_nu_w1.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,1], 2.5))
    posterior_q975_nu_w1.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,1], 97.5))
    posterior_q25_nu_w2.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,2], 2.5))
    posterior_q975_nu_w2.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,2], 97.5))
    posterior_q25_nu_w3.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,3], 2.5))
    posterior_q975_nu_w3.append(np.percentile(all_posterior_samples_nu_w0[dataset][burn_in:,3], 97.5))
    
    mean_subject_true_a.append(jnp.mean(all_true_params[dataset]['a']))
    sd_subject_true_a.append(jnp.std(all_true_params[dataset]['a']))
    
    mean_subject_true_w0.append(jnp.mean(all_true_params[dataset]['w'][:,0]))
    mean_subject_true_w1.append(jnp.mean(all_true_params[dataset]['w'][:,1]))
    mean_subject_true_w2.append(jnp.mean(all_true_params[dataset]['w'][:,2]))
    mean_subject_true_w3.append(jnp.mean(all_true_params[dataset]['w'][:,3]))
    
    sd_subject_true_w0.append(jnp.std(all_true_params[dataset]['w'][:,0]))
    sd_subject_true_w1.append(jnp.std(all_true_params[dataset]['w'][:,1]))
    sd_subject_true_w2.append(jnp.std(all_true_params[dataset]['w'][:,2]))
    sd_subject_true_w3.append(jnp.std(all_true_params[dataset]['w'][:,3]))
    
    
    
df_posterior_means = pd.DataFrame({
    'alpha': (posterior_mean_alpha),
    'beta': (posterior_mean_beta),
    'a0': (posterior_mean_a0),
    'nu_a0': (posterior_mean_nu_a0),
    'w0': (posterior_mean_w0),
    'w1': (posterior_mean_w1),
    'w2': (posterior_mean_w2),
    'w3': (posterior_mean_w3),
    'nu_w0': (posterior_mean_nu_w0),
    'nu_w1': (posterior_mean_nu_w1),
    'nu_w2': (posterior_mean_nu_w2),
    'nu_w3': (posterior_mean_nu_w3),
    'alpha_var': (posterior_var_alpha),
    'beta_var': (posterior_var_beta),
    'a0_var': (posterior_var_a0),
    'nu_a0_var': (posterior_var_nu_a0),
    'w0_var': (posterior_var_w0),
    'w1_var': (posterior_var_w1),
    'w2_var': (posterior_var_w2),
    'w3_var': (posterior_var_w3),
    'nu_w0_var': (posterior_var_nu_w0),
    'nu_w1_var': (posterior_var_nu_w1),
    'nu_w2_var': (posterior_var_nu_w2),
    'nu_w3_var': (posterior_var_nu_w3),
    'posterior_q25_alpha' : posterior_q25_alpha,
    'posterior_q975_alpha' : posterior_q975_alpha,
    'posterior_q25_beta':posterior_q25_beta,
    'posterior_q975_beta':posterior_q975_beta,
    'posterior_q25_a0':posterior_q25_a0,
    'posterior_q975_a0':posterior_q975_a0,
    'posterior_q25_nu_a0':posterior_q25_nu_a0,
    'posterior_q975_nu_a0':posterior_q975_nu_a0,
    'posterior_q25_w0': posterior_q25_w0,
    'posterior_q975_w0':posterior_q975_w0,
    'posterior_q25_w1': posterior_q25_w1,
    'posterior_q975_w1':posterior_q975_w1,
    'posterior_q25_w2':posterior_q25_w2,
    'posterior_q975_w2':posterior_q975_w2,
    'posterior_q25_w3':posterior_q25_w3,
    'posterior_q975_w3':posterior_q975_w3,
    'posterior_q25_nu_w0':posterior_q25_nu_w0,
    'posterior_q975_nu_w0':posterior_q975_nu_w0,
    'posterior_q25_nu_w1':posterior_q25_nu_w1,
    'posterior_q975_nu_w1':posterior_q975_nu_w1,
    'posterior_q25_nu_w2':posterior_q25_nu_w2,
    'posterior_q975_nu_w2':posterior_q975_nu_w2,
    'posterior_q25_nu_w3':posterior_q25_nu_w3,
    'posterior_q975_nu_w3':posterior_q975_nu_w3,
    'mean_subject_true_a': (mean_subject_true_a),
    'sd_subject_true_a': (sd_subject_true_a),
    'mean_subject_true_w0': (mean_subject_true_w0),
    'mean_subject_true_w1': (mean_subject_true_w1),
    'mean_subject_true_w2': (mean_subject_true_w2),
    'mean_subject_true_w3': (mean_subject_true_w3),
    'sd_subject_true_w0': (sd_subject_true_w0),
    'sd_subject_true_w1': (sd_subject_true_w1),
    'sd_subject_true_w2': (sd_subject_true_w2),
    'sd_subject_true_w3': (sd_subject_true_w3),
    'num_trials': (num_trials),
    'num_subjects': (num_subjects)
})


df_posterior_means.to_csv(f'{num_trials}trials_{num_subjects}subjects_posterior_means3.csv', index=False)



""" 
Save environment
"""

# Give other name such that dill file from other 5000trials recovery can be loaded in in the same environment
# without overwritting each other
all_true_params3 = all_true_params 
all_true_states3 = all_true_states
all_inf_params3 = all_inf_params
all_inf_states3 = all_inf_states
all_emissions3 = all_emissions
all_inputs3 = all_inputs
all_lps3 = all_lps
all_posterior_samples_a03 = all_posterior_samples_a0
all_posterior_samples_nu_a03 = all_posterior_samples_nu_a0
all_posterior_samples_w_03 = all_posterior_samples_w_0
all_posterior_samples_nu_w03 = all_posterior_samples_nu_w0
all_posterior_samples_alpha3 = all_posterior_samples_alpha
all_posterior_samples_beta3 = all_posterior_samples_beta

all_posterior_samples_a3 = all_posterior_samples_a
all_posterior_samples_sigmasq3 = all_posterior_samples_sigmasq
all_posterior_samples_w3 = all_posterior_samples_w

all_posterior_samples_states3 = all_posterior_samples_states
all_posterior_samples_states_sd3 = all_posterior_samples_states_sd




file_name = '/vsc-hard-mounts/leuven-data/343/vsc34314/5000 trials/50subjects_5000trials3.dil'
list_of_variable_names = (
    "all_lps3", "all_true_params3", "all_true_states3","all_inf_params3", "all_inf_states3",
    "all_emissions3", "all_inputs3",
    "all_posterior_samples_a3", "all_posterior_samples_sigmasq3", "all_posterior_samples_w3", "all_posterior_samples_states3","all_posterior_samples_states_sd3",
    "all_posterior_samples_a03", "all_posterior_samples_nu_a03",
    "all_posterior_samples_alpha3", "all_posterior_samples_beta3",
    "all_posterior_samples_w_03","all_posterior_samples_nu_w03",
    "true_w0", "true_nu_w", "true_a", "true_nu_a", "true_alpha", "true_beta",
    "num_trials","num_subjects", "num_inputs","num_datasets"
    )


with open(file_name, 'wb') as file:
    dill.dump(list_of_variable_names, file)  # Store all the names first
    for variable_name in list_of_variable_names:
        dill.dump(eval(variable_name), file) # Store the objects themselves


