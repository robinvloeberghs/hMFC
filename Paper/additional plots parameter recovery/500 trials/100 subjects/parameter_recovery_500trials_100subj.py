import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import sigmoid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import numpy as np
from fastprogress import progress_bar
from scipy.stats import spearmanr, pearsonr
import dill

from jax import vmap, lax

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from hmfc.model import HierarchicalBernoulliLDS
from hmfc.gibbs import gibbs_step
from hmfc.utils import convert_mean_to_std_ig_params
from hmfc.constants import A_MAX



def simulate_one_dataset(key):

    key = jr.PRNGKey(key) if isinstance(key, int) else key

    # Sample inputs
    inputs = tfd.Normal(0, 1.0).sample(sample_shape=(num_subjects, num_trials, num_inputs), seed=key)

    # REQUIRE THAT INPUTS ARE MEAN ZERO
    inputs -= inputs.mean(axis=0)

    # Sample true params, states, and emissions
    true_params, true_states, emissions = true_model.sample(key, inputs)
    masks = jnp.ones_like(emissions)

    """
    Make sure we sample reasonable trajectories (not too extreme values, but also not just random noise)
    """
    crit_value_extreme = 9 # absolute max
    crit_value_fluct = 3 # difference between min and max should be at least 3
    max_iter = 75

    too_extreme = jnp.where(jnp.max(jnp.abs(true_states),axis=1) > crit_value_extreme)[0]
    no_fluctuations = jnp.where(jnp.max(true_states, axis=1) - jnp.min(true_states, axis=1) < crit_value_fluct)[0]

    bad_subjects = jnp.concatenate([too_extreme,no_fluctuations])
    seed = 0 # resampling seed
    
    if bad_subjects.size > 0:
        for subj in bad_subjects:
            print(subj)
            
            # Call the true params and input from bad subject
            x_it = true_states[subj]
            u_i = inputs[subj]

            w_i = true_params['w'][subj]
            a_i = true_params['a'][subj]
            sigmasq_i = true_params['sigmasq'][subj]
            mu_x_i = true_params['mu_x'][subj]

            # Resample true_states and emissions from true params and inputs
            iter = 0
            while ((jnp.any(jnp.abs(x_it) > crit_value_extreme)) or ((jnp.max(x_it) - jnp.min(x_it)) < crit_value_fluct)):
  
                key = jr.PRNGKey(0)
                k1, k2, k3 = jr.split(key,3)

                # Sample latent states
                b_i = mu_x_i * (1 - a_i)
                sigmasq0_i = 1
                
                x_it0 = tfd.Normal(mu_x_i, jnp.sqrt(sigmasq0_i)).sample(seed=k1)
                def _step(x_itp, key):
                    x_itp1 = tfd.Normal(a_i * x_itp + b_i, jnp.sqrt(sigmasq_i)).sample(seed=key)
                    return x_itp1, x_itp
                _, x_it = lax.scan(_step, x_it0, jr.split(k2, num_trials))

                # Sample emissions
                y_it = tfd.Bernoulli(x_it + u_i @ w_i).sample(seed=k3)

                iter=iter+1

                if iter == max_iter: # if stuck in a loop, resample parameters
                    
                    key = jr.PRNGKey(seed)
                    k1, k2, k3 = jr.split(key,3)
                    # sample new params
                    print('new params sampled!')
                    a_i = tfd.TruncatedNormal(true_mu_a,
                                              true_sigma_a,
                                              0.0, A_MAX).sample(seed=k1)

                    sigmasq_i = tfd.InverseGamma(*convert_mean_to_std_ig_params(true_mu_sigmasq, true_beta_sigmasq)).sample(seed=k2)

                    mu_x_i = tfd.Normal(0, true_sigma_mu_x).sample(seed=k3)

                    # save new params
                    true_params['a'] = true_params['a'].at[subj].set(a_i)
                    true_params['sigmasq'] = true_params['sigmasq'].at[subj].set(sigmasq_i)
                    true_params['mu_x'] = true_params['mu_x'].at[subj].set(mu_x_i)

                    # change seed such that if we resample the parameter are not the same
                    seed=seed+1
                    iter = 0 

            # Save new true_states and emissions

            true_states = true_states.at[subj].set(x_it)
            emissions = emissions.at[subj].set(y_it)


    return true_params, true_states, emissions, inputs, masks



def fit_model(key, emissions, inputs, masks):
    
    key = jr.PRNGKey(key) if isinstance(key, int) else key

    # Initialize model
    model = HierarchicalBernoulliLDS(num_inputs,
                                     mu_a=init_mu_a, 
                                     sigma_a=init_sigma_a, 
                                     mu_w=init_mu_w, 
                                     sigma_w=init_sigma_w, 
                                     mu_sigmasq=init_mu_sigmasq,
                                     beta_sigmasq=init_beta_sigmasq, 
                                     sigma_mu_x=init_sigma_mu_x)
    
    params, states, _ = model.sample(key, inputs)

    # Fit model
    lps = jnp.zeros((num_iters,)) # log probability
    
    posterior_samples_mu_a = jnp.zeros((num_iters,))
    posterior_samples_sigma_a = jnp.zeros((num_iters,))
    posterior_samples_mu_w = jnp.zeros((num_iters, num_inputs))
    posterior_samples_sigma_w = jnp.zeros((num_iters, num_inputs))
    posterior_samples_mu_sigmasq = jnp.zeros((num_iters,))
    posterior_samples_beta_sigmasq = jnp.zeros((num_iters,))
    posterior_samples_sigma_mu_x = jnp.zeros((num_iters,))
    
    posterior_samples_a = jnp.zeros((num_iters, num_subjects))
    posterior_samples_sigmasq = jnp.zeros((num_iters, num_subjects))
    posterior_samples_w = jnp.zeros((num_iters, num_subjects, num_inputs))
    posterior_samples_mu_x = jnp.zeros((num_iters, num_subjects))
    
    for itr in progress_bar(range(num_iters)):
    
        this_key, key = jr.split(key)
        lp, states, params, model = gibbs_step(this_key, emissions, masks, states, inputs, params, model)
    
        lps = lps.at[itr].set(lp)
    
        posterior_samples_mu_a = posterior_samples_mu_a.at[itr].set(sigmoid(model.logit_mu_a))
        posterior_samples_sigma_a = posterior_samples_sigma_a.at[itr].set(jnp.exp(model.log_sigma_a))
        posterior_samples_mu_w = posterior_samples_mu_w.at[itr].set(model.mu_w)
        posterior_samples_sigma_w = posterior_samples_sigma_w.at[itr].set(jnp.exp(model.log_sigma_w))
        posterior_samples_mu_sigmasq = posterior_samples_mu_sigmasq.at[itr].set(jnp.exp(model.log_mu_sigmasq))
        posterior_samples_beta_sigmasq = posterior_samples_beta_sigmasq.at[itr].set(jnp.exp(model.log_beta_sigmasq))
        posterior_samples_sigma_mu_x = posterior_samples_sigma_mu_x.at[itr].set(jnp.exp(model.log_sigma_mu_x))
    
        posterior_samples_a = posterior_samples_a.at[itr].set(params['a'])
        posterior_samples_sigmasq = posterior_samples_sigmasq.at[itr].set(params['sigmasq'])
        posterior_samples_w = posterior_samples_w.at[itr].set(params['w'])
        posterior_samples_mu_x = posterior_samples_mu_x.at[itr].set(params['mu_x'])
        
        # due to memory issues we cannot save the states for all iterations
        # solution: we saved the summed posterior samples (requires less memory, and allows to calculate mean and variance, see later)
        # disadvantage: we already have to specify burn_in here without looking at the joint log prob trajectory
        if itr == burn_in:
            states_sum = states
            states_sum_squared = states**2
        elif itr > burn_in:
            states_sum += states
            states_sum_squared += states**2 # for calculation standard deviation later

    return lps, posterior_samples_mu_a, posterior_samples_sigma_a, posterior_samples_mu_w, posterior_samples_sigma_w, posterior_samples_mu_sigmasq, posterior_samples_beta_sigmasq, posterior_samples_sigma_mu_x, posterior_samples_a, posterior_samples_sigmasq, posterior_samples_w, posterior_samples_mu_x, states_sum, states_sum_squared  



# Set up some simulation variables
num_subjects = 100
num_trials = 500
num_iters = 1500
num_inputs = 3
num_datasets = 50
burn_in = 500

true_mu_a = 0.98
true_sigma_a = 0.03
true_mu_w = jnp.array([0.0, 0.2, -0.1])
true_sigma_w = jnp.array([1.0, 1.0, 1.0])
true_mu_sigmasq = 0.2
true_beta_sigmasq = 0.5
true_sigma_mu_x = 0.5



# Create our true model
true_model = HierarchicalBernoulliLDS(num_inputs,
                                      mu_a=true_mu_a, 
                                      sigma_a=true_sigma_a, 
                                      mu_w=true_mu_w, 
                                      sigma_w=true_sigma_w, 
                                      mu_sigmasq=true_mu_sigmasq,
                                      beta_sigmasq=true_beta_sigmasq, 
                                      sigma_mu_x=true_sigma_mu_x)



# Initialize model
init_mu_a = 0.90
init_sigma_a = 0.1
init_mu_w = jnp.zeros(num_inputs)
init_sigma_w = jnp.repeat(1.25, num_inputs)
init_mu_sigmasq = 0.3
init_beta_sigmasq = 0.6
init_sigma_mu_x = 0.6



# Run fitting procedure over datasets
all_true_params = []
all_true_states = []
all_emissions = []
all_inputs = []
all_lps = []

all_posterior_samples_mu_a = []
all_posterior_samples_sigma_a = []
all_posterior_samples_mu_w = []
all_posterior_samples_sigma_w = []
all_posterior_samples_mu_sigmasq = []
all_posterior_samples_beta_sigmasq = []
all_posterior_samples_sigma_mu_x = []

all_posterior_samples_a = []
all_posterior_samples_sigmasq = []
all_posterior_samples_w = []
all_posterior_samples_mu_x = []

all_posterior_samples_states = []
all_posterior_samples_states_sd = [] # standard deviation

keys = jr.split(jr.PRNGKey(1234), num_datasets)

for key in keys: #can't do this with vmap due to memory issues
    true_params, true_states, emissions, inputs, masks = simulate_one_dataset(key)
    lps, posterior_samples_mu_a, posterior_samples_sigma_a, posterior_samples_mu_w, posterior_samples_sigma_w, posterior_samples_mu_sigmasq, posterior_samples_beta_sigmasq, posterior_samples_sigma_mu_x, posterior_samples_a, posterior_samples_sigmasq, posterior_samples_w, posterior_samples_mu_x, states_sum, states_sum_squared = fit_model(key, emissions, inputs, masks)

    all_true_params.append(true_params)
    all_true_states.append(true_states)
    all_emissions.append(emissions)
    all_inputs.append(inputs)
    all_lps.append(lps)
    
    all_posterior_samples_mu_a.append(posterior_samples_mu_a)
    all_posterior_samples_sigma_a.append(posterior_samples_sigma_a)
    all_posterior_samples_mu_w.append(posterior_samples_mu_w)
    all_posterior_samples_sigma_w.append(posterior_samples_sigma_w)
    all_posterior_samples_mu_sigmasq.append(posterior_samples_mu_sigmasq)
    all_posterior_samples_beta_sigmasq.append(posterior_samples_beta_sigmasq)
    
    all_posterior_samples_sigma_mu_x.append(posterior_samples_sigma_mu_x)
    
    all_posterior_samples_a.append(posterior_samples_a)
    all_posterior_samples_sigmasq.append(posterior_samples_sigmasq)
    all_posterior_samples_w.append(posterior_samples_w)
    all_posterior_samples_mu_x.append(posterior_samples_mu_x)

    # based on the summed states we calculate the posterior mean and standard deviation
    mean_states = states_sum / (num_iters - burn_in)  # take average states over all iterations without burn_in
    sd_states = jnp.sqrt(states_sum_squared/(num_iters - burn_in)-mean_states**2) # chiastic formula for standard deviation
    
    all_posterior_samples_states.append(mean_states)
    all_posterior_samples_states_sd.append(sd_states)



# Save some stuff
# For recovery local parameters plot
average_recovery_states_all_datasets = []
median_recovery_states_all_datasets = []

for dataset in range(num_datasets):

  correlations_inferred_and_fitted_states = []
    
  for subject in range(num_subjects):
      r, _= spearmanr(all_true_states[dataset][subject], all_posterior_samples_states[dataset][subject])
      correlations_inferred_and_fitted_states.append(r)
  
  correlations_inferred_and_fitted_states = jnp.array(correlations_inferred_and_fitted_states)
  
  mean_value = jnp.mean(correlations_inferred_and_fitted_states)
  median_value = jnp.median(correlations_inferred_and_fitted_states)
  
  average_recovery_states_all_datasets.append(mean_value)
  median_recovery_states_all_datasets.append(median_value)

average_recovery_states_all_datasets = jnp.stack(average_recovery_states_all_datasets)
median_recovery_states_all_datasets = jnp.stack(median_recovery_states_all_datasets)


all_r_w0 = []
all_r_w1 = []
all_r_w2 = []
all_r_mu_x = []
all_r_a = []
all_r_sigmasq = []

for dataset in range(num_datasets):

    generative_w0 = all_true_params[dataset]['w'][:,0]
    generative_w1 = all_true_params[dataset]['w'][:,1]
    generative_w2 = all_true_params[dataset]['w'][:,2]
    generative_mu_x = all_true_params[dataset]['mu_x']
    generative_a = all_true_params[dataset]['a']
    generative_sigmasq = all_true_params[dataset]['sigmasq']

    
    fitted_w0 = jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,0], axis=0)
    fitted_w1 = jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,1], axis=0)
    fitted_w2 = jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,2], axis=0)
    fitted_mu_x = jnp.mean(all_posterior_samples_mu_x[dataset][burn_in:], axis=0)
    fitted_a = jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0)
    fitted_sigmasq = jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0)

    r_w0, p_w0 = spearmanr(generative_w0, fitted_w0)
    r_w1, p_w1 = spearmanr(generative_w1, fitted_w1)
    r_w2, p_w2 = spearmanr(generative_w2, fitted_w2)
    r_mu_x, p_mu_x = spearmanr(generative_mu_x, fitted_mu_x)
    r_a, p_a = spearmanr(generative_a, fitted_a)
    r_sigmasq, p_sigmasq = spearmanr(generative_sigmasq, fitted_sigmasq)

    all_r_w0.append(r_w0)
    all_r_w1.append(r_w1)
    all_r_w2.append(r_w2)
    all_r_mu_x.append(r_mu_x)
    all_r_a.append(r_a)
    all_r_sigmasq.append(r_sigmasq)


df_recovery_local = pd.DataFrame({
    'correlation_w0': all_r_w0,
    'correlation_w1': all_r_w1,
    'correlation_w2': all_r_w2,
    'correlation_mu_x': all_r_mu_x,
    'correlation_a': all_r_a,
    'correlation_sigmasq': all_r_sigmasq,
    'mean_correlation_states': average_recovery_states_all_datasets,
    'median_correlation_states': median_recovery_states_all_datasets
})

df_recovery_local.to_csv(f'{num_trials}trials_{num_subjects}subjects_recovery_localparams.csv', index=False)



# For heatmap plot
correlations_inferred_and_fitted_states = []
rmse = []
generative_a = []
generative_sigmasq = []
generative_mu_x = []
estimated_a = []
estimated_sigmasq = []
estimated_mu_x = []

for dataset in range(num_datasets):

    generative_a.append(all_true_params[dataset]['a'])
    generative_sigmasq.append(all_true_params[dataset]['sigmasq'])
    generative_mu_x.append(all_true_params[dataset]['mu_x'])
    
    estimated_a.append(jnp.mean(all_posterior_samples_a[dataset][burn_in:,:], axis=0))
    estimated_sigmasq.append(jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:,:], axis=0))
    estimated_mu_x.append(jnp.mean(all_posterior_samples_mu_x[dataset][burn_in:,:], axis=0))
        
    for subject in range(num_subjects):
        r_spearman, _= spearmanr(all_true_states[dataset][subject], all_posterior_samples_states[dataset][subject])
        correlations_inferred_and_fitted_states.append(r_spearman)

        # square root of mean squared error 
        rmse.append(jnp.sqrt(jnp.mean((all_true_states[dataset][subject]-all_posterior_samples_states[dataset][subject])**2)))

generative_a = jnp.hstack(generative_a)
generative_sigmasq = jnp.hstack(generative_sigmasq)
generative_mu_x = jnp.hstack(generative_mu_x)

estimated_a = jnp.hstack(estimated_a)
estimated_sigmasq = jnp.hstack(estimated_sigmasq)
estimated_mu_x = jnp.hstack(estimated_mu_x)

rmse = jnp.stack(rmse)

df_recovery_criterion_heatmap = pd.DataFrame({
    'correlation': correlations_inferred_and_fitted_states,
    'rmse': rmse,
    'generative_a': generative_a,
    'estimated_a': estimated_a,
    'generative_sigmasq': generative_sigmasq,
    'estimated_sigmasq': estimated_sigmasq,
    'generative_mu_x': generative_mu_x,
    'estimated_mu_x': estimated_mu_x
})

df_recovery_criterion_heatmap.to_csv(f'{num_trials}trials_{num_subjects}subjects_recovery_criterion_heatmap.csv', index=False)



# For posterior means plot
# Calculate posterior mean for each dataset

posterior_mean_mu_sigmasq = []
posterior_mean_beta_sigmasq = []
posterior_mean_sigma_mu_x = []
posterior_mean_mu_a = []
posterior_mean_sigma_a = []
posterior_mean_w0 = []
posterior_mean_w1 = []
posterior_mean_w2 = []
posterior_mean_sigma_w0 = []
posterior_mean_sigma_w1 = []
posterior_mean_sigma_w2 = []

posterior_var_mu_sigmasq = []
posterior_var_beta_sigmasq = []
posterior_var_sigma_mu_x = []
posterior_var_mu_a = []
posterior_var_sigma_a = []
posterior_var_w0 = []
posterior_var_w1 = []
posterior_var_w2 = []
posterior_var_sigma_w0 = []
posterior_var_sigma_w1 = []
posterior_var_sigma_w2 = []

posterior_q25_mu_sigmasq = []
posterior_q975_mu_sigmasq = []
posterior_q25_beta_sigmasq = []
posterior_q975_beta_sigmasq = []
posterior_q25_sigma_mu_x = []
posterior_q975_sigma_mu_x = []
posterior_q25_mu_a = []
posterior_q975_mu_a = []
posterior_q25_sigma_a = []
posterior_q975_sigma_a = []
posterior_q25_w0 = []
posterior_q975_w0 = []
posterior_q25_w1 = []
posterior_q975_w1 = []
posterior_q25_w2 = []
posterior_q975_w2 = []
posterior_q25_sigma_w0 = []
posterior_q975_sigma_w0 = []
posterior_q25_sigma_w1 = []
posterior_q975_sigma_w1 = []
posterior_q25_sigma_w2 = []
posterior_q975_sigma_w2 = []

# per dataset calculate the mean over each subject's true parameter
mean_subject_true_a = []
sd_subject_true_a = []
mean_subject_true_w0 = []
mean_subject_true_w1 = []
mean_subject_true_w2 = []
sd_subject_true_w0 = []
sd_subject_true_w1 = []
sd_subject_true_w2 = []

for dataset in range(num_datasets):
    
    posterior_mean_mu_sigmasq.append(jnp.mean(all_posterior_samples_mu_sigmasq[dataset][burn_in:]))
    posterior_mean_beta_sigmasq.append(jnp.mean(all_posterior_samples_beta_sigmasq[dataset][burn_in:]))
    posterior_mean_sigma_mu_x.append(jnp.mean(all_posterior_samples_sigma_mu_x[dataset][burn_in:]))
    posterior_mean_mu_a.append(jnp.mean(all_posterior_samples_mu_a[dataset][burn_in:]))
    posterior_mean_sigma_a.append(jnp.mean(all_posterior_samples_sigma_a[dataset][burn_in:]))
    posterior_mean_w0.append(jnp.mean(all_posterior_samples_mu_w[dataset][burn_in:,0]))
    posterior_mean_w1.append(jnp.mean(all_posterior_samples_mu_w[dataset][burn_in:,1]))
    posterior_mean_w2.append(jnp.mean(all_posterior_samples_mu_w[dataset][burn_in:,2]))
    posterior_mean_sigma_w0.append(jnp.mean(all_posterior_samples_sigma_w[dataset][burn_in:,0]))
    posterior_mean_sigma_w1.append(jnp.mean(all_posterior_samples_sigma_w[dataset][burn_in:,1]))
    posterior_mean_sigma_w2.append(jnp.mean(all_posterior_samples_sigma_w[dataset][burn_in:,2]))

    posterior_var_mu_sigmasq.append(jnp.var(all_posterior_samples_mu_sigmasq[dataset][burn_in:]))
    posterior_var_beta_sigmasq.append(jnp.var(all_posterior_samples_beta_sigmasq[dataset][burn_in:]))
    posterior_var_sigma_mu_x.append(jnp.mean(all_posterior_samples_sigma_mu_x[dataset][burn_in:]))
    posterior_var_mu_a.append(jnp.var(all_posterior_samples_mu_a[dataset][burn_in:]))
    posterior_var_sigma_a.append(jnp.var(all_posterior_samples_sigma_a[dataset][burn_in:]))
    posterior_var_w0.append(jnp.var(all_posterior_samples_mu_w[dataset][burn_in:,0]))
    posterior_var_w1.append(jnp.var(all_posterior_samples_mu_w[dataset][burn_in:,1]))
    posterior_var_w2.append(jnp.var(all_posterior_samples_mu_w[dataset][burn_in:,2]))
    posterior_var_sigma_w0.append(jnp.var(all_posterior_samples_sigma_w[dataset][burn_in:,0]))
    posterior_var_sigma_w1.append(jnp.var(all_posterior_samples_sigma_w[dataset][burn_in:,1]))
    posterior_var_sigma_w2.append(jnp.var(all_posterior_samples_sigma_w[dataset][burn_in:,2]))

    posterior_q25_mu_sigmasq.append(np.percentile(all_posterior_samples_mu_sigmasq[dataset][burn_in:], 2.5))
    posterior_q975_mu_sigmasq.append(np.percentile(all_posterior_samples_mu_sigmasq[dataset][burn_in:], 97.5))
    posterior_q25_beta_sigmasq.append(np.percentile(all_posterior_samples_beta_sigmasq[dataset][burn_in:], 2.5))
    posterior_q975_beta_sigmasq.append(np.percentile(all_posterior_samples_beta_sigmasq[dataset][burn_in:], 97.5))
    posterior_q25_sigma_mu_x.append(np.percentile(all_posterior_samples_sigma_mu_x[dataset][burn_in:], 2.5))
    posterior_q975_sigma_mu_x.append(np.percentile(all_posterior_samples_sigma_mu_x[dataset][burn_in:], 97.5))
    posterior_q25_mu_a.append(np.percentile(all_posterior_samples_mu_a[dataset][burn_in:], 2.5))
    posterior_q975_mu_a.append(np.percentile(all_posterior_samples_mu_a[dataset][burn_in:], 97.5))
    posterior_q25_sigma_a.append(np.percentile(all_posterior_samples_sigma_a[dataset][burn_in:], 2.5))
    posterior_q975_sigma_a.append(np.percentile(all_posterior_samples_sigma_a[dataset][burn_in:], 97.5))
    posterior_q25_w0.append(np.percentile(all_posterior_samples_mu_w[dataset][burn_in:,0], 2.5))
    posterior_q975_w0.append(np.percentile(all_posterior_samples_mu_w[dataset][burn_in:,0], 97.5))
    posterior_q25_w1.append(np.percentile(all_posterior_samples_mu_w[dataset][burn_in:,1], 2.5))
    posterior_q975_w1.append(np.percentile(all_posterior_samples_mu_w[dataset][burn_in:,1], 97.5))
    posterior_q25_w2.append(np.percentile(all_posterior_samples_mu_w[dataset][burn_in:,2], 2.5))
    posterior_q975_w2.append(np.percentile(all_posterior_samples_mu_w[dataset][burn_in:,2], 97.5))
    posterior_q25_sigma_w0.append(np.percentile(all_posterior_samples_sigma_w[dataset][burn_in:,0], 2.5))
    posterior_q975_sigma_w0.append(np.percentile(all_posterior_samples_sigma_w[dataset][burn_in:,0], 97.5))
    posterior_q25_sigma_w1.append(np.percentile(all_posterior_samples_sigma_w[dataset][burn_in:,1], 2.5))
    posterior_q975_sigma_w1.append(np.percentile(all_posterior_samples_sigma_w[dataset][burn_in:,1], 97.5))
    posterior_q25_sigma_w2.append(np.percentile(all_posterior_samples_sigma_w[dataset][burn_in:,2], 2.5))
    posterior_q975_sigma_w2.append(np.percentile(all_posterior_samples_sigma_w[dataset][burn_in:,2], 97.5))

    mean_subject_true_a.append(jnp.mean(all_true_params[dataset]['a']))
    sd_subject_true_a.append(jnp.std(all_true_params[dataset]['a']))
    
    mean_subject_true_w0.append(jnp.mean(all_true_params[dataset]['w'][:,0]))
    mean_subject_true_w1.append(jnp.mean(all_true_params[dataset]['w'][:,1]))
    mean_subject_true_w2.append(jnp.mean(all_true_params[dataset]['w'][:,2]))

    sd_subject_true_w0.append(jnp.std(all_true_params[dataset]['w'][:,0]))
    sd_subject_true_w1.append(jnp.std(all_true_params[dataset]['w'][:,1]))
    sd_subject_true_w2.append(jnp.std(all_true_params[dataset]['w'][:,2]))

    
df_posterior_means = pd.DataFrame({
    'mu_sigmasq': posterior_mean_mu_sigmasq,
    'beta_sigmasq': posterior_mean_beta_sigmasq,
    'sigma_mu_x': posterior_mean_sigma_mu_x,
    'mu_a': posterior_mean_mu_a,
    'sigma_a': posterior_mean_sigma_a,
    'w0': posterior_mean_w0,
    'w1': posterior_mean_w1,
    'w2': posterior_mean_w2,
    'sigma_w0': posterior_mean_sigma_w0,
    'sigma_w1': posterior_mean_sigma_w1,
    'sigma_w2': posterior_mean_sigma_w2,
    'mu_sigmasq_var': posterior_var_mu_sigmasq,
    'beta_sigmasq_var': posterior_var_beta_sigmasq,
    'mu_a_var': posterior_var_mu_a,
    'sigma_a_var': posterior_var_sigma_a,
    'w0_var': posterior_var_w0,
    'w1_var': posterior_var_w1,
    'w2_var': posterior_var_w2,
    'sigma_w0_var': posterior_var_sigma_w0,
    'sigma_w1_var': posterior_var_sigma_w1,
    'sigma_w2_var': posterior_var_sigma_w2,
    'posterior_q25_mu_sigmasq' : posterior_q25_mu_sigmasq,
    'posterior_q975_mu_sigmasq' : posterior_q975_mu_sigmasq,
    'posterior_q25_beta_sigmasq':posterior_q25_beta_sigmasq,
    'posterior_q975_beta_sigmasq':posterior_q975_beta_sigmasq,
    'posterior_q25_sigma_mu_x' : posterior_q25_sigma_mu_x,
    'posterior_q975_sigma_mu_x' : posterior_q975_sigma_mu_x,
    'posterior_q25_mu_a':posterior_q25_mu_a,
    'posterior_q975_mu_a':posterior_q975_mu_a,
    'posterior_q25_sigma_a':posterior_q25_sigma_a,
    'posterior_q975_sigma_a':posterior_q975_sigma_a,
    'posterior_q25_w0': posterior_q25_w0,
    'posterior_q975_w0':posterior_q975_w0,
    'posterior_q25_w1': posterior_q25_w1,
    'posterior_q975_w1':posterior_q975_w1,
    'posterior_q25_w2':posterior_q25_w2,
    'posterior_q975_w2':posterior_q975_w2,
    'posterior_q25_sigma_w0':posterior_q25_sigma_w0,
    'posterior_q975_sigma_w0':posterior_q975_sigma_w0,
    'posterior_q25_sigma_w1':posterior_q25_sigma_w1,
    'posterior_q975_sigma_w1':posterior_q975_sigma_w1,
    'posterior_q25_sigma_w2':posterior_q25_sigma_w2,
    'posterior_q975_sigma_w2':posterior_q975_sigma_w2,
    'mean_subject_true_a': mean_subject_true_a,
    'sd_subject_true_a': sd_subject_true_a,
    'mean_subject_true_w0': mean_subject_true_w0,
    'mean_subject_true_w1': mean_subject_true_w1,
    'mean_subject_true_w2': mean_subject_true_w2,
    'sd_subject_true_w0': sd_subject_true_w0,
    'sd_subject_true_w1': sd_subject_true_w1,
    'sd_subject_true_w2': sd_subject_true_w2,
    'num_trials': num_trials,
    'num_subjects': num_subjects
})

df_posterior_means.to_csv(f'{num_trials}trials_{num_subjects}subjects_posterior_means.csv', index=False)



# Save environment to dil file
file_name = '/vsc-hard-mounts/leuven-data/343/vsc34314/hMFC/100subjects_500trials.dil'

list_of_variable_names = (
    "all_lps", "all_true_params", "all_true_states",
    "all_posterior_samples_states","all_posterior_samples_states_sd",
    "all_posterior_samples_a", "all_posterior_samples_sigmasq", "all_posterior_samples_w", "all_posterior_samples_mu_x",
    "all_posterior_samples_mu_a", "all_posterior_samples_sigma_a",
    "all_posterior_samples_mu_sigmasq", "all_posterior_samples_beta_sigmasq",
    "all_posterior_samples_sigma_mu_x",
    "all_posterior_samples_mu_w","all_posterior_samples_sigma_w",
    
    )

with open(file_name, 'wb') as file:
    dill.dump(list_of_variable_names, file)  # Store all the names first
    for variable_name in list_of_variable_names:
        dill.dump(eval(variable_name), file) # Store the objects themselves
