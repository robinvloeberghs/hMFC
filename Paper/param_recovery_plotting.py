# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:28:44 2024

@author: Robin Vloeberghs
"""



"""
Code for plotting parameter recovery stuff (local params, global params, criterion fluctuations).
50 datasets were simulated, each with 50 subjects.
Number of trials per subject was varied (500, 1000, 2500, 5000 trials).
"""


import dill
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import jax.numpy as jnp
import numpy as np
import pandas as pd


num_trials = 500 # 500, 1000, 2500, 5000
burn_in = 250 # number of trials thrown away (make sure this is the same as in simulation script)
num_datasets = 50
num_subjects = 50


"""
Set seaborn style for plotting figures
"""

sns.set(style="ticks", context="paper",
        font="Arial",
        rc={"font.size": 25,
            "axes.titlesize": 25,
            "axes.labelsize": 27,
            "lines.linewidth": 1.5,
            "xtick.labelsize": 22,
            "ytick.labelsize": 15,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "axes.spines.right": False,
            "axes.spines.top": False
            })



"""
Load in dill file (contains the environment when simulation ran on cluster computer)
Due to memory issues on the cluster computer the simulations for high trial counts (2500 and 5000)
had to be split up, simulating the 50 datasets in multiple runs (25/25, or 10/10/10/10/10), hence the multiple dill files.
"""

def load_dill_file(file_path):
    with open(file_path, 'rb') as file:
        list_of_variable_names = dill.load(file)  # Get the names of stored objects
        for variable_name in list_of_variable_names:
            globals()[variable_name] = dill.load(file)  # Get the objects themselves




wd = "C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/hLDS/hLDS Gibbs/Parameter recovery/MH for sigmasq/"

file_paths_mapping = {
    (500, 25): [wd + "500 trials/25 subjects/25subjects_500trials.dil"],
    (500, 50): [wd + "500 trials/50 subjects/50subjects_500trials.dil"],
    (500, 100): [wd + "500 trials/100 subjects/100subjects_500trials.dil"],
    (500, 200): [wd + "500 trials/200 subjects/200subjects_500trials.dil"],
    (1000, 50): [wd + "1000 trials/50subjects_1000trials.dil"],
    (2500, 50): [wd + "2500 trials/50subjects_2500trials1.dil",
                 wd + "2500 trials/50subjects_2500trials2.dil"],
    (5000, 50): [wd + "5000 trials/50subjects_5000trials1.dil",
                 wd + "5000 trials/50subjects_5000trials2.dil",
                 wd + "5000 trials/50subjects_5000trials3.dil",
                 wd + "5000 trials/50subjects_5000trials4.dil",
                 wd + "5000 trials/50subjects_5000trials5.dil"]
}


# Get the file paths based on the number of trials and subjects
which_file = file_paths_mapping.get((num_trials, num_subjects))



# Load in the files
for file_path in which_file:
    load_dill_file(file_path)


# Merge the files in case the 50 datasets were simulated over multiple runs
# 2500 trials in two runs, 5000 trials in five runs
if num_trials == 2500:
    num_datasets = 50 # gets overwritten so respecify again
    all_emissions = all_emissions + all_emissions2
    all_inputs = all_inputs + all_inputs2    
    all_true_states = all_true_states + all_true_states2
    all_true_params = all_true_params + all_true_params2
    all_inf_states = all_inf_states + all_inf_states2
    all_inf_params = all_inf_params + all_inf_params2
    all_lps = all_lps + all_lps2 
    
    all_posterior_samples_w_0 = all_posterior_samples_w_0 + all_posterior_samples_w_02
    all_posterior_samples_nu_w0 = all_posterior_samples_nu_w0 + all_posterior_samples_nu_w02
    all_posterior_samples_a0 = all_posterior_samples_a0 + all_posterior_samples_a02
    all_posterior_samples_nu_a0 = all_posterior_samples_nu_a0 + all_posterior_samples_nu_a02
    all_posterior_samples_alpha = all_posterior_samples_alpha + all_posterior_samples_alpha2
    all_posterior_samples_beta = all_posterior_samples_beta + all_posterior_samples_beta2
    
    all_posterior_samples_a = all_posterior_samples_a + all_posterior_samples_a2
    all_posterior_samples_sigmasq = all_posterior_samples_sigmasq + all_posterior_samples_sigmasq2
    all_posterior_samples_w = all_posterior_samples_w + all_posterior_samples_w2
    # all_posterior_samples_states is already the averaged states over all iterations!
    # due to memory issues on the cluster computer the data before averaging could not be saved away
    # so basically, these are the posterior means of the states on each trial
    all_posterior_samples_states = all_posterior_samples_states + all_posterior_samples_states2 
    

if num_trials == 5000:
    num_datasets = 50 # gets overwritten so respecify again
    all_emissions = all_emissions + all_emissions2 + all_emissions3 + all_emissions4 + all_emissions5
    all_inputs = all_inputs + all_inputs2 + all_inputs3 + all_inputs4 + all_inputs5
    all_true_states = all_true_states + all_true_states2 + all_true_states3 + all_true_states4 + all_true_states5
    all_true_params = all_true_params + all_true_params2 + all_true_params3 + all_true_params4 + all_true_params5
    all_inf_states = all_inf_states + all_inf_states2 + all_inf_states3 + all_inf_states4 + all_inf_states5
    all_inf_params = all_inf_params + all_inf_params2 + all_inf_params3 + all_inf_params4 + all_inf_params5
    all_lps = all_lps + all_lps2 + all_lps3 + all_lps4 + all_lps5
    
    all_posterior_samples_w_0 = all_posterior_samples_w_0 + all_posterior_samples_w_02 + all_posterior_samples_w_03 + all_posterior_samples_w_04 + all_posterior_samples_w_05
    all_posterior_samples_nu_w0 = all_posterior_samples_nu_w0 + all_posterior_samples_nu_w02 + all_posterior_samples_nu_w03 + all_posterior_samples_nu_w04 + all_posterior_samples_nu_w05
    all_posterior_samples_a0 = all_posterior_samples_a0 + all_posterior_samples_a02 + all_posterior_samples_a03 + all_posterior_samples_a04 + all_posterior_samples_a05
    all_posterior_samples_nu_a0 = all_posterior_samples_nu_a0 + all_posterior_samples_nu_a02 + all_posterior_samples_nu_a03 + all_posterior_samples_nu_a04 + all_posterior_samples_nu_a05
    all_posterior_samples_alpha = all_posterior_samples_alpha + all_posterior_samples_alpha2 + all_posterior_samples_alpha3 + all_posterior_samples_alpha4 + all_posterior_samples_alpha5
    all_posterior_samples_beta = all_posterior_samples_beta + all_posterior_samples_beta2 + all_posterior_samples_beta3 + all_posterior_samples_beta4 + all_posterior_samples_beta5
    
    all_posterior_samples_a = all_posterior_samples_a + all_posterior_samples_a2 + all_posterior_samples_a3 + all_posterior_samples_a4 + all_posterior_samples_a5
    all_posterior_samples_sigmasq = all_posterior_samples_sigmasq + all_posterior_samples_sigmasq2 + all_posterior_samples_sigmasq3 + all_posterior_samples_sigmasq4 + all_posterior_samples_sigmasq5
    all_posterior_samples_w = all_posterior_samples_w + all_posterior_samples_w2 + all_posterior_samples_w3 + all_posterior_samples_w4 + all_posterior_samples_w5
    # all_posterior_samples_states is already the averaged states over all iterations!
    # due to memory issues on the cluster computer the data before averaging could not be saved away
    # so basically, these are the posterior means of the states on each trial
    all_posterior_samples_states = all_posterior_samples_states + all_posterior_samples_states2 + all_posterior_samples_states3 + all_posterior_samples_states4 + all_posterior_samples_states5 
    



"""
Check log joint probability to assess convergence and determine number of burn-in iterations
"""

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_log_joint_prob.pdf') as pdf:
  for dataset in range(num_datasets):
      
      plt.figure(figsize=(8, 6), dpi=600)
      plt.plot(jnp.stack(all_lps[dataset])/all_emissions[dataset].size) # normalized log joint prob
      plt.title(r"Dataset {:d}".format(dataset))
      plt.xlabel("Iteration")
      plt.ylabel("Log joint probability")

      pdf.savefig()
      plt.close()



"""
Estimated criterion fluctuations for one example dataset
"""

# Select a dataset
dataset = 0

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_estimated_states_one_dataset.pdf') as pdf:
  for subject in range(num_subjects):
      
      plt.figure(figsize=(8, 6), dpi=600)
      plt.plot(all_true_states[dataset][subject], label="True states")
      plt.plot(all_posterior_samples_states[dataset][subject], label="Inferred states")
      plt.fill_between(jnp.arange(num_trials),
                all_posterior_samples_states[dataset][subject] - 2 * all_posterior_samples_states_sd[dataset][subject],
                all_posterior_samples_states[dataset][subject] + 2 * all_posterior_samples_states_sd[dataset][subject], color='r', alpha=0.25)
      plt.xlabel("Trial")
      plt.ylabel("Criterion $x_t$")
      plt.title(r"Subject {:d}".format(subject))
      plt.legend(loc='lower right')
      r, p = spearmanr(all_true_states[dataset][subject], all_posterior_samples_states[dataset][subject])
      plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.90), xycoords='axes fraction')

      pdf.savefig()
      plt.close()


"""
Estimated criterion fluctuations for one example participant
"""

dataset = 0
subject = 0

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(all_true_states[dataset][subject], label="True states", linewidth=2.5, color='black')
plt.plot(all_posterior_samples_states[dataset][subject], label="Inferred states", linewidth=2.5)
plt.fill_between(jnp.arange(num_trials),
          all_posterior_samples_states[dataset][subject] - 2 * all_posterior_samples_states_sd[dataset][subject],
          all_posterior_samples_states[dataset][subject] + 2 * all_posterior_samples_states_sd[dataset][subject], color='r', alpha=0.25)
plt.xlabel("Trial $t$", fontsize=27)
plt.ylabel("Criterion $x_t$", fontsize=27)
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylim(-5.5,5.5)
#plt.title(r"Subject {:d}".format(subject))
plt.legend(loc='lower right', fontsize=17)
r, p = spearmanr(all_true_states[dataset][subject], all_posterior_samples_states[dataset][subject])
plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.90), xycoords='axes fraction', fontsize=22)

plt.show()



"""
Recovery criterion fluctuations all datasets
"""

average_recovery_states_all_datasets = []
median_recovery_states_all_datasets = []

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_recovery_states.pdf') as pdf:
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
    
    plt.figure(figsize=(8, 6), dpi=600)
    sns.histplot(data=correlations_inferred_and_fitted_states, bins=25)
    plt.annotate(f'Mean: {mean_value:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=18)
    plt.annotate(f'Median: {median_value:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=18)
    plt.xlabel('Spearman correlation generative and fitted criterion', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.title(r"Dataset {:d}".format(dataset))
    plt.xlim(0,1)
    
    pdf.savefig()
    plt.close()


"""
Recovery criterion fluctuations one example dataset
"""

dataset = 23


correlations_inferred_and_fitted_states = []

for subject in range(num_subjects):
    r, _= spearmanr(all_true_states[dataset][subject], all_posterior_samples_states[dataset][subject])
    correlations_inferred_and_fitted_states.append(r)

correlations_inferred_and_fitted_states = jnp.array(correlations_inferred_and_fitted_states)

mean_value = jnp.mean(correlations_inferred_and_fitted_states)
median_value = jnp.median(correlations_inferred_and_fitted_states)

average_recovery_states_all_datasets.append(mean_value)
median_recovery_states_all_datasets.append(median_value)

plt.figure(figsize=(8, 6), dpi=600)
sns.histplot(data=correlations_inferred_and_fitted_states, bins=15)

#plt.title("Correlation between inferred and fitted states over subjects")
plt.annotate(f'Mean: {mean_value:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=20)
plt.annotate(f'Median: {median_value:.2f}', xy=(0.05, 0.84), xycoords='axes fraction', fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Correlation true and fitted criterion trajectory', fontsize=27)
plt.ylabel('Count', fontsize=27)
#plt.title(r"Dataset {:d}".format(dataset))
plt.xlim(0,1)
plt.show()



"""
Compare true and inferred estimates of a_i averaged over iterations (minus the burn-in) for all datasets
"""


with PdfPages(f'{num_subjects}subjects_{num_trials}trials_recovery_a.pdf') as pdf:
  for dataset in range(num_datasets):
    
    a_min = min(all_true_params[dataset]['a'].min(), jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0).min())
    a_max = max(all_true_params[dataset]['a'].max(), jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0).max())
    
    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(all_true_params[dataset]['a'], jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0))
    plt.plot([a_min, a_max], [a_min, a_max], '-k')
    plt.xlabel("True $a$")
    plt.ylabel("Inferred $a$")
    plt.title(r"Dataset {:d}".format(dataset))
    plt.gca().set_aspect(1.0)
    r, p = spearmanr(all_true_params[dataset]['a'], jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0))
    plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction')
        
    pdf.savefig()
    plt.close()



"""
Compare true and inferred estimates of a_i averaged over iterations (minus the burn-in) for one example dataset
"""

dataset = 38

a_min = min(all_true_params[dataset]['a'].min(), jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0).min())
a_max = max(all_true_params[dataset]['a'].max(), jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0).max())

plt.figure(figsize=(8, 6), dpi=600)
plt.scatter(all_true_params[dataset]['a'], jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0), color="purple")
plt.plot([a_min, a_max], [a_min, a_max], '-k')
plt.xlabel("True $a$", size=30)
plt.ylabel("Inferred $a$",size=30)
plt.xticks(size=23)
plt.yticks(size=23)

#plt.title(r"Dataset {:d}".format(dataset))
plt.gca().set_aspect(1.0)
r, p = spearmanr(all_true_params[dataset]['a'], jnp.mean(all_posterior_samples_a[dataset][burn_in:], axis=0))
plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction')
plt.show() 



"""
Compare true and inferred estimates of sigma_sq averaged over iterations (minus the burn-in) for all datasets
"""

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_recovery_sigmasq.pdf') as pdf:
  for dataset in range(num_datasets):
    
    sigmasq_min = min(all_true_params[dataset]['sigmasq'].min(), jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0).min())
    sigmasq_max = max(all_true_params[dataset]['sigmasq'].max(), jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0).max())
    
    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(all_true_params[dataset]['sigmasq'], jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0))
    plt.plot([sigmasq_min, sigmasq_max], [sigmasq_min, sigmasq_max], '-k')
    plt.xlabel("True $\sigma^2$")
    plt.ylabel("Inferred $\sigma^2$")
    plt.title(r"Dataset {:d}".format(dataset))
    plt.gca().set_aspect(1.0)
    r, p = spearmanr(all_true_params[dataset]['sigmasq'], jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0))
    plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction')
        
    pdf.savefig()
    plt.close()



"""
Compare true and inferred estimates of sigma_sq averaged over iterations (minus the burn-in) for one example dataset
"""

dataset = 20

sigmasq_min = min(all_true_params[dataset]['sigmasq'].min(), jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0).min())
sigmasq_max = max(all_true_params[dataset]['sigmasq'].max(), jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0).max())

plt.figure(figsize=(8, 6), dpi=600)
plt.scatter(all_true_params[dataset]['sigmasq'], jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0), color="magenta")
plt.plot([sigmasq_min, sigmasq_max], [sigmasq_min, sigmasq_max], '-k')
plt.xticks(size=23)
plt.yticks(size=23)
plt.xlabel("True $\sigma^2$", size=30)
plt.ylabel("Inferred $\sigma^2$",size=30)
#plt.title(r"Dataset {:d}".format(dataset))
plt.gca().set_aspect(1.0)
r, p = spearmanr(all_true_params[dataset]['sigmasq'], jnp.mean(all_posterior_samples_sigmasq[dataset][burn_in:], axis=0))
plt.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction')
plt.show()
    


"""
Compare true and inferred estimates of w averaged over iterations (minus the burn-in) for all datasets
"""

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_recovery_w.pdf') as pdf:
  for dataset in range(num_datasets):
      
    w_min = min(all_true_params[dataset]["w"].min(), jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,:], axis=0).min())
    w_max = max(all_true_params[dataset]["w"].max(), jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,:], axis=0).max())
    
    fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
    fig.suptitle(r"Dataset {:d}".format(dataset), size=20)
    fig.subplots_adjust(top=1.3)
    for d, ax in enumerate(axs):
        ax.scatter(all_true_params[dataset]["w"][:, d], jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,d], axis=0))
        ax.set_xlabel(r"True $w_{:d}$".format(d), size=20)
        ax.set_ylabel(r"Inferred $w_{:d}$".format(d), size=20)
        ax.plot([w_min, w_max], [w_min, w_max], '-k')
        ax.set_aspect(1.0)
        r, p = spearmanr(all_true_params[dataset]["w"][:, d], jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,d], axis=0))
        ax.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=18)
        
    pdf.savefig()
    plt.close()


"""
Compare true and inferred estimates of w averaged over iterations (minus the burn-in) for one example dataset
"""

dataset = 2
colors = ['darkcyan', '#2ca02c', '#1f77b4', '#ff7f0e']
w_min = min(all_true_params[dataset]["w"].min(), jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,:], axis=0).min())
w_max = max(all_true_params[dataset]["w"].max(), jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,:], axis=0).max())

fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
#fig.suptitle(r"Dataset {:d}".format(dataset))
fig.subplots_adjust(top=1.3)

for d, ax in enumerate(axs):
    ax.scatter(all_true_params[dataset]["w"][:, d], jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,d], axis=0), color= colors[d])
    ax.set_xlabel(r"True $w_{:d}$".format(d), size=21)
    ax.set_ylabel(r"Inferred $w_{:d}$".format(d), size=21)
    ax.plot([w_min, w_max], [w_min, w_max], '-k')
    ax.set_aspect(1.0)
    r, p = spearmanr(all_true_params[dataset]["w"][:, d], jnp.mean(all_posterior_samples_w[dataset][burn_in:,:,d], axis=0))
    ax.annotate('r = {:.2f}'.format(r), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=18)
    



# ================================================================== #
# GLOBAL (HIERARCHICAL) PARAMETERS
# ================================================================== #

"""
Posterior distributions of w0 (mean of normal distribution for w_i)
"""



with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_w0.pdf') as pdf:
    for dataset in range(num_datasets):
        fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
        fig.suptitle("Posterior distributions", fontsize=16)
        fig.text(0.5, 0.93, "Dataset {:d}".format(dataset), ha='center', va='top', fontsize=12)
        
        mean_true_per_subject_w = jnp.mean(all_true_params[dataset]['w'], axis=0)
        for d, ax in enumerate(axs):
            ax.hist(all_posterior_samples_w_0[dataset][burn_in:, d], bins=30)
            ax.axvline(x=true_w0[d], color='black', linewidth=3)
            ax.axvline(x=mean_true_per_subject_w[d], color='red', linewidth=3)
            
            if d == 0:
                ax.set_ylabel("Count", fontsize=15)
            ax.set_xlabel(r"$w_{:d}$".format(d), fontsize=15)
            
        fig.legend([r"True $w$", r'Mean true per-subject $w_i$'])
        
        pdf.savefig()
        plt.close()
        

# Same as above but now shown over time
with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_w0_over_iterations.pdf') as pdf:
    for dataset in range(num_datasets):
        fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
        fig.suptitle("Posterior distributions", fontsize=16)
        fig.text(0.5, 0.93, "Dataset {:d}".format(dataset), ha='center', va='top', fontsize=12)
        
        mean_true_per_subject_w = jnp.mean(all_true_params[dataset]['w'], axis=0)
        for d, ax in enumerate(axs):
            ax.plot(all_posterior_samples_w_0[dataset][burn_in:, d])
            ax.axhline(y=true_w0[d], color='black')
            ax.axhline(y=mean_true_per_subject_w[d], color='red')
            ax.set_xlabel("Iteration", fontsize=15)
            ax.set_ylabel(r"$w_{:d}$".format(d), fontsize=15)
            
        fig.legend([Line2D([0], [0], color='black'), Line2D([0], [0], color='red')],[r"True $w$", r'Mean true per-subject $w_i$']) # fix to have black line in legend (without it is blue)

        pdf.savefig()
        plt.close()



"""
Posterior distributions of nu_w0 (variance of normal distribution for w_i)
"""

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_nu_w0.pdf') as pdf:
    for dataset in range(num_datasets):
        fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
        fig.suptitle("Posterior distributions", fontsize=16)
        fig.text(0.5, 0.93, "Dataset {:d}".format(dataset), ha='center', va='top', fontsize=12)
        
        sd_true_per_subject_w = jnp.std(all_true_params[dataset]['w'], axis=0)
        for d, ax in enumerate(axs):
            ax.hist(all_posterior_samples_nu_w0[dataset][burn_in:, d], bins=30)
            ax.axvline(x=true_nu_w[d], color='black')
            ax.axvline(x=sd_true_per_subject_w[d], color='red')
            if d == 0:
                ax.set_ylabel("Count", fontsize=15)
            ax.set_xlabel(r"$\nu_{{w_{:d}}}$".format(d), fontsize=15)

        fig.legend([r"True $\nu_w$", r'Sd true per-subject $w_i$'])
        
        pdf.savefig()
        plt.close()



# Same as above but now shown over time
with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_nu_w0_over_iterations.pdf') as pdf:
    for dataset in range(num_datasets):
        fig, axs = plt.subplots(1, num_inputs, sharey=True, figsize=(10, 5), dpi=600)
        fig.suptitle("Posterior distributions", fontsize=16)
        fig.text(0.5, 0.93, "Dataset {:d}".format(dataset), ha='center', va='top', fontsize=12)
        
        sd_true_per_subject_w = jnp.std(all_true_params[dataset]['w'], axis=0)
        for d, ax in enumerate(axs):
            ax.plot(all_posterior_samples_nu_w0[dataset][burn_in:, d])
            ax.axhline(y=true_nu_w[d], color='black')
            ax.axhline(y=sd_true_per_subject_w[d], color='red')
            ax.set_xlabel("Iteration", fontsize=15)
            ax.set_ylabel(r"$\nu_{{w_{:d}}}$".format(d), fontsize=15)

        fig.legend([Line2D([0], [0], color='black'), Line2D([0], [0], color='red')],[r"True $\nu_w$", r'Mean true per-subject $w_i$']) # fix to have black line in legend (without it is blue)

        pdf.savefig()
        plt.close()



"""
Posterior distributions of a0 (mean of truncated normal for a_i)
"""


with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_a0.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.hist(all_posterior_samples_a0[dataset][burn_in:], bins=30)
    plt.axvline(true_a, color='black', label="True $a_0$")
    plt.axvline(jnp.mean(all_true_params[dataset]['a']), color='red', label="Mean true per-subject $a_i$")
    plt.xlabel("$a_0$")
    plt.ylabel("Count")
    plt.legend(loc='upper right')
    plt.suptitle("Posterior distributions", fontsize=16)
    plt.title("Dataset {:d}".format(dataset), fontsize=12)

    pdf.savefig()
    plt.close()
    
# Same as above but now shown over time
with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_a0_over_iterations.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(all_posterior_samples_a0[dataset][burn_in:])
    plt.axhline(true_a, color='black', label="True $a_0$")
    plt.axhline(jnp.mean(all_true_params[dataset]['a']), color='red', label="Mean true per-subject $a_i$")
    plt.xlabel("Iteration")
    plt.ylabel("$a_0$")
    plt.legend(loc='upper right')
    plt.title(r"Dataset {:d}".format(dataset))
        
    pdf.savefig()
    plt.close()


"""
Posterior distributions of nu_a0 (standard deviation of truncated normal for a_i)
"""


with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_nu_a0.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.hist(all_posterior_samples_nu_a0[dataset][burn_in:], bins=30)
    plt.axvline(true_nu_a, color='black', label=r"True $\nu_a$")
    plt.axvline(jnp.std(all_true_params[dataset]['a']), color='red', label="Std true per-subject $a_i$")
    plt.ylabel("Count")
    plt.xlabel(r"$\nu_a$")
    plt.legend(loc='upper right')
    plt.title(r"Dataset {:d}".format(dataset))
        
    pdf.savefig()
    plt.close()

# Same as above but now shown over time
with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_nu_a0_over_iterations.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(all_posterior_samples_nu_a0[dataset][burn_in:])
    plt.axhline(true_nu_a, color='black', label=r"True $\nu_a$")
    plt.axhline(jnp.std(all_true_params[dataset]['a']), color='red', label="Std true per-subject $a_i$")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\nu_a$")
    plt.legend(loc='upper right')
    plt.title(r"Dataset {:d}".format(dataset))
        
    pdf.savefig()
    plt.close()


"""
Posterior distributions of alpha (shape parameter of inverse gamma for sigmasq)
"""

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_alpha.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.hist(all_posterior_samples_alpha[dataset][burn_in:], bins=30)
    plt.axvline(true_alpha, color='black', label=r"True $\alpha$")
    plt.ylabel("Count")
    plt.xlabel(r"$\alpha$")
    plt.legend(loc='upper right')
    plt.title(r"Dataset {:d}".format(dataset))
        
    pdf.savefig()
    plt.close()

# Same as above but now shown over time
with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_alpha_over_iterations.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(all_posterior_samples_alpha[dataset][burn_in:])
    plt.axhline(true_alpha, color='black', label=r"True $\alpha$")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\alpha$")
    plt.legend(loc='upper right')
    plt.title(r"Dataset {:d}".format(dataset))
        
    pdf.savefig()
    plt.close()



"""
Posterior distributions of beta (scale parameter of inverse gamma for sigmasq)
"""

with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_beta.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.hist(all_posterior_samples_beta[dataset][burn_in:], bins=30)
    plt.axvline(true_beta, color='black', label=r"True $\beta$")
    plt.ylabel("Count")
    plt.xlabel(r"$\beta$")
    plt.legend(loc='upper right')
    plt.title(r"Dataset {:d}".format(dataset))
        
    pdf.savefig()
    plt.close()
    
    
    

# Same as above but now shown over time
with PdfPages(f'{num_subjects}subjects_{num_trials}trials_posterior_beta_over_iterations.pdf') as pdf:
  for dataset in range(num_datasets):
      
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(all_posterior_samples_beta[dataset][burn_in:])
    plt.axhline(true_beta, color='black', label=r"True $\beta$")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\beta$")
    plt.legend(loc='upper right')
    plt.title(r"Dataset {:d}".format(dataset))
        
    pdf.savefig()
    plt.close()



"""
Check for how many datasets the TRUE global (hierarchical) parameter value is within the 95% credible interval
"""

def within_95_ci(posterior_samples,
                 true_param,
                 w_true=False,
                 variable_index=0): # indicate which w
    
    inside_outside = []
    for dataset in range(num_datasets):
        posterior = posterior_samples[dataset][burn_in:]
        if w_true: # to bypass different data formatting for w0 and nu_w0
            posterior = posterior_samples[dataset][burn_in:,variable_index]
            
        if jnp.percentile(posterior, 2.5) < true_param < jnp.percentile(posterior, 97.5):
            inside_outside.append(1)
        else:
            inside_outside.append(0)
            
    return sum(inside_outside)



alpha = within_95_ci(all_posterior_samples_alpha, true_alpha)
beta = within_95_ci(all_posterior_samples_beta, true_beta)

a0 = within_95_ci(all_posterior_samples_a0, true_a)
nu_a0 = within_95_ci(all_posterior_samples_nu_a0, true_nu_a)

w0 = within_95_ci(all_posterior_samples_w_0, true_w0[0], w_true=True, variable_index=0)
w1 = within_95_ci(all_posterior_samples_w_0, true_w0[1], w_true=True, variable_index=1)
w2 = within_95_ci(all_posterior_samples_w_0, true_w0[2], w_true=True, variable_index=2)
w3 = within_95_ci(all_posterior_samples_w_0, true_w0[3], w_true=True, variable_index=3)

nu_w0 = within_95_ci(all_posterior_samples_nu_w0, true_nu_w[0], w_true=True, variable_index=0)
nu_w1 = within_95_ci(all_posterior_samples_nu_w0, true_nu_w[1], w_true=True, variable_index=1)
nu_w2 = within_95_ci(all_posterior_samples_nu_w0, true_nu_w[2], w_true=True, variable_index=2)
nu_w3 = within_95_ci(all_posterior_samples_nu_w0, true_nu_w[3], w_true=True, variable_index=3)



"""
Check for how many datasets the average per-subject true parameter value is within the 95% credible interval
"""

# Note that we don't do this for alpha and beta (from inverse gamma for sigma_sq) 
# Would require some estimation based on true per-subject sigma_sq to obtain a proxy for alpha and beta
# If needed, use method of moments?

def within_95_ci_subject_w(posterior_samples,
                           variable_index=0,
                           mean=True): # if False then true value is standard deviation of per-subject true w's
    inside_outside = []
    for dataset in range(num_datasets):
        posterior = posterior_samples[dataset][burn_in:,variable_index]
        true_value = jnp.std(all_true_params[dataset]['w'][:,variable_index])
        if mean:
            true_value = jnp.mean(all_true_params[dataset]['w'][:,variable_index])
                
        if np.percentile(posterior, 2.5) < true_value < np.percentile(posterior, 97.5):
            inside_outside.append(1)
        else:
            inside_outside.append(0)
            
    return sum(inside_outside)



def within_95_ci_subject_a(posterior_samples,
                           mean=True): # if False then true value is standard deviation of per-subject true a's
    
    inside_outside = []
    for dataset in range(num_datasets):
        posterior = posterior_samples[dataset][burn_in:]
        true_value = jnp.std(all_true_params[dataset]['a'])
        if mean:
            true_value = jnp.mean(all_true_params[dataset]['a'])
                
        if np.percentile(posterior, 2.5) < true_value < np.percentile(posterior, 97.5):
            inside_outside.append(1)
        else:
            inside_outside.append(0)
            
    return sum(inside_outside)


a0_subject = within_95_ci_subject_a(all_posterior_samples_a0)
nu_a0_subject = within_95_ci_subject_a(all_posterior_samples_nu_a0, mean=False)

w0_subject = within_95_ci_subject_w(all_posterior_samples_w_0, variable_index=0)
w1_subject = within_95_ci_subject_w(all_posterior_samples_w_0, variable_index=1)
w2_subject = within_95_ci_subject_w(all_posterior_samples_w_0, variable_index=2)
w3_subject = within_95_ci_subject_w(all_posterior_samples_w_0, variable_index=3)

nu_w0_subject = within_95_ci_subject_w(all_posterior_samples_nu_w0, variable_index=0, mean=False)
nu_w1_subject = within_95_ci_subject_w(all_posterior_samples_nu_w0, variable_index=1, mean=False)
nu_w2_subject = within_95_ci_subject_w(all_posterior_samples_nu_w0, variable_index=2, mean=False)
nu_w3_subject = within_95_ci_subject_w(all_posterior_samples_nu_w0, variable_index=3, mean=False)



"""
Plotting recovery local parameters
"""


jitter_strength = 0.05
plot_with_each_dataset_as_point = True


# Read in files (burn in trials already removed)
wd = "C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/hLDS/hLDS Gibbs/Parameter recovery/MH for sigmasq/"

data500 = pd.read_csv(wd + '500 trials/50 subjects/500trials_50subjects_recovery_localparams.csv')
data1000 = pd.read_csv(wd + '1000 trials/1000trials_50subjects_recovery_localparams.csv')
data2500 = pd.read_csv(wd + '2500 trials/2500trials_50subjects_recovery_localparams.csv')
data5000 = pd.read_csv(wd + '5000 trials/5000trials_50subjects_recovery_localparams.csv')


# Define function to read data and calculate means
def process_data(filename):
    data = filename
    means = {
        'w0': np.mean(data["correlation_w0"]),
        'w1': np.mean(data["correlation_w1"]),
        'w2': np.mean(data["correlation_w2"]),
        'w3': np.mean(data["correlation_w3"]),
        'a': np.mean(data["correlation_a"]),
        'sigmasq': np.mean(data["correlation_sigmasq"]),
        'criterion': np.mean(data["mean_correlation_states"])
    }
    std_errors = {
        'w0': np.std(data["correlation_w0"])/np.sqrt(num_datasets),
        'w1': np.std(data["correlation_w1"])/np.sqrt(num_datasets),
        'w2': np.std(data["correlation_w2"])/np.sqrt(num_datasets),
        'w3': np.std(data["correlation_w3"])/np.sqrt(num_datasets),
        'a': np.std(data["correlation_a"])/np.sqrt(num_datasets),
        'sigmasq': np.std(data["correlation_sigmasq"])/np.sqrt(num_datasets),
        'criterion': np.std(data["mean_correlation_states"])/np.sqrt(num_datasets)
    }
    return data, means, std_errors


# File names
files = [data500, data1000, data2500, data5000]

# Process data for each file
data = [process_data(file) for file in files]

# Extract values
names = ['500', '1000', '2500', '5000']
individual_data = {key: [data[i][0]["correlation_" + key] for i in range(4)] for key in ['w0', 'w1', 'w2', 'w3', 'a', 'sigmasq']}
values = {key: [data[i][1][key] for i in range(4)] for key in ['w0', 'w1', 'w2', 'w3', 'a', 'sigmasq']}
std_errors = {key: [data[i][2][key] for i in range(4)] for key in ['w0', 'w1', 'w2', 'w3', 'a', 'sigmasq']}

# Define legend labels with symbols
legend_labels = {
    'w0': r"$w_0$",
    'w1': r"$w_1$",
    'w2': r"$w_2$",
    'w3': r"$w_3$",
    'a': r"$a$",
    'sigmasq': r"$\sigma^2$",
}


plt.figure(figsize=(6, 6), dpi=600)
colors = ['darkcyan', '#2ca02c', '#1f77b4', '#ff7f0e', 'purple', '#A6A6A6']

for i, (key, value) in enumerate(values.items()):
    plt.errorbar(names, value, yerr=std_errors[key], label=legend_labels[key], marker='o', color=colors[i % len(colors)])
    if plot_with_each_dataset_as_point: # Plot individual data points with jitter
        for j in range(len(names)):  # Loop over the 4 data files
            x_jitter = np.random.normal(loc=j, scale=jitter_strength, size=num_datasets)
            plt.scatter(x_jitter, individual_data[key][j], color=colors[i % len(colors)], alpha=0.15, s=10)

plt.xlabel('Number of trials per subject (50 subjects)', size=15)
plt.ylabel('Spearman correlation', size=15)
plt.ylim(0,1.05)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), fontsize=15)

plt.show()




"""
Plotting recovery criterion trajectory
"""


jitter_strength_criterion = 0.1
plot_with_each_dataset_as_point = True # if False then all individual subjects are shown


# Read in extra files (contains recovery correlation of criterion trajectory for each subject, burn in trials already removed)
data500_subjects = pd.read_csv(wd + '500 trials/50 subjects/500trials_50subjects_recovery_criterion_heatmap.csv')
data1000_subjects = pd.read_csv(wd + '1000 trials/1000trials_50subjects_recovery_criterion_heatmap.csv')
data2500_subjects = pd.read_csv(wd + '2500 trials/2500trials_50subjects_recovery_criterion_heatmap.csv')
data5000_subjects = pd.read_csv(wd + '5000 trials/5000trials_50subjects_recovery_criterion_heatmap.csv')


files = [data500, data1000, data2500, data5000]
files_subjects = [data500_subjects, data1000_subjects, data2500_subjects, data5000_subjects]

# Process data for each file
data = [process_data(file) for file in files]

# Extract values
names = ['500', '1000', '2500', '5000']
values = {key: [data[i][1][key] for i in range(4)] for key in ['criterion']}
std_errors = {key: [data[i][2][key] for i in range(4)] for key in ['criterion']}


plt.figure(figsize=(6, 6), dpi=600)

if plot_with_each_dataset_as_point: # Plot datasets as points
    for i, file in enumerate(files):
        jitter = np.random.normal(0, jitter_strength_criterion, num_datasets)
        plt.scatter(np.array([i + 1] * num_datasets) + jitter, file["mean_correlation_states"], alpha=0.3, color="gray")
else: # Plot all individuals as points
    for i, file in enumerate(files_subjects):
        jitter = np.random.normal(0, jitter_strength_criterion, num_datasets*50)
        plt.scatter(np.array([i + 1] * num_datasets * 50) + jitter, file['correlation'], alpha=0.1, color="gray")
        
# Plot mean values with error bars
for key, value in values.items():
    plt.errorbar(range(1, len(names) + 1), value, yerr=std_errors[key], marker='o', markersize=10)

plt.xticks(ticks=range(1, len(names) + 1), labels=names, fontsize=20)
plt.xlabel('Number of trials per subject (50 subjects)', size=27)
plt.ylabel('Spearman correlation', size=27)
plt.ylim(0.0, 1.05)
plt.yticks(fontsize=20)

plt.show()



"""
Plotting heatmap
"""


# Contains recovery correlation of criterion trajectory for each subject
# Each file consists of 50 simulated datasets, each with 50 subjects (burn in trials already removed)
data500_subjects = pd.read_csv(wd + '500 trials/50 subjects/500trials_50subjects_recovery_criterion_heatmap.csv')
data1000_subjects = pd.read_csv(wd + '1000 trials/1000trials_50subjects_recovery_criterion_heatmap.csv')
data2500_subjects = pd.read_csv(wd + '2500 trials/2500trials_50subjects_recovery_criterion_heatmap.csv')
data5000_subjects = pd.read_csv(wd + '5000 trials/5000trials_50subjects_recovery_criterion_heatmap.csv')

data500_25subjects = pd.read_csv(wd + '500 trials/25 subjects/500trials_25subjects_recovery_criterion_heatmap.csv')
data500_100subjects = pd.read_csv(wd + '500 trials/100 subjects/500trials_100subjects_recovery_criterion_heatmap.csv')
data500_200subjects = pd.read_csv(wd + '500 trials/200 subjects/500trials_200subjects_recovery_criterion_heatmap.csv')

combined_data_basic = pd.concat([data500_subjects, data1000_subjects, data2500_subjects, data5000_subjects], axis=0)
combined_data_all = pd.concat([data500_subjects, data1000_subjects, data2500_subjects, data5000_subjects, data500_25subjects, data500_100subjects, data500_200subjects], axis=0)


# Choose which dataframe
data = combined_data_basic
#data = combined_data_all


# Plot with generative parameter values
plt.figure(figsize=(8, 6), dpi=600)
sns.scatterplot(x = data["generative_a"], y = data["generative_sigmasq"], hue=data['correlation'], palette='viridis', s=25, alpha=.2)
plt.title('Recoverability criterion trajectory')
plt.xlabel('True $a$')
plt.ylabel(r'True $\sigma^2$')
plt.show()

# Plot with estimated parameter values
plt.figure(figsize=(8, 6), dpi=600)
sns.scatterplot(x = data["estimated_a"], y = data["estimated_sigmasq"], hue=data['correlation'], palette='viridis', s=25, alpha=.2)
plt.title('Recoverability criterion trajectory')
plt.xlabel('Estimated $a$')
plt.ylabel('Estimated $\sigma^2$')
plt.show()





"""
Plots for posterior distributions for the group (global) parameters
"""

# Load in the different dill files
# Because variables within these dill files have the same names we have to rename them
num_trials=500

file_25 = file_paths_mapping.get((num_trials, 25))
file_50 = file_paths_mapping.get((num_trials, 50))
file_100 = file_paths_mapping.get((num_trials, 100))
file_200 = file_paths_mapping.get((num_trials, 200))

load_dill_file(file_25[0])

posterior_samples_alpha_25 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_25 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_25 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_25 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_25 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_25 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]

load_dill_file(file_50[0])

posterior_samples_alpha_50 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_50 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_50 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_50 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_50 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_50 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]

load_dill_file(file_100[0])

posterior_samples_alpha_100 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_100 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_100 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_100 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_100 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_100 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]


load_dill_file(file_200[0])

posterior_samples_alpha_200 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_200 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_200 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_200 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_200 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_200 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]



true_a0 = 0.99
true_nu_a0 = 0.025
true_w0 = jnp.array([0.0, 0.2, -0.3, 0.6])       
true_nu_w = jnp.array([1.0, 1.0, 1.0, 1.0])
true_alpha = 5.0
true_beta = 5.0 * 0.3**2

# Contains the posterior means and 95% CI for each dataset (calculated without the burn in trials)
data500_posterior_mean = pd.read_csv(wd + '500 trials/50 subjects/500trials_50subjects_posterior_means.csv')
data1000_posterior_mean = pd.read_csv(wd + '1000 trials/1000trials_50subjects_posterior_means.csv')
data2500_posterior_mean = pd.read_csv(wd + '2500 trials/2500trials_50subjects_posterior_means.csv')
data5000_posterior_mean = pd.read_csv(wd + '5000 trials/5000trials_50subjects_posterior_means.csv')

data500_25subj_posterior_mean = pd.read_csv(wd + '500 trials/25 subjects/500trials_25subjects_posterior_means.csv')
data500_50subj_posterior_mean = pd.read_csv(wd + '500 trials/50 subjects/500trials_50subjects_posterior_means.csv')
data500_100subj_posterior_mean = pd.read_csv(wd + '500 trials/100 subjects/500trials_100subjects_posterior_means.csv')
data500_200subj_posterior_mean = pd.read_csv(wd + '500 trials/200 subjects/500trials_200subjects_posterior_means.csv')


all_posterior_means = pd.concat([data500_posterior_mean, data1000_posterior_mean, data2500_posterior_mean, data5000_posterior_mean], axis=0)
all_posterior_means_subj = pd.concat([data500_25subj_posterior_mean, data500_50subj_posterior_mean, data500_100subj_posterior_mean, data500_200subj_posterior_mean], axis=0)


df_posterior_mean_subj = pd.concat([data500_25subj_posterior_mean, data500_posterior_mean, data500_100subj_posterior_mean], axis=0)
df_posterior_mean_subj['dataset'] = np.arange(0,num_datasets).tolist() * 3 # 3 levels of number of subjects



posterior_a0 = jnp.array([posterior_samples_a0_25,posterior_samples_a0_50,posterior_samples_a0_100]).flatten()
posterior_nu_a0 = jnp.array([posterior_samples_nu_a0_25,posterior_samples_nu_a0_50,posterior_samples_nu_a0_100]).flatten()
posterior_alpha = jnp.array([posterior_samples_alpha_25,posterior_samples_alpha_50,posterior_samples_alpha_100]).flatten()
posterior_beta = jnp.array([posterior_samples_beta_25,posterior_samples_beta_50,posterior_samples_beta_100]).flatten()
posterior_w0 = jnp.array([jnp.stack(posterior_samples_w0_25)[:,:,0], jnp.stack(posterior_samples_w0_50)[:,:,0], jnp.stack(posterior_samples_w0_100)[:,:,0]]).flatten()
posterior_nu_w0 = jnp.array([jnp.stack(posterior_samples_nu_w0_25)[:,:,0], jnp.stack(posterior_samples_nu_w0_50)[:,:,0], jnp.stack(posterior_samples_nu_w0_100)[:,:,0]]).flatten()


corrected_posterior_a0 = (jnp.array([posterior_samples_a0_25,posterior_samples_a0_50,posterior_samples_a0_100]) - jnp.array(df_posterior_mean_subj['mean_subject_true_a']).reshape(3,50,1)).flatten()
corrected_posterior_nu_a0 = (jnp.array([posterior_samples_nu_a0_25,posterior_samples_nu_a0_50,posterior_samples_nu_a0_100]) - jnp.array(df_posterior_mean_subj['sd_subject_true_a']).reshape(3,50,1)).flatten()
corrected_posterior_alpha = (jnp.array([posterior_samples_alpha_25,posterior_samples_alpha_50,posterior_samples_alpha_100]) - true_alpha).flatten()
corrected_posterior_beta = (jnp.array([posterior_samples_beta_25,posterior_samples_beta_50,posterior_samples_beta_100]) - true_beta).flatten()
# we take the intercept (doesn't matter, plots look the same for all input variables)
corrected_posterior_w0 = (jnp.array([jnp.stack(posterior_samples_w0_25)[:,:,0], jnp.stack(posterior_samples_w0_50)[:,:,0], jnp.stack(posterior_samples_w0_100)[:,:,0]]) - jnp.array(df_posterior_mean_subj['mean_subject_true_w0']).reshape(3,50,1)).flatten()
corrected_posterior_nu_w0 = (jnp.array([jnp.stack(posterior_samples_nu_w0_25)[:,:,0], jnp.stack(posterior_samples_nu_w0_50)[:,:,0], jnp.stack(posterior_samples_nu_w0_100)[:,:,0]]) - jnp.array(df_posterior_mean_subj['sd_subject_true_w0']).reshape(3,50,1)).flatten()

num_subjects = jnp.repeat(jnp.array([25,50,100]),50*1000) #50 datasets * 1000 iterations
dataset = jnp.tile(jnp.repeat(jnp.arange(0,num_datasets),1000, axis=0),3)

df_example_post = pd.DataFrame({
        'cor_a0': corrected_posterior_a0,
        'cor_nu_a0': corrected_posterior_nu_a0,
        'cor_alpha': corrected_posterior_alpha,
        'cor_beta': corrected_posterior_beta,
        'cor_w0': corrected_posterior_w0,
        'cor_nu_w0': corrected_posterior_nu_w0,
        'a0': posterior_a0,
        'nu_a0': posterior_nu_a0,
        'alpha': posterior_alpha,
        'beta': posterior_beta,
        'w0': posterior_w0,
        'nu_w0': posterior_nu_w0,
        'num_subjects': num_subjects,
        'dataset': dataset})

df_example_post['num_subjects']= pd.Categorical(df_example_post['num_subjects'], categories=[25, 50, 100])




colors_subject=['#ADE19C','#81CDB6','#3b528b']

# Could use some improvement
def plot_corrected_posteriors(datasets_list, true_params, matched_true_param, matched_dataset, x_label, x_label_top, x_lim_top, x_lim, y_lim, param, filename):

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(5, 12), sharex=False, dpi=600)
    
    part1 = df_example_post[(df_example_post['num_subjects'] == 25) & (df_example_post['dataset'] == matched_dataset[0])]
    part2 = df_example_post[(df_example_post['num_subjects'] == 50) & (df_example_post['dataset'] == matched_dataset[1])]
    part3 = df_example_post[(df_example_post['num_subjects'] == 100) & (df_example_post['dataset'] == matched_dataset[2])]

    combined = pd.concat([part1, part2, part3], axis=0)
    
    sns.kdeplot(
        data=combined, 
        x=param, 
        hue='num_subjects', 
        palette=colors_subject,
        hue_order=[25, 50, 100], 
        ax=axs[0],  # Moved to the top
        bw_adjust=2, fill=True, alpha=0.7, linewidth=0
    )
    axs[0].axvline(x=matched_true_param, color='red', linestyle='--')
    axs[0].set_yticks([])
    axs[0].set_xlim(x_lim_top)
    axs[0].yaxis.set_tick_params(size=0)
    axs[0].xaxis.set_tick_params(size=5)
    axs[0].set_xlabel(x_label_top)
    
    # Loop through other datasets and corresponding axes
    # Stack datasets
    datasets = jnp.stack([jnp.stack(df) for df in datasets_list])
    
    # Subtract true parameters
    corrected_datasets = datasets - true_params
    
    for i, dataset in enumerate(corrected_datasets):
        # Plot the dataset for specific posterior
        for posterior in dataset:
            sns.kdeplot(
                posterior, bw_adjust=2, ax=axs[i + 1],  # Shift by 1 for the remaining axes
                fill=True, color=colors_subject[i], alpha=0.15, linewidth=0
            )
        axs[i + 1].set_ylabel("Density")  # Shifted to axs[i + 1]
        axs[i + 1].set_xlim(x_lim)
        axs[i + 1].set_ylim(y_lim)
        axs[i + 1].set_yticks([])
        axs[i + 1].yaxis.set_tick_params(size=0)
        axs[i + 1].xaxis.set_tick_params(size=0)
    
    axs[-1].set_xlabel(x_label)
    axs[-1].xaxis.set_tick_params(size=5)
    axs[1].set_xticklabels([]) 
    axs[2].set_xticklabels([]) 
    
    plt.tight_layout()
    # Reduce space between subplots globally
    plt.subplots_adjust(hspace=0.2)  # Uniform spacing for plots (change value as needed)
    
    # Manually adjust the position of the top plot to create more space
    pos = axs[0].get_position()  # Get current position of the top plot
    axs[0].set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height])  # Move it higher by increasing pos.y0
    

    plt.savefig(filename,transparent=True, bbox_inches='tight')
    plt.show()





df_temp = df_posterior_mean_subj

# Posteriors a0
jnp.array(df_temp[df_temp['dataset']==0]['mean_subject_true_a'])

mean_a0 = jnp.mean(jnp.array([0.97701627, 0.9767193,0.9790508]))

plot_corrected_posteriors(
    datasets_list=[posterior_samples_a0_25, posterior_samples_a0_50, posterior_samples_a0_100],
    true_params=jnp.array(df_posterior_mean_subj['mean_subject_true_a']).reshape(3,50,1),
    x_label_top=r"$\mu_a$",
    x_label=r"$\mu_a-\mu_a^*$",
    x_lim_top=(.9525, 1.005),
    x_lim=(-.035, .035),
    y_lim=(0, 190),
    matched_true_param=mean_a0,
    param='a0',
    matched_dataset=[4,4,0], #dataset for 25,50,100 subjects
    filename='corrected_posteriors_a0.png'
)

# Posteriors nu_a0
jnp.array(df_temp[df_temp['dataset']==5]['sd_subject_true_a'])

mean_nu_a0 = jnp.mean(jnp.array([0.01329772, 0.01440097,0.01437333]))

plot_corrected_posteriors(
    datasets_list=[posterior_samples_nu_a0_25, posterior_samples_nu_a0_50, posterior_samples_nu_a0_100],
    true_params=jnp.array(df_posterior_mean_subj['sd_subject_true_a']).reshape(3,50,1),
    x_label_top=r"$\sigma_a$",
    x_label=r"$\sigma_a-\sigma_a^*$",
    x_lim_top=(-.005, .05),
    x_lim=(-.06, .06),
    y_lim=(0, 160),
    matched_true_param=mean_nu_a0,
    param='nu_a0',
    matched_dataset=[4,4,5],
    filename='corrected_posteriors_nu_a0.png'
)

# Posteriors alpha
plot_corrected_posteriors(
    datasets_list=[posterior_samples_alpha_25, posterior_samples_alpha_50, posterior_samples_alpha_100],
    true_params=true_alpha,
    x_label_top=r"$\alpha$",
    x_label=r"$\alpha-\alpha^*$",
    x_lim_top=(-0.2, 12.5),
    x_lim=(-6.5,6.5),
    y_lim=(0,.9),
    matched_true_param=true_alpha,
    param='alpha',
    matched_dataset=[40,40,40],
    filename='corrected_posteriors_alpha.png'
)

# Posteriors beta
plot_corrected_posteriors(
    datasets_list=[posterior_samples_beta_25, posterior_samples_beta_50, posterior_samples_beta_100],
    true_params=true_beta,
    x_label_top=r"$\beta$",
    x_label=r"$\beta-\beta^*$",
    x_lim_top=(-0.1, 1.1),
    x_lim=(-1, 1),
    y_lim=(0,8.5),
    matched_true_param=true_beta,
    param='beta',
    matched_dataset=[40,40,40],
    filename='corrected_posteriors_beta.png'
)

# Posteriors w0

#0,6
jnp.array(df_temp[df_temp['dataset']==26]['mean_subject_true_w0'])

mean_w0 = jnp.mean(jnp.array([-0.00035373, 0.01036947, -0.00928506]))

plot_corrected_posteriors(
    datasets_list=[jnp.stack(posterior_samples_w0_25)[:,:,0], jnp.stack(posterior_samples_w0_50)[:,:,0], jnp.stack(posterior_samples_w0_100)[:,:,0]],
    true_params=jnp.array(df_posterior_mean_subj['mean_subject_true_w0']).reshape(3,50,1),
    x_label_top=r"$\mu_{w_0}$",
    x_label=r"$\mu_{w_0}-\mu_{w_0}^*$",
    x_lim_top=(-.8, .8),
    x_lim=(-.7, .7),
    y_lim=(0, 4.5),
    matched_true_param=mean_w0,
    param='w0',
    matched_dataset=[12,26,26],
    filename='corrected_posteriors_w0.png'
)

# Posteriors nu_w0
jnp.array(df_posterior_mean_subj[df_posterior_mean_subj['dataset']==27]['sd_subject_true_w0'])

mean_nu_w0 = jnp.mean(jnp.array([0.89304733, 0.8982136, 0.89172435]))

plot_corrected_posteriors(
    datasets_list=[jnp.stack(posterior_samples_nu_w0_25)[:,:,0], jnp.stack(posterior_samples_nu_w0_50)[:,:,0], jnp.stack(posterior_samples_nu_w0_100)[:,:,0]],
    true_params=jnp.array(df_posterior_mean_subj['sd_subject_true_w0']).reshape(3,50,1),
    x_label_top=r"$\sigma_{w_0}$",
    x_label=r"$\sigma_{w_0}-\sigma_{w_0}^*$",
    x_lim_top=(.45, 1.35),
    x_lim=(-.7, .7),
    y_lim=(0, 7),
    matched_true_param=mean_nu_w0,
    param='nu_w0',
    matched_dataset=[30,9,27],
    filename='corrected_posteriors_nu_w0.png'
)




"""
Plots with number of trials
"""
# Load in the different dil files
# Because variables within these dil files have the same names we have to rename them
num_subjects=50

file_500trials = file_paths_mapping.get((500, num_subjects))
file_1000trials = file_paths_mapping.get((1000, num_subjects))
file_2500trials = file_paths_mapping.get((2500, num_subjects))
file_5000trials = file_paths_mapping.get((5000, num_subjects))

load_dill_file(file_500trials[0])

posterior_samples_alpha_500 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_500 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_500 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_500 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_500 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_500 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]

load_dill_file(file_1000trials[0])

posterior_samples_alpha_1000 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_1000 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_1000 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_1000 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_1000 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_1000 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]


for file_path in file_2500trials:
    load_dill_file(file_path)

all_posterior_samples_w_0 = all_posterior_samples_w_0 + all_posterior_samples_w_02
all_posterior_samples_nu_w0 = all_posterior_samples_nu_w0 + all_posterior_samples_nu_w02
all_posterior_samples_a0 = all_posterior_samples_a0 + all_posterior_samples_a02
all_posterior_samples_nu_a0 = all_posterior_samples_nu_a0 + all_posterior_samples_nu_a02
all_posterior_samples_alpha = all_posterior_samples_alpha + all_posterior_samples_alpha2
all_posterior_samples_beta = all_posterior_samples_beta + all_posterior_samples_beta2


posterior_samples_alpha_2500 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_2500 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_2500 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_2500 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_2500 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_2500 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]


for file_path in file_5000trials:
    load_dill_file(file_path)

all_posterior_samples_w_0 = all_posterior_samples_w_0 + all_posterior_samples_w_02 + all_posterior_samples_w_03 + all_posterior_samples_w_04 + all_posterior_samples_w_05
all_posterior_samples_nu_w0 = all_posterior_samples_nu_w0 + all_posterior_samples_nu_w02 + all_posterior_samples_nu_w03 + all_posterior_samples_nu_w04 + all_posterior_samples_nu_w05
all_posterior_samples_a0 = all_posterior_samples_a0 + all_posterior_samples_a02 + all_posterior_samples_a03 + all_posterior_samples_a04 + all_posterior_samples_a05
all_posterior_samples_nu_a0 = all_posterior_samples_nu_a0 + all_posterior_samples_nu_a02 + all_posterior_samples_nu_a03 + all_posterior_samples_nu_a04 + all_posterior_samples_nu_a05
all_posterior_samples_alpha = all_posterior_samples_alpha + all_posterior_samples_alpha2 + all_posterior_samples_alpha3 + all_posterior_samples_alpha4 + all_posterior_samples_alpha5
all_posterior_samples_beta = all_posterior_samples_beta + all_posterior_samples_beta2 + all_posterior_samples_beta3 + all_posterior_samples_beta4 + all_posterior_samples_beta5

posterior_samples_alpha_5000 = jnp.stack(all_posterior_samples_alpha)[:,burn_in:]
posterior_samples_beta_5000 = jnp.stack(all_posterior_samples_beta)[:,burn_in:]
posterior_samples_a0_5000 = jnp.stack(all_posterior_samples_a0)[:,burn_in:]
posterior_samples_nu_a0_5000 = jnp.stack(all_posterior_samples_nu_a0)[:,burn_in:]
posterior_samples_w0_5000 = jnp.stack(all_posterior_samples_w_0)[:,burn_in:,:]
posterior_samples_nu_w0_5000 = jnp.stack(all_posterior_samples_nu_w0)[:,burn_in:,:]


num_datasets = 50

df_posterior_mean_plotting = pd.concat([data500_posterior_mean, data1000_posterior_mean, data2500_posterior_mean, data5000_posterior_mean], axis=0)
df_posterior_mean_plotting['dataset'] = np.arange(0,num_datasets).tolist() * 4 # 4 levels of number of trials



posterior_a0 = jnp.array([posterior_samples_a0_500,posterior_samples_a0_1000,posterior_samples_a0_2500,posterior_samples_a0_5000]).flatten()
posterior_nu_a0 = jnp.array([posterior_samples_nu_a0_500,posterior_samples_nu_a0_1000,posterior_samples_nu_a0_2500,posterior_samples_nu_a0_5000]).flatten()
posterior_alpha = jnp.array([posterior_samples_alpha_500,posterior_samples_alpha_1000,posterior_samples_alpha_2500,posterior_samples_alpha_5000]).flatten()
posterior_beta = jnp.array([posterior_samples_beta_500,posterior_samples_beta_1000,posterior_samples_beta_2500,posterior_samples_beta_5000]).flatten()
posterior_w0 = jnp.array([jnp.stack(posterior_samples_w0_500)[:,:,0], jnp.stack(posterior_samples_w0_1000)[:,:,0], jnp.stack(posterior_samples_w0_2500)[:,:,0], jnp.stack(posterior_samples_w0_5000)[:,:,0]]).flatten()
posterior_nu_w0 = jnp.array([jnp.stack(posterior_samples_nu_w0_500)[:,:,0], jnp.stack(posterior_samples_nu_w0_1000)[:,:,0], jnp.stack(posterior_samples_nu_w0_2500)[:,:,0], jnp.stack(posterior_samples_nu_w0_5000)[:,:,0]]).flatten()


corrected_posterior_a0 = (jnp.array([posterior_samples_a0_500,posterior_samples_a0_1000,posterior_samples_a0_2500,posterior_samples_a0_5000]) - jnp.array(df_posterior_mean_plotting['mean_subject_true_a']).reshape(4,50,1)).flatten()
corrected_posterior_nu_a0 = (jnp.array([posterior_samples_nu_a0_500,posterior_samples_nu_a0_1000,posterior_samples_nu_a0_2500,posterior_samples_nu_a0_5000]) - jnp.array(df_posterior_mean_plotting['sd_subject_true_a']).reshape(4,50,1)).flatten()
corrected_posterior_alpha = (jnp.array([posterior_samples_alpha_500,posterior_samples_alpha_1000,posterior_samples_alpha_2500,posterior_samples_alpha_5000]) - true_alpha).flatten()
corrected_posterior_beta = (jnp.array([posterior_samples_beta_500,posterior_samples_beta_1000,posterior_samples_beta_2500,posterior_samples_beta_5000]) - true_beta).flatten()
# we take the intercept (doesn't matter, plots look the same for all input variables)
corrected_posterior_w0 = (jnp.array([jnp.stack(posterior_samples_w0_500)[:,:,0], jnp.stack(posterior_samples_w0_1000)[:,:,0], jnp.stack(posterior_samples_w0_2500)[:,:,0], jnp.stack(posterior_samples_w0_5000)[:,:,0]]) - jnp.array(df_posterior_mean_plotting['mean_subject_true_w0']).reshape(4,50,1)).flatten()
corrected_posterior_nu_w0 = (jnp.array([jnp.stack(posterior_samples_nu_w0_500)[:,:,0], jnp.stack(posterior_samples_nu_w0_1000)[:,:,0], jnp.stack(posterior_samples_nu_w0_2500)[:,:,0], jnp.stack(posterior_samples_nu_w0_5000)[:,:,0]]) - jnp.array(df_posterior_mean_plotting['sd_subject_true_w0']).reshape(4,50,1)).flatten()

num_trials = jnp.repeat(jnp.array([500,1000,2500,5000]),50*1000) #50 datasets * 1000 iterations
dataset = jnp.tile(jnp.repeat(jnp.arange(0,num_datasets),1000, axis=0),4)

df_example_post_trials = pd.DataFrame({
        'cor_a0': corrected_posterior_a0,
        'cor_nu_a0': corrected_posterior_nu_a0,
        'cor_alpha': corrected_posterior_alpha,
        'cor_beta': corrected_posterior_beta,
        'cor_w0': corrected_posterior_w0,
        'cor_nu_w0': corrected_posterior_nu_w0,
        'a0': posterior_a0,
        'nu_a0': posterior_nu_a0,
        'alpha': posterior_alpha,
        'beta': posterior_beta,
        'w0': posterior_w0,
        'nu_w0': posterior_nu_w0,
        'num_trials': num_trials,
        'dataset': dataset})

df_example_post_trials['num_trials']= pd.Categorical(df_example_post_trials['num_trials'], categories=[500, 1000, 2500, 5000])



colors=['#ADE19C','#81CDB6','#94BBC6','#9FA1C2']


def plot_corrected_posteriors_trials(datasets_list, true_params, matched_true_param, matched_dataset, x_label, x_label_top, x_lim_top, x_lim, y_lim, param, filename):

    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(5, 12), sharex=False, dpi=600)
    
    part1 = df_example_post_trials[(df_example_post_trials['num_trials'] == 500) & (df_example_post_trials['dataset'] == matched_dataset[0])]
    part2 = df_example_post_trials[(df_example_post_trials['num_trials'] == 1000) & (df_example_post_trials['dataset'] == matched_dataset[1])]
    part3 = df_example_post_trials[(df_example_post_trials['num_trials'] == 2500) & (df_example_post_trials['dataset'] == matched_dataset[2])]
    part4 = df_example_post_trials[(df_example_post_trials['num_trials'] == 5000) & (df_example_post_trials['dataset'] == matched_dataset[3])]

    combined = pd.concat([part1, part2, part3, part4], axis=0)
    
    sns.kdeplot(
        data=combined, 
        x=param, 
        hue='num_trials', 
        palette=colors,
        hue_order=[500, 1000, 2500, 5000], 
        ax=axs[0],  # Moved to the top
        bw_adjust=2, fill=True, alpha=0.7, linewidth=0, legend=False
    )
    axs[0].axvline(x=matched_true_param, color='red', linestyle='--')
    axs[0].set_yticks([])
    axs[0].set_xlim(x_lim_top)
    axs[0].yaxis.set_tick_params(size=0)
    axs[0].xaxis.set_tick_params(size=5)
    axs[0].set_xlabel(x_label_top)
    
    
    # Loop through other datasets and corresponding axes
    datasets = jnp.stack([jnp.stack(df) for df in datasets_list])
    
    # Subtract true parameters
    corrected_datasets = datasets - true_params
    
    for i, dataset in enumerate(corrected_datasets):
        # Plot the dataset for specific posterior
        for posterior in dataset:
            sns.kdeplot(
                posterior, bw_adjust=2, ax=axs[i + 1],  # Shift by 1 for the remaining axes
                fill=True, color=colors[i], alpha=0.15, linewidth=0
            )
        axs[i + 1].set_ylabel("Density")  # Shifted to axs[i + 1]
        axs[i + 1].set_xlim(x_lim)
        axs[i + 1].set_ylim(y_lim)
        axs[i + 1].set_yticks([])
        axs[i + 1].yaxis.set_tick_params(size=0)
        axs[i + 1].xaxis.set_tick_params(size=0)
    
    axs[-1].set_xlabel(x_label)
    axs[-1].xaxis.set_tick_params(size=5)
    axs[1].set_xticklabels([]) 
    axs[2].set_xticklabels([]) 
    axs[3].set_xticklabels([]) 
    
    plt.tight_layout()
    # Reduce space between subplots globally
    plt.subplots_adjust(hspace=0.2)  # Uniform spacing for plots (change value as needed)
    
    # Manually adjust the position of the top plot to create more space
    pos = axs[0].get_position()  # Get current position of the top plot
    axs[0].set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height])  # Move it higher by increasing pos.y0
    

    plt.savefig(filename,transparent=True, bbox_inches='tight')
    plt.show()




df_temp = df_posterior_mean_plotting

# Posteriors a0
jnp.array(df_temp[df_temp['dataset']==7]['mean_subject_true_a'])

mean_a0 = jnp.mean(jnp.array([0.9787493, 0.9787493, 0.9787493, 0.9781124]))

plot_corrected_posteriors_trials(
    datasets_list=[posterior_samples_a0_500, posterior_samples_a0_1000, posterior_samples_a0_2500,posterior_samples_a0_5000],
    true_params=jnp.array(df_posterior_mean_plotting['mean_subject_true_a']).reshape(4,50,1),
    x_label_top=r"$\mu_a$",
    x_label=r"$\mu_a-\mu_a^*$",
    x_lim_top=(.9525, 1.01),
    x_lim=(-.035, .035),
    y_lim=(0, 190),
    matched_true_param=mean_a0,
    param='a0',
    matched_dataset=[7,7,7,7], #dataset for 500,1000,2500,5000 trials
    filename='corrected_posteriors_a0_trials.png'
)

# Posteriors nu_a0
jnp.array(df_temp[df_temp['dataset']==14]['sd_subject_true_a'])

mean_nu_a0 = jnp.mean(jnp.array([0.01770392, 0.01770392, 0.01770392, 0.01747185]))

plot_corrected_posteriors_trials(
    datasets_list=[posterior_samples_nu_a0_500, posterior_samples_nu_a0_1000, posterior_samples_nu_a0_2500,posterior_samples_nu_a0_5000],
    true_params=jnp.array(df_posterior_mean_plotting['sd_subject_true_a']).reshape(4,50,1),
    x_label_top=r"$\sigma_a$",
    x_label=r"$\sigma_a-\sigma_a^*$",
    x_lim_top=(-.008, .06),
    x_lim=(-.04, .04),
    y_lim=(0, 150),
    matched_true_param=mean_nu_a0,
    param='nu_a0',
    matched_dataset=[14,14,14,14],
    filename='corrected_posteriors_nu_a0_trials.png'
)

# Posteriors alpha
plot_corrected_posteriors_trials(
    datasets_list=[posterior_samples_alpha_500, posterior_samples_alpha_1000, posterior_samples_alpha_2500,posterior_samples_alpha_5000],
    true_params=true_alpha,
    x_label_top=r"$\alpha$",
    x_label=r"$\alpha-\alpha^*$",
    x_lim_top=(-.01, 12),
    x_lim=(-6, 6),
    y_lim=(0, .7),
    matched_true_param=true_alpha,
    param='alpha',
    matched_dataset=[1,1,1,1],
    filename='corrected_posteriors_alpha_trials.png'
)

# Posteriors beta
plot_corrected_posteriors_trials(
    datasets_list=[posterior_samples_beta_500, posterior_samples_beta_1000, posterior_samples_beta_2500,posterior_samples_beta_5000],
    true_params=true_beta,
    x_label_top=r"$\beta$",
    x_label=r"$\beta-\beta^*$",
    x_lim_top=(-.04, 1.4),
    x_lim=(-.9, .9),
    y_lim=(0, 7),
    matched_true_param=true_beta,
    param='beta',
    matched_dataset=[1,1,1,1],
    filename='corrected_posteriors_beta_trials.png'
)


# Posteriors w0
jnp.array(df_temp[df_temp['dataset']==1]['mean_subject_true_w0'])

mean_w0 = jnp.mean(jnp.array([0.01770392, 0.01770392, 0.01770392, 0.01747185]))


plot_corrected_posteriors_trials(
    datasets_list=[jnp.stack(posterior_samples_w0_500)[:,:,0], jnp.stack(posterior_samples_w0_1000)[:,:,0], jnp.stack(posterior_samples_w0_2500)[:,:,0], jnp.stack(posterior_samples_w0_5000)[:,:,0]],
    true_params=jnp.array(df_posterior_mean_plotting['mean_subject_true_w0']).reshape(4,50,1),
    x_label_top=r"$\mu_{w_0}$",
    x_label=r"$\mu_{w_0}-\mu_{w_0}^*$",
    x_lim_top=(-.7, .7),
    x_lim=(-.6, .6),
    y_lim=(0, 3.7),
    matched_true_param=-0.11574253,
    param='w0',
    matched_dataset=[1,1,1,1],
    filename='corrected_posteriors_w0_trials.png'
)


# Posteriors nu_w0
jnp.array(df_temp[df_temp['dataset']==1]['sd_subject_true_w0'])

sd_w0 = jnp.mean(jnp.array([0.01770392, 0.01770392, 0.01770392, 0.01747185]))


plot_corrected_posteriors_trials(
    datasets_list=[jnp.stack(posterior_samples_nu_w0_500)[:,:,0], jnp.stack(posterior_samples_nu_w0_1000)[:,:,0], jnp.stack(posterior_samples_nu_w0_2500)[:,:,0], jnp.stack(posterior_samples_nu_w0_5000)[:,:,0]],
    true_params=jnp.array(df_posterior_mean_plotting['sd_subject_true_w0']).reshape(4,50,1),
    x_label_top=r"$\sigma_{w_0}$",
    x_label=r"$\sigma_{w_0}-\sigma_{w_0}^*$",
    x_lim_top=(.5, 1.5),
    x_lim=(-.6, .6),
    y_lim=(0, 4.5),
    matched_true_param=1.0205454,
    param='nu_w0',
    matched_dataset=[1,1,1,1],
    filename='corrected_posteriors_nu_w0_trials.png'
)






"""
Additional plots for posteriors (not in the paper)
"""


# if all_posterior_means_subj should be plotted, change data=all_posterior_means and x="num_trials" to data=all_posterior_means_subj and x="num_subjects" 
def plot_posterior_means(y, true_value, xlabel, ylabel, hue="num_trials", alpha=.3, data=all_posterior_means, x="num_trials"):
    sns.stripplot(data=data, x=x, hue=hue, y=y, alpha=alpha)
    plt.axhline(y=true_value, color='red', linestyle='--')
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend('', frameon=False)
    plt.show()



# Posterior means alpha (inverse gamma for sigma_sq)
plot_posterior_means(y="alpha", true_value=true_alpha, xlabel='Number of trials', ylabel=r"Posterior mean $\alpha$")


# Posterior means beta (inverse gamma for sigma_sq_i)
plot_posterior_means(y="beta", true_value=true_beta, xlabel='Number of trials', ylabel=r"Posterior mean $\beta$")


# Posterior means a0 (truncated normal for a_i)
plot_posterior_means(y="a0", true_value=true_a, xlabel='Number of trials', ylabel=r"Posterior mean $a_0$")
## difference between estimated a0 per dataset and the average per dataset of true a_i over subjects
all_posterior_means['difference_a0_subject_true_a'] = all_posterior_means['a0'] - all_posterior_means['mean_subject_true_a']
plot_posterior_means(y="difference_a0_subject_true_a", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $a_0$ - mean true $a_i$")


# Posterior means nu_a0 (truncated normal for a_i)
plot_posterior_means(y="nu_a0", true_value=true_nu_a, xlabel='Number of trials', ylabel=r"Posterior mean $\nu_{a_0}$")
## difference between estimated a0 per dataset and the standard deviation per dataset of true a_i over subjects
all_posterior_means['difference_nu_a0_subject_true_a'] = all_posterior_means['nu_a0'] - all_posterior_means['sd_subject_true_a']
plot_posterior_means(y="difference_nu_a0_subject_true_a", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $\nu_{a_0}$ - sd true $a_i$")


# Posterior means w0 (normal for w_0_i)
plot_posterior_means(y="w0", true_value=true_w0[0], xlabel='Number of trials', ylabel=r"Posterior mean $w_0$")
## difference between estimated w0 per dataset and the mean per dataset of true w_0_i over subjects
all_posterior_means['difference_w0_subject_true_w0'] = all_posterior_means['w0'] - all_posterior_means['mean_subject_true_w0']
plot_posterior_means(y="difference_w0_subject_true_w0", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $w_0$ - mean true $w_{i_0}$")


# Posterior means w1 (normal for w_1_i)
plot_posterior_means(y="w1", true_value=true_w0[1], xlabel='Number of trials', ylabel=r"Posterior mean $w_1$")
## difference between estimated w1 per dataset and the mean per dataset of true w_1_i over subjects
all_posterior_means['difference_w1_subject_true_w1'] = all_posterior_means['w1'] - all_posterior_means['mean_subject_true_w1']
plot_posterior_means(y="difference_w1_subject_true_w1", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $w_1$ - mean true $w_{i_1}$")


# Posterior means w2 (normal for w_2_i)
plot_posterior_means(y="w2", true_value=true_w0[2], xlabel='Number of trials', ylabel=r"Posterior mean $w_2$")
## difference between estimated w2 per dataset and the mean per dataset of true w_2_i over subjects
all_posterior_means['difference_w2_subject_true_w2'] = all_posterior_means['w2'] - all_posterior_means['mean_subject_true_w2']
plot_posterior_means(y="difference_w2_subject_true_w2", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $w_2$ - mean true $w_{i_2}$")


# Posterior means w3 (normal for w_3_i)
plot_posterior_means(y="w3", true_value=true_w0[3], xlabel='Number of trials', ylabel=r"Posterior mean $w_3$")
## difference between estimated w3 per dataset and the mean per dataset of true w_3_i over subjects
all_posterior_means['difference_w3_subject_true_w3'] = all_posterior_means['w3'] - all_posterior_means['mean_subject_true_w3']
plot_posterior_means(y="difference_w3_subject_true_w3", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $w_3$ - mean true $w_{i_3}$")


# Posterior means nu_w0 (normal for w_0_i)
plot_posterior_means(y="nu_w0", true_value=true_nu_w[0], xlabel='Number of trials', ylabel=r"Posterior mean $\nu_{w_0}$")
## difference between estimated nu_w0 per dataset and the sd per dataset of true w_0_i over subjects
all_posterior_means['difference_nu_w0_subject_true_w0'] = all_posterior_means['nu_w0'] - all_posterior_means['sd_subject_true_w0']
plot_posterior_means(y="difference_nu_w0_subject_true_w0", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $\nu_{w_0}$ - sd true $w_{i_0}$")


# Posterior means nu_w1 (normal for w_1_i)
plot_posterior_means(y="nu_w1", true_value=true_nu_w[1], xlabel='Number of trials', ylabel=r"Posterior mean $\nu_{w_1}$")
## difference between estimated nu_w1 per dataset and the sd per dataset of true w_1_i over subjects
all_posterior_means['difference_nu_w1_subject_true_w1'] = all_posterior_means['nu_w1'] - all_posterior_means['sd_subject_true_w1']
plot_posterior_means(y="difference_nu_w1_subject_true_w1", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $\nu_{w_1}$ - sd true $w_{i_1}$")


# Posterior means nu_w2 (normal for w_2_i)
plot_posterior_means(y="nu_w2", true_value=true_nu_w[2], xlabel='Number of trials', ylabel=r"Posterior mean $\nu_{w_2}$")
## difference between estimated nu_w2 per dataset and the sd per dataset of true w_2_i over subjects
all_posterior_means['difference_nu_w2_subject_true_w2'] = all_posterior_means['nu_w2'] - all_posterior_means['sd_subject_true_w2']
plot_posterior_means(y="difference_nu_w2_subject_true_w2", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $\nu_{w_2}$ - sd true $w_{i_2}$")


# Posterior means nu_w3 (normal for w_3_i)
plot_posterior_means(y="nu_w3", true_value=true_nu_w[3], xlabel='Number of trials', ylabel=r"Posterior mean $\nu_{w_3}$")
## difference between estimated nu_w3 per dataset and the sd per dataset of true w_3_i over subjects
all_posterior_means['difference_nu_w3_subject_true_w3'] = all_posterior_means['nu_w3'] - all_posterior_means['sd_subject_true_w3']
plot_posterior_means(y="difference_nu_w3_subject_true_w3", true_value=0, xlabel='Number of trials', ylabel=r"Estimated $\nu_{w_3}$ - sd true $w_{i_3}$")





"""
Plot posterior means and 95% CI minus true value (to assess bias) (not in the paper)
"""

# merge all the combination of num_trials and num_subjects
# note: all_posterior_means_subj_no50subj is without data500_50subj_posterior_mean because already in all_posterior_means
all_posterior_means = pd.concat([data500_posterior_mean, data1000_posterior_mean, data2500_posterior_mean, data5000_posterior_mean], axis=0)
all_posterior_means_subj_no50subj = pd.concat([data500_25subj_posterior_mean, data500_100subj_posterior_mean, data500_200subj_posterior_mean], axis=0)


df_posterior_means = pd.concat([all_posterior_means, all_posterior_means_subj_no50subj], axis=0)
df_posterior_means['dataset'] = np.arange(0,num_datasets).tolist() * 7 # 7 different combinations of num_trials and num_subjects


num_subjects = 50
num_trials = 2500

df = df_posterior_means[(df_posterior_means['num_subjects']==num_subjects) & (df_posterior_means['num_trials']==num_trials)]



# Set sort to False for unsorted plots
def plot_corrected_post_means(variable, true_value, lower_bound, upper_bound, y_label, ylim, file_name, sort=True):
    df[f'{variable}_diff'] = df[variable] - true_value
    
    if sort:
        df_sorted = df.sort_values(by=f'{variable}_diff') # Use the sorted indices for the x-axis
        x_values = range(len(df_sorted))
        y_values = df_sorted[f'{variable}_diff']
        yerr = [df_sorted[variable] - df_sorted[lower_bound], df_sorted[upper_bound] - df_sorted[variable]]
    else:
        x_values = df['dataset']
        y_values = df[f'{variable}_diff']
        yerr = [df[variable] - df[lower_bound], df[upper_bound] - df[variable]]
    
    plt.figure(figsize=(8, 6), dpi=600)
    plt.axhline(0, color='red', linestyle='--')
    plt.errorbar(x_values, y_values, yerr=yerr, fmt='o', ecolor='black', linestyle='None', capsize=3)
    plt.xticks(range(len(df_sorted)), df_sorted['dataset'], rotation=90)  
    plt.xticks([], []) # comment out to see the ordered dataset numbers if needed
    plt.xlabel('Dataset')
    plt.ylabel(y_label)
    plt.ylim(ylim)
    plt.tight_layout()

    plt.savefig(file_name)


"""
Alpha: correct with true_alpha
"""

# In the model we have an upperbound of 10 for alpha. 10-5 (true value) = 5 which explains the upperbound in the plot for all datasets
plot_corrected_post_means(
    variable='alpha', 
    true_value=true_alpha, 
    lower_bound='posterior_q25_alpha', 
    upper_bound='posterior_q975_alpha', 
    y_label=r'$\hat{\alpha}$ - $\alpha$', 
    ylim=(-3.5, 5), #(-5, 5) 
    file_name=f"alpha_{num_subjects}subjects_{num_trials}trials.png"
)


"""
Beta: correct with true_beta
"""


plot_corrected_post_means(
    variable='beta', 
    true_value=true_beta, 
    lower_bound='posterior_q25_beta', 
    upper_bound='posterior_q975_beta', 
    y_label=r'$\hat{\beta}$ - $\beta$', 
    ylim=(-.4, 1.0), #(-.5, 1.05) 
    file_name=f"beta_{num_subjects}subjects_{num_trials}trials.png"
)



"""
a0: corrected with mean of all subject's true a
"""

plot_corrected_post_means(
    variable='a0', 
    true_value=df['mean_subject_true_a'], 
    lower_bound='posterior_q25_a0', 
    upper_bound='posterior_q975_a0', 
    y_label=r'$\hat{a_0}$ - $a_0$', 
    ylim=(-0.021, 0.029), #(-0.025, 0.031) 
    file_name=f"a0_{num_subjects}subjects_{num_trials}trials.png"
)


"""
nu_a0: corrected with standard deviation of all subject's true a
"""

plot_corrected_post_means(
    variable='nu_a0', 
    true_value=df['sd_subject_true_a'], 
    lower_bound='posterior_q25_nu_a0', 
    upper_bound='posterior_q975_nu_a0', 
    y_label=r'$\hat{\nu_{a_0}}$ - $\nu_{a_0}$', 
    ylim=(-0.019, 0.05), #(-0.021, 0.07) 
    file_name=f"nu_a0_{num_subjects}subjects_{num_trials}trials.png"
)


"""
w0
"""

plot_corrected_post_means(
    variable='w0', 
    true_value=df['mean_subject_true_w0'], 
    lower_bound='posterior_q25_w0', 
    upper_bound='posterior_q975_w0', 
    y_label=r'$\hat{w_0}$ - $w_0$', 
    ylim=(-0.42, 0.4), #(-0.68, 0.65) 
    file_name=f"w0_{num_subjects}subjects_{num_trials}trials.png"
)

"""
w1
"""

plot_corrected_post_means(
    variable='w1', 
    true_value=df['mean_subject_true_w1'], 
    lower_bound='posterior_q25_w1', 
    upper_bound='posterior_q975_w1', 
    y_label=r'$\hat{w_1}$ - $w_1$', 
    ylim=(-0.39, 0.41), #(-0.6, 0.65) 
    file_name=f"w1_{num_subjects}subjects_{num_trials}trials.png"
)


"""
nu_w0
"""

plot_corrected_post_means(
    variable='nu_w0', 
    true_value=df['sd_subject_true_w0'], 
    lower_bound='posterior_q25_nu_w0', 
    upper_bound='posterior_q975_nu_w0', 
    y_label=r'$\hat{\nu_{w_0}}$ - $\nu_{w_0}$', 
    ylim=(-0.25, 0.45), #(-0.35, 0.63)
    file_name=f"nu_w0_{num_subjects}subjects_{num_trials}trials.png"
)


"""
nu_w1
"""

plot_corrected_post_means(
    variable='nu_w1', 
    true_value=df['sd_subject_true_w1'], 
    lower_bound='posterior_q25_nu_w1', 
    upper_bound='posterior_q975_nu_w1', 
    y_label=r'$\hat{\nu_{w_1}}$ - $\nu_{w_1}$', 
    ylim=(-0.28, 0.38), #(-0.35, 0.6) 
    file_name=f"nu_w1_{num_subjects}subjects_{num_trials}trials.png"
)

