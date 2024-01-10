# main.py
import numpy as np
from data_processing import process_data_and_covariance
from likelihood import log_likelihood_lcdm, log_prior_lcdm, log_posterior_lcdm,log_likelihood_tsallis, log_prior_tsallis, log_posterior_tsallis
import emcee
import matplotlib.pyplot as plt
import corner

# Define file paths
data_file_path = "Pantheon+SH0ES.txt"
cov_file_path = "Pantheon+SH0ES_STAT+SYS.txt"

# Process data and covariance
zcmb, zhel, mu, cov = process_data_and_covariance(data_file_path, cov_file_path)

# Define cosmological parameters
cosmology_choice = input("Enter the desired cosmology (lcdm or tsallis): ").lower()

# Define cosmological parameters
if cosmology_choice == "lcdm":
    # Define number of walkers and steps
    n_walkers = 32
    n_steps = 1000

# Initialize the walkers with a small random perturbation
    ndim = 2 # Number of parameters
    initial_params = [ 0.3, 73] + 1e-4 * np.random.randn(n_walkers, ndim)

# Create the emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_lcdm)

# Runing the sampler
    sampler.run_mcmc(initial_params, n_steps, progress=True)

# Plotting chains
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["omega_M","H0"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

# Getting chain and discarding first 100 entries
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)

# Corner plot
    fig = corner.corner(flat_samples,show_titles=True,labels=labels,levels = (0.68, 0.95))
    plt.show()
    
elif cosmology_choice == "tsallis":
    # Define number of walkers and steps
    n_walkers = 32
    n_steps = 1000

# Initialize the walkers with a small random perturbation
    ndim = 3 # Number of parameters
    initial_params = [ 73,  0.3, 1] + 1e-2 * np.random.randn(n_walkers, ndim)

# Create the emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_tsallis)

# Runing the sampler
    sampler.run_mcmc(initial_params, n_steps, progress=True)

# Plotting chains
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["H0", "omega_M","beta"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

# Getting chain and discarding first 100 entries
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)

# Corner plot
    fig = corner.corner(flat_samples,show_titles=True,labels=labels,levels = (0.68, 0.95))
    plt.show()
