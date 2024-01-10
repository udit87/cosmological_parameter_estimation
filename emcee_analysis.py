import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from likelihood import log_likelihood, log_prior, log_posterior

def run_mcmc_analysis(initial_params, n_walkers=32, n_steps=1000, discard=500, thin=15):
    ndim = len(initial_params[0])  # Number of parameters
    
    # Create the emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)

    # Running the sampler
    sampler.run_mcmc(initial_params, n_steps, progress=True)

    # Plotting chains
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["omega_M", "omega_lambda", "H0"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

    # Getting chain and discarding first 'discard' entries
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    print(flat_samples.shape)

    # Corner plot
    fig = corner.corner(flat_samples, show_titles=True, labels=labels)
    plt.show()

