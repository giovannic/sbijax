"""
SEIR (Susceptible-Exposed-Infectious-Recovered) model implementation
using Flow Matching for Posterior Estimation.

This script implements a modern epidemiological model with:
- Functional random variables (obs) with temporal indexing
- PyTree bijectors for continuous data transformation  
- Both structured (SFMPE) and flat (FMPE) inference approaches
- Hydra configuration management
- LC2ST evaluation framework

Updated to use latest package interfaces following hierarchical_brownian.py pattern.
This version performs estimation only and saves results to .npy files.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict
from jaxtyping import PyTree, Array
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from jax import numpy as jnp, random as jr, tree, vmap, jit
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

import arviz as az
from seir_utils import (
    seir_dynamics, prior_fn, create_simulator_dist, create_simulator_fn,
    apply_dequantization, f_in_fn, f_in_fn_observed, flatten_theta_dict,
    create_flat_blockwise_bijector, reconstruct_theta_dict, _flatten,
    flatten_f_in
)


def run(cfg: DictConfig) -> None:
    """Main execution function."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running SEIR with n_simulations={cfg.n_simulations}, "
                f"n_rounds={cfg.n_rounds}, n_epochs={cfg.n_epochs}")
    
    # Extract parameters
    n_timesteps = cfg.n_timesteps
    n_obs = cfg.n_obs
    n_sites = cfg.n_sites
    n_warmup = cfg.n_warmup
    n_simulations = cfg.n_simulations
    n_rounds = cfg.n_rounds
    n_epochs = cfg.n_epochs
    n_post_samples = cfg.n_post_samples


    key = jr.PRNGKey(cfg.seed)
    
    # Create functions
    simulator_dist = create_simulator_dist(n_timesteps, cfg.dt, cfg.population, cfg.I0_prop, n_warmup)
    simulator_fn = create_simulator_fn(simulator_dist)
    
    # Generate ground truth and observations
    theta_key, obs_key, f_in_key, key = jr.split(key, 4)
    
    theta_truth = prior_fn(n_sites).sample((1,), seed=theta_key)
    f_in = f_in_fn(n_obs, n_sites, n_timesteps).sample((1,), seed=f_in_key)
    y_observed = simulator_fn(obs_key, theta_truth, f_in)

    # Generate representative data for consistent Z-scaling across all bijectors
    repr_key, key = jr.split(key)
    repr_theta = prior_fn(n_sites).sample((1000,), seed=repr_key)
    
    # Define bijector specifications for constrained -> unconstrained transformation
    bijector_specs = {
        'beta_0': tfb.Invert(tfb.Sigmoid(low=0.1, high=2.0)),
        'alpha': tfb.Invert(tfb.Sigmoid(low=1/30, high=1/7)),
        'sigma': tfb.Invert(tfb.Sigmoid(low=1/21, high=1/7)),
        'A': tfb.Invert(tfb.Sigmoid(low=0.0, high=1.0)),
        'T_season': tfb.Invert(tfb.Softplus()),
        'phi': tfb.Invert(tfb.Sigmoid(low=0.0, high=2*jnp.pi)),
    }
    
    
    # Create MCMC bijector
    flat_theta_bijector = create_flat_blockwise_bijector(repr_theta, bijector_specs, n_sites)

    # Create proxy functions for MCMC sampling
    def flat_prior_fn(key: Array, n_samples: int) -> Array:
        """Prior function compatible with FMPE interface"""
        theta_samples = prior_fn(n_sites).sample((n_samples,), seed=key)
        # Flatten and transform to unconstrained space
        return flatten_theta_dict(theta_samples)

    prior = prior_fn(n_sites)

    @jit
    def flat_simulator_log_prob(theta_flat: Array) -> Array:
        """Simulator function compatible with FMPE interface"""
        # Reconstruct structured theta from flat representation
        theta_dict = reconstruct_theta_dict(theta_flat, n_sites)
        
        # Run simulator
        n_simulations = theta_flat.shape[0]
        f_in_matched = tree.map(
            lambda leaf: jnp.repeat(leaf, n_simulations, axis=0),
            f_in
        )
        sim_dist = simulator_dist(theta_dict, f_in_matched)
        prior_p = prior.log_prob(theta_dict)
        return jnp.sum(prior_p, axis=1) + jnp.sum(sim_dist.log_prob(y_observed), axis=(1, 2))

    # Train using round-based approach
    logger.info("Starting MCMC sampling")
    start_time = time.time()

    # Sample from MCMC
    n_burnin = cfg.n_simulations - cfg.n_post_samples
    sample_key, init_key, key = jr.split(key, 3)

    init_state = flat_prior_fn(init_key, cfg.mcmc.n_chains)
    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=flat_simulator_log_prob,
            step_size=cfg.mcmc.step_size,
            max_doublings=cfg.mcmc.max_doublings
        ),
        bijector=flat_theta_bijector
    )

    mcmc_posterior_samples = tfp.mcmc.sample_chain(
        num_results=n_post_samples,
        num_burnin_steps=n_burnin,
        current_state=init_state,
        kernel=kernel,
        seed=sample_key
    )

    logger.info(f'MCMC posterior mean: {jnp.mean(mcmc_posterior_samples.all_states, axis=(0, 1))}')
    logger.info(f"MCMC posterior sampling completed in {time.time() - start_time:.2f} seconds")

    logger.info(f'Analysing MCMC')
    start_time = time.time()
    logger.info(f"Converting MCMC posterior to az format")
    post_dict = reconstruct_theta_dict(
        jnp.swapaxes(mcmc_posterior_samples.all_states, 0, 1),
        n_sites
    )
    posterior = az.from_dict(posterior=post_dict)
    logger.info(f"Summarising MCMC posterior")
    print(az.summary(posterior))
    logger.info(f"MCMC summarisation completed in {time.time() - start_time:.2f} seconds")

    # Use Hydra's output directory
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.runtime.output_dir)
    
    # Save estimation results as .npy files for later visualization
    logger.info("Saving results to .npy files")
    jnp.save(out_dir / "theta_truth.npy", theta_truth)
    jnp.save(out_dir / "y_observed.npy", y_observed)
    jnp.save(out_dir / "f_in.npy", f_in)
    jnp.save(out_dir / "mcmc_posterior_samples.npy", mcmc_posterior_samples.all_states)
    
    # Save configuration parameters needed for plotting
    plot_config = {
        'n_sites': n_sites,
        'n_timesteps': n_timesteps,
        'n_warmup': n_warmup,
        'population': cfg.population,
        'I0_prop': cfg.I0_prop,
        'dt': cfg.dt
    }
    
    # Save as JSON for easy loading
    with open(out_dir / "plot_config.json", 'w') as f:
        json.dump(plot_config, f, indent=2)
    
    logger.info("SEIR MCMC estimation completed successfully!")



@hydra.main(version_base=None, config_path="conf", config_name="seir_mcmc_config")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration management."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting SEIR experiment")
    
    # Run the experiment
    run(cfg)

if __name__ == "__main__":
    main()
