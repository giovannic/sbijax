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

from jax import numpy as jnp, random as jr, tree, vmap
from jax.experimental.ode import odeint
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

from tensorflow_probability.substrates.jax import bijectors as tfb
import matplotlib.pyplot as plt
import arviz as az

def seir_dynamics(
    state: Array, 
    t: float, 
    params: Dict[str, float]
) -> Array:
    """
    SEIR differential equation system for single age group.
    
    Args:
        state: [S, E, I, R] compartment sizes
        t: Current time
        params: Model parameters
        
    Returns:
        Derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
    """
    S, E, I, R = state
    N = S + E + I + R
    mu = 1. / 50.
    
    # Seasonal transmission rate
    beta = params['beta_0'] * (
        1 + params['A'] * jnp.sin(
            2 * jnp.pi * t / params['T_season'] - params['phi']
        )
    )
    
    # Force of infection
    lambda_force = beta * I / N
    
    # Transitions
    s_to_e = lambda_force * S
    e_to_i = params['alpha'] * E
    i_to_r = params['sigma'] * I

    # Mortality
    mu_i = mu * I
    mu_e = mu * E
    mu_r = mu * R
    rebirth = mu * (E + I + R)
    
    # Derivatives
    dS = -s_to_e + rebirth
    dE = s_to_e - e_to_i - mu_e
    dI = e_to_i - i_to_r - mu_i
    dR = i_to_r - mu_r
    
    return jnp.array([dS, dE, dI, dR])

def prior_fn(n):
    """Global prior distribution."""
    t_season_spread = 1./50.
    return tfd.JointDistributionNamed(
        dict(
            # Global parameters (independent of obs by exchangeability)
            beta_0 = tfd.Uniform(jnp.full((1, 1), 0.1), jnp.full((1, 1), 2.0)),
            alpha = tfd.Uniform(jnp.full((1, 1), 1/30), jnp.full((1, 1), 1/7)),
            sigma = tfd.Uniform(jnp.full((1, 1), 1/21), jnp.full((1, 1), 1/7)),
            
            # Local parameters are independent of global parameters
            A = tfd.Uniform(jnp.zeros((n, 1)), jnp.ones((n, 1))),
            T_season = tfd.Gamma(
                jnp.full((n, 1), 365.0 * t_season_spread), 
                jnp.full((n, 1), t_season_spread)
            ),
            phi = tfd.Uniform(
                jnp.zeros((n, 1)), 
                jnp.full((n, 1), 2*jnp.pi)
            )
        ),
        batch_ndims=1,
    )

def create_simulator_fn(simulator_dist: Callable) -> Callable:
    def simulator_fn(key: Array, theta: Dict[str, Array], f_in: dict) -> Dict[str, Array]:
        return simulator_dist(theta, f_in).sample(seed=key)
    return simulator_fn

def create_simulator_dist(
    n_timesteps: int,
    dt: float = 1.0,
    population: int = 10000,
    I0_prop: float = 0.001,
    n_warmup: int = 0
) -> Callable:
    """Create simulator function for SEIR dynamics."""
    
    def simulator_dist(theta: Dict[str, Array], f_in: dict) -> tfd.Distribution:
        """
        Simulate SEIR dynamics and return indexed observations.
        
        Args:
            key: Random key
            theta: Parameters
            f_in: Functional input data containing observation indices
            
        Returns:
            Dictionary with 'obs' key containing observations at specified indices
        """
        batch_size = theta['beta_0'].shape[0]
        obs_times = f_in['obs']  # Extract observation times from f_in
        # obs_times shape: (batch_size, n_sites, n_obs, 1)
        n_sites = obs_times.shape[1] 
        
        # Initial conditions for SEIR
        I0 = population * I0_prop
        S0 = population - I0
        initial_state = jnp.array([S0, 0.0, I0, 0.0])  # [S, E, I, R]
        
        def solve_single_site(site_idx: int, params_single: Dict[str, Array], obs_times_single: Array) -> Array:
            """Solve ODE for single site and parameter set."""
            params_dict = {
                'beta_0': params_single['beta_0'][0, 0],
                'alpha': params_single['alpha'][0, 0],
                'sigma': params_single['sigma'][0, 0],
                'A': params_single['A'][site_idx, 0],
                'T_season': params_single['T_season'][site_idx, 0],
                'phi': params_single['phi'][site_idx, 0]
            }

            def ode_func(state, t):
                return seir_dynamics(state, t, params_dict)
            
            # Solve ODE at observation times for this site
            # Add warmup offset to observation times
            t_eval = jnp.concatenate([jnp.array([0.]), obs_times_single[:, 0] + n_warmup])  # Extract times (n_obs,)
            sort_indices = jnp.argsort(t_eval)
            t_eval_sorted = t_eval[sort_indices]
            solution = odeint(ode_func, initial_state, t_eval_sorted)
            
            # Reorder solution to match original time sequence
            reorder_indices = jnp.argsort(sort_indices)
            solution_reordered = solution[reorder_indices]
            
            # Extract incidence - since we're solving at the observation times directly,
            # we return the infection rate (new infections per day) at those times
            # For incidence, we use the infection rate α*E at observation times  
            exposed = solution_reordered[1:, 1]  # E compartment (index 1 in SEIR)

            incidence = params_dict['alpha'] * exposed
            incidence = jnp.maximum(incidence, 1e-8)  # Ensure positive with small delta
            
            return incidence
        
        # Vectorize over batch and sites  
        solve_batch_sites = vmap(
            vmap(solve_single_site, in_axes=(0, None, 0)),  # site_idx, params, obs_times_per_site
            in_axes=(None, 0, 0)  # site_indices, theta_batch, obs_times_batch
        )
        
        # Generate incidence for all sites
        site_indices = jnp.arange(n_sites)
        incidence_batch = solve_batch_sites(site_indices, theta, obs_times)
        return tfd.JointDistributionNamed(
            dict(
                obs = tfd.Independent(
                    tfd.Poisson(jnp.maximum(incidence_batch, 0.1)[..., None]), 
                    reinterpreted_batch_ndims=1
                )
            )
        )

    return simulator_dist

def apply_dequantization(
    obs_data: Dict[str, Array], 
    key: Array
) -> Dict[str, Array]:
    """
    Apply uniform dequantization to discrete observations while preserving positivity.
    
    Args:
        obs_data: Observation data (discrete, non-negative)
        key: Random key
        
    Returns:
        Dequantized observation data (continuous, positive)
    """
    # Dequantize with uniform [0, 1) noise to preserve positivity
    obs_dequant = {}
    for name, data in obs_data.items():
        key, subkey = jr.split(key)
        noise = jr.uniform(subkey, data.shape, minval=0.0, maxval=1.0)
        obs_dequant[name] = data.astype(float) + noise
    
    return obs_dequant

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

    def f_in_fn(n_obs, n_sites):
        """Function input sampler for observation indices."""
        return tfd.JointDistributionNamed(
            dict(
                # Global parameters - dummy entries for structure
                beta_0 = tfd.Deterministic(jnp.zeros((1, 1))),
                alpha = tfd.Deterministic(jnp.zeros((1, 1))), 
                sigma = tfd.Deterministic(jnp.zeros((1, 1))),
                
                # Local parameters - dummy entries for structure  
                A = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                T_season = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                phi = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                
                # Functional observation times
                obs = tfd.Uniform(
                    jnp.zeros((n_sites, n_obs, 1), dtype=float),
                    jnp.full((n_sites, n_obs, 1), float(n_timesteps))
                )
            ),
            batch_ndims=1
        )

    def f_in_fn_observed(n_obs, n_sites, f_in):
        """Function input sampler for observation indices."""
        if n_sites == 1:
            return tfd.JointDistributionNamed(
                dict(
                    # Global parameters - dummy entries for structure
                    beta_0 = tfd.Deterministic(jnp.zeros((1, 1))),
                    alpha = tfd.Deterministic(jnp.zeros((1, 1))), 
                    sigma = tfd.Deterministic(jnp.zeros((1, 1))),
                    
                    # Local parameters - dummy entries for structure  
                    A = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                    T_season = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                    phi = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                    
                    # Functional observation times
                    obs_index = tfd.FiniteDiscrete(
                        jnp.arange(n_sites),
                        logits=jnp.ones((n_sites,))
                    ),
                    obs = lambda obs_index: tfd.Deterministic(
                        jnp.expand_dims(f_in['obs'][0, obs_index, ...], 1)
                    )
                ),
                batch_ndims=1
            )
        elif n_sites == f_in['obs'].shape[1]:
            return tfd.JointDistributionNamed(
                dict(
                    # Global parameters - dummy entries for structure
                    beta_0 = tfd.Deterministic(jnp.zeros((1, 1))),
                    alpha = tfd.Deterministic(jnp.zeros((1, 1))), 
                    sigma = tfd.Deterministic(jnp.zeros((1, 1))),
                    
                    # Local parameters - dummy entries for structure  
                    A = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                    T_season = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                    phi = tfd.Deterministic(jnp.zeros((n_sites, 1))),
                    obs = tfd.Deterministic(f_in['obs'][0])
                ),
                batch_ndims=1
            )

    key = jr.PRNGKey(cfg.seed)
    
    # Create functions
    simulator_dist = create_simulator_dist(n_timesteps, cfg.dt, cfg.population, cfg.I0_prop, n_warmup)
    simulator_fn = create_simulator_fn(simulator_dist)
    
    # Generate ground truth and observations
    theta_key, obs_key, f_in_key, key = jr.split(key, 4)
    
    theta_truth = prior_fn(n_sites).sample((1,), seed=theta_key)
    f_in = f_in_fn(n_obs, n_sites).sample((1,), seed=f_in_key)
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
    
    # Helper functions for FMPE bijector integration
    def flatten_theta_dict(theta_dict: Dict[str, Array]) -> Array:
        """Flatten theta dictionary to 1D array for FMPE."""
        flattened_parts = []
        for param_name in ['beta_0', 'alpha', 'sigma']:
            flattened_parts.append(theta_dict[param_name].reshape(theta_dict[param_name].shape[0], 1))
        for param_name in ['A', 'T_season', 'phi']:
            flattened_parts.append(theta_dict[param_name].reshape(theta_dict[param_name].shape[0], n_sites))
        return jnp.concatenate(flattened_parts, axis=1)
    
   
    def create_flat_blockwise_bijector(repr_theta: Dict[str, Array], bijector_specs: Dict[str, tfb.Bijector], n_sites: int) -> tfb.Bijector:
        """Create blockwise bijector for FMPE using same Z-scaling as SFMPE."""
        individual_bijectors = []
        
        # Global parameters (3 parameters, 1 each)
        for param in ['beta_0', 'alpha', 'sigma']:
            base_bij = bijector_specs[param]
            param_data = repr_theta[param].reshape(-1, 1)
            mean_val = jnp.mean(base_bij.forward(param_data))
            std_val = jnp.std(base_bij.forward(param_data))
            z_scaled_bij = tfb.Chain([
                tfb.Scale(1.0 / jnp.maximum(std_val, 1e-8)),
                tfb.Shift(-mean_val),
                base_bij
            ])
            individual_bijectors.append(z_scaled_bij)
        
        # Site-specific parameters (3 parameters, n_sites each)
        for param in ['A', 'T_season', 'phi']:
            base_bij = bijector_specs[param]
            param_data = repr_theta[param].reshape(-1, n_sites)
            mean_val = jnp.mean(base_bij.forward(param_data), axis=0)
            std_val = jnp.std(base_bij.forward(param_data), axis=0)
            z_scaled_bij = tfb.Chain([
                tfb.Scale(1.0 / jnp.maximum(std_val, 1e-8)),
                tfb.Shift(-mean_val),
                base_bij
            ])
            individual_bijectors.append(z_scaled_bij)
        
        # Create blockwise bijector
        return tfb.Blockwise(
            bijectors=individual_bijectors,
            block_sizes=[1, 1, 1, n_sites, n_sites, n_sites]
        )
    
    # Create MCMC bijector
    flat_theta_bijector = create_flat_blockwise_bijector(repr_theta, bijector_specs, n_sites)

    # Create proxy functions for MCMC sampling
    def flat_prior_fn(key: Array, n_samples: int) -> Array:
        """Prior function compatible with FMPE interface"""
        theta_samples = prior_fn(n_sites).sample((n_samples,), seed=key)
        # Flatten and transform to unconstrained space
        return flatten_theta_dict(theta_samples)

    prior = prior_fn(n_sites)

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

    az.plot_posterior(
        posterior,
        var_names=['beta_0', 'alpha', 'sigma'],
        ref_val=[
            float(theta_truth['beta_0'][0, 0, 0]),
            float(theta_truth['alpha'][0, 0, 0]),
            float(theta_truth['sigma'][0, 0, 0])
        ],
    )

    # Use Hydra's output directory
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.runtime.output_dir)
    plt.savefig(out_dir / "seir_mcmc_posterior.png", dpi=300)
    plt.close()
   
    # Generate posterior predictive checks
    logger.info("Generating posterior predictive check plots")
    ppc_key, key = jr.split(key)
    plot_posterior_predictive_checks(
        mcmc_posterior_samples=mcmc_posterior_samples.all_states, #flatten_theta_dict(repr_theta), 
        theta_truth=theta_truth,
        y_observed=y_observed,
        f_in=f_in,
        cfg=cfg,
        out_dir=out_dir,
        key=ppc_key,
        n_warmup=n_warmup
    )
    
    logger.info("SEIR MCMC experiment completed successfully!")


def plot_posterior_predictive_checks(
    mcmc_posterior_samples: Array,
    theta_truth: Dict[str, Array],
    y_observed: Dict[str, Array],
    f_in: Dict[str, Array],
    cfg: DictConfig,
    out_dir: Path,
    key: Array,
    n_warmup: int
) -> None:
    """
    Generate posterior predictive check plots showing incidence trajectories.
    
    Creates separate plots for each site showing:
    - True trajectory (solid line) from theta_truth
    - 95% credible bands from posterior samples (shaded area)
    - Observed data points (crosses) at observation times
    
    Parameters
    ----------
    mcmc_posterior_samples : Array
        MCMC samples from posterior distribution
    theta_truth : Dict[str, Array]
        Ground truth parameters used to generate observations
    y_observed : Dict[str, Array]
        Observed incidence data
    f_in : Dict[str, Array]
        Functional input containing observation times
    cfg : DictConfig
        Configuration parameters
    out_dir : Path
        Output directory for saving plots
    key : Array
        Random key for posterior sampling
    """
    n_sites = cfg.n_sites
    n_timesteps = cfg.n_timesteps
    population = cfg.population
    I0_prop = cfg.I0_prop
    
    # Create dense time grid for smooth trajectories including warmup
    t_dense_full = jnp.linspace(0, n_warmup + n_timesteps, (n_warmup + n_timesteps) * 4)
    # Create plotting grid (post-warmup only)
    t_dense_plot = jnp.linspace(0, n_timesteps, n_timesteps * 4)
    
    # Initial conditions
    I0 = population * I0_prop
    S0 = population - I0
    initial_state = jnp.array([S0, 0.0, I0, 0.0])  # [S, E, I, R]
    
    def solve_trajectory(params_dict: Dict[str, Array], times: Array) -> Array:
        """Solve SEIR ODE for given parameters and return incidence."""
        def ode_func(state, t):
            return seir_dynamics(state, t, params_dict)
        
        solution = odeint(ode_func, initial_state, times)
        exposed = solution[:, 1]  # E compartment
        incidence = params_dict['alpha'] * exposed
        return jnp.maximum(incidence, 1e-8)
    
    def solve_trajectory_post_warmup(params_dict: Dict[str, Array]) -> Array:
        """Solve trajectory over full time including warmup, return post-warmup."""
        full_solution = solve_trajectory(params_dict, t_dense_full)
        # Extract post-warmup portion by finding indices after warmup
        warmup_idx = int(len(t_dense_full) * n_warmup / (n_warmup + n_timesteps))
        return full_solution[warmup_idx:]
    
    # Generate true trajectories for each site
    true_trajectories = []
    for site_idx in range(n_sites):
        params_dict = {
            'beta_0': theta_truth['beta_0'][0, 0, 0],
            'alpha': theta_truth['alpha'][0, 0, 0],
            'sigma': theta_truth['sigma'][0, 0, 0],
            'A': theta_truth['A'][0, site_idx, 0],
            'T_season': theta_truth['T_season'][0, site_idx, 0],
            'phi': theta_truth['phi'][0, site_idx, 0]
        }
        true_traj = solve_trajectory_post_warmup(params_dict)
        true_trajectories.append(true_traj)
    
    # Convert MCMC samples to structured format
    post_dict = reconstruct_theta_dict(
        mcmc_posterior_samples,
        n_sites
    )
    
    # Flatten first two dimensions using tree.map
    flattened_post = tree.map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
        post_dict
    )
    
    n_total_samples = flattened_post['beta_0'].shape[0]
    n_ensemble = min(100, n_total_samples)
    ensemble_indices = jr.choice(key, n_total_samples, shape=(n_ensemble,), replace=False)
    
    # Sample parameters using tree.map
    sampled_post = tree.map(
        lambda x: x[ensemble_indices],
        flattened_post
    )
    
    # Generate posterior trajectories for each site
    posterior_trajectories = {site_idx: [] for site_idx in range(n_sites)}
    
    for sample_idx in range(n_ensemble):
        for site_idx in range(n_sites):
            params_dict = {
                'beta_0': sampled_post['beta_0'][sample_idx, 0],
                'alpha': sampled_post['alpha'][sample_idx, 0],
                'sigma': sampled_post['sigma'][sample_idx, 0],
                'A': sampled_post['A'][sample_idx, site_idx],
                'T_season': sampled_post['T_season'][sample_idx, site_idx],
                'phi': sampled_post['phi'][sample_idx, site_idx]
            }
            
            traj = solve_trajectory_post_warmup(params_dict)
            posterior_trajectories[site_idx].append(traj)
    
    # Convert to arrays and compute percentiles
    for site_idx in range(n_sites):
        posterior_trajectories[site_idx] = jnp.array(posterior_trajectories[site_idx])
    
    # Create plots for each site
    for site_idx in range(n_sites):
        plt.figure(figsize=(10, 6))
        
        # Convert to cases per 100,000
        scale_factor = 100000.0 / population
        
        # Plot true trajectory
        true_scaled = true_trajectories[site_idx] * scale_factor
        plt.plot(t_dense_plot, true_scaled, 'k-', linewidth=2, label='True trajectory')
        
        # Plot posterior credible bands
        post_traj = posterior_trajectories[site_idx]
        post_scaled = post_traj * scale_factor
        percentiles = jnp.percentile(
            post_scaled,
            jnp.array([2.5, 97.5]),
            axis=0
        )
        plt.fill_between(
            t_dense_plot, percentiles[0], percentiles[1], 
            alpha=0.3, color='blue', label='95% Credible interval'
        )
        
        # Plot observed data points
        obs_times = f_in['obs'][0, site_idx, :, 0]  # Extract times for this site
        obs_values = y_observed['obs'][0, site_idx, :, 0]  # Extract observations
        obs_scaled = obs_values * scale_factor
        plt.scatter(obs_times, obs_scaled, marker='x', s=50, color='red', 
                   linewidth=2, label='Observations')
        
        plt.xlabel('Time since warmup (days)')
        plt.ylabel('Incidence per 100,000')
        plt.title(f'Posterior Predictive Check - Site {site_idx + 1} (Post-warmup)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(out_dir / f"seir_ppc_site_{site_idx + 1}.png", dpi=300, bbox_inches='tight')
        plt.close()


def _flatten(x: PyTree) -> jnp.ndarray:
    """Flatten a batched SFMPE PyTree into a 2D array."""
    return jnp.concatenate(
        [v.reshape(v.shape[0], -1) for v in x.values()],
        axis=-1
    )

def flatten_f_in(f_in_data: PyTree, pad_value: float = -1e8, 
                 data_sample_ndims: int = 1) -> PyTree:
    """
    Flatten f_in data for use as index in SFMPE posterior sampling.
    
    Uses the same methodology as flatten_structured: splits f_in into 
    'theta' and 'y' blocks based on parameter structure, then applies
    _flatten_index to each block.
    """
    from sfmpe.util.dataloader import _flatten_index
    
    # Define which keys go to which block (matching the data structure)
    theta_keys = ['beta_0', 'alpha', 'sigma', 'A', 'T_season', 'phi']  # all parameters
    y_keys = ['obs']              # observations
    
    # Split f_in_data into theta and y components
    theta_f_in = {k: f_in_data[k] for k in f_in_data.keys() if k in theta_keys}
    y_f_in = {k: f_in_data[k] for k in f_in_data.keys() if k in y_keys}
    
    # Apply _flatten_index to each block
    flattened_index = {
        'theta': _flatten_index(theta_f_in, pad_value, data_sample_ndims),
        'y': _flatten_index(y_f_in, pad_value, data_sample_ndims)
    }
    
    return flattened_index

def reconstruct_theta_dict(theta_flat: Array, n_sites: int) -> Dict[str, Array]:
    """Reconstruct structured theta from flattened array."""
    theta_dict = {}
    idx = 0
    
    # Global parameters (3 parameters, 1 each)
    theta_dict['beta_0'] = theta_flat[..., idx:idx+1, None]
    idx += 1
    theta_dict['alpha'] = theta_flat[..., idx:idx+1, None]
    idx += 1
    theta_dict['sigma'] = theta_flat[..., idx:idx+1, None]
    idx += 1
    
    # Site-specific parameters (3 parameters, n_sites each)
    for param_name in ['A', 'T_season', 'phi']:
        theta_dict[param_name] = theta_flat[..., idx:idx+n_sites, None]
        idx += n_sites
        
    return theta_dict


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
