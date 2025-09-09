"""
SEIR MCMC Visualization Script

This script loads saved MCMC results and generates visualization plots including:
- Posterior distribution plots with true parameter values
- Posterior predictive checks showing credible intervals vs observations

Usage:
    python seir_mcmc_plot.py <hydra_output_directory>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
from jaxtyping import Array

import jax.numpy as jnp
from jax import random as jr, tree
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import arviz as az

from seir_utils import (
    seir_dynamics, reconstruct_theta_dict
)


def plot_posterior_predictive_checks(
    mcmc_posterior_samples: Array,
    theta_truth: Dict[str, Array],
    y_observed: Dict[str, Array],
    f_in: Dict[str, Array],
    plot_config: Dict,
    out_dir: Path,
    key: Array
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
    plot_config : Dict
        Configuration parameters for plotting
    out_dir : Path
        Output directory for saving plots
    key : Array
        Random key for posterior sampling
    """
    n_sites = plot_config['n_sites']
    n_timesteps = plot_config['n_timesteps']
    n_warmup = plot_config['n_warmup']
    population = plot_config['population']
    I0_prop = plot_config['I0_prop']
    
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


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Generate SEIR MCMC visualization plots')
    parser.add_argument('output_dir', type=str, help='Hydra output directory containing saved results')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    out_dir = Path(args.output_dir)
    
    # Verify output directory and required files exist
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")
    
    required_files = [
        "theta_truth.npy",
        "y_observed.npy", 
        "f_in.npy",
        "mcmc_posterior_samples.npy",
        "plot_config.json"
    ]
    
    for filename in required_files:
        if not (out_dir / filename).exists():
            raise FileNotFoundError(f"Required file not found: {out_dir / filename}")
    
    logger.info(f"Loading results from {out_dir}")
    
    # Load saved data
    theta_truth = jnp.load(out_dir / "theta_truth.npy", allow_pickle=True).item()
    y_observed = jnp.load(out_dir / "y_observed.npy", allow_pickle=True).item()
    f_in = jnp.load(out_dir / "f_in.npy", allow_pickle=True).item()
    mcmc_posterior_samples = jnp.load(out_dir / "mcmc_posterior_samples.npy")
    
    with open(out_dir / "plot_config.json", 'r') as f:
        plot_config = json.load(f)
    
    # Convert MCMC samples to structured format for ArviZ plotting
    logger.info("Creating posterior distribution plots")
    n_sites = plot_config['n_sites']
    post_dict = reconstruct_theta_dict(
        mcmc_posterior_samples,
        n_sites
    )
    posterior = az.from_dict(posterior=post_dict)
    
    # Create posterior plots
    az.plot_posterior(
        posterior,
        var_names=['beta_0', 'alpha', 'sigma'],
        ref_val=[
            float(theta_truth['beta_0'][0, 0, 0]),
            float(theta_truth['alpha'][0, 0, 0]),
            float(theta_truth['sigma'][0, 0, 0])
        ],
    )
    plt.savefig(out_dir / "seir_mcmc_posterior.png", dpi=300)
    plt.close()
    
    # Generate posterior predictive checks
    logger.info("Generating posterior predictive check plots")
    key = jr.PRNGKey(42)  # Fixed seed for reproducible plots
    plot_posterior_predictive_checks(
        mcmc_posterior_samples=mcmc_posterior_samples,
        theta_truth=theta_truth,
        y_observed=y_observed,
        f_in=f_in,
        plot_config=plot_config,
        out_dir=out_dir,
        key=key
    )
    
    logger.info("SEIR MCMC visualization completed successfully!")
    logger.info(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
