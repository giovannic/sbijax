"""
SEIR MCMC Visualization Script

This script loads saved MCMC/SFMPE results and generates visualization plots including:
- Posterior distribution plots with true parameter values
- Posterior predictive checks showing credible intervals vs observations
- Comparative plots when multiple methods are provided

Usage:
    Single method:
        python seir_mcmc_plot.py path/to/job_dir --plot_dir output_dir

    Multiple methods (comparative):
        python seir_mcmc_plot.py path/to/job1 path/to/job2 --plot_dir output_dir

    Example:
        python seir_mcmc_plot.py /Volumes/gc1610/home/ess_selective /Volumes/gc1610/home/nuts_selective --n_ppc_samples 10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from jaxtyping import Array
import yaml

import jax.numpy as jnp
from jax import random as jr, tree
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
import pandas as pd

from seir_utils import (
    seir_dynamics, reconstruct_selective_theta_dict
)


@dataclass
class JobData:
    """Container for data loaded from a single job directory."""
    job_dir: Path
    method_label: str
    theta_truth: Dict[str, Array]
    y_observed: Dict[str, Array]
    f_in: Dict[str, Array]
    mcmc_posterior_samples: Array
    prior_samples: Array
    plot_config: Dict
    selective_config: Dict


def identify_method(job_dir: Path) -> str:
    """
    Identify the method used for a job by reading its Hydra config.

    Parameters
    ----------
    job_dir : Path
        Job directory containing .hydra/config.yaml

    Returns
    -------
    str
        Method label (e.g., "SFMPE", "MCMC-slice", "MCMC-nuts", "MCMC-ess")
    """
    config_file = job_dir / ".hydra" / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    method = config.get('method', 'UNKNOWN')

    if method == 'SFMPE':
        return 'SFMPE'
    elif method == 'MCMC':
        sampler = config.get('mcmc', {}).get('sampler', 'unknown')
        return f'MCMC-{sampler}'
    else:
        return f'UNKNOWN-{method}'


def load_job_data(job_dir: Path) -> JobData:
    """
    Load all required data from a job directory.

    Parameters
    ----------
    job_dir : Path
        Job directory containing saved results

    Returns
    -------
    JobData
        Container with all loaded data
    """
    # Verify job directory exists
    if not job_dir.exists():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    # Check required files exist
    required_files = [
        "theta_truth.npy",
        "y_observed.npy",
        "f_in.npy",
        "mcmc_posterior_samples.npy",
        "prior_samples.npy",
        "plot_config.json",
        "selective_inference_config.json"
    ]

    for filename in required_files:
        if not (job_dir / filename).exists():
            raise FileNotFoundError(f"Required file not found: {job_dir / filename}")

    # Identify method
    method_label = identify_method(job_dir)

    # Load data
    theta_truth = jnp.load(job_dir / "theta_truth.npy", allow_pickle=True).item()
    y_observed = jnp.load(job_dir / "y_observed.npy", allow_pickle=True).item()
    f_in = jnp.load(job_dir / "f_in.npy", allow_pickle=True).item()
    mcmc_posterior_samples = jnp.load(job_dir / "mcmc_posterior_samples.npy")
    prior_samples = jnp.load(job_dir / "prior_samples.npy")

    with open(job_dir / "plot_config.json", 'r') as f:
        plot_config = json.load(f)

    with open(job_dir / "selective_inference_config.json", 'r') as f:
        selective_config = json.load(f)

    return JobData(
        job_dir=job_dir,
        method_label=method_label,
        theta_truth=theta_truth,
        y_observed=y_observed,
        f_in=f_in,
        mcmc_posterior_samples=mcmc_posterior_samples,
        prior_samples=prior_samples,
        plot_config=plot_config,
        selective_config=selective_config
    )


def validate_job_compatibility(job_data_list: List[JobData]) -> None:
    """
    Validate that all jobs have compatible configurations for comparison.

    Parameters
    ----------
    job_data_list : List[JobData]
        List of loaded job data

    Raises
    ------
    ValueError
        If jobs have incompatible configurations
    """
    if len(job_data_list) < 1:
        raise ValueError("At least one job required")

    reference = job_data_list[0]

    # Check plot config compatibility
    for job_data in job_data_list[1:]:
        for key in ['n_sites', 'n_timesteps', 'n_warmup', 'population', 'I0_prop', 'dt']:
            if job_data.plot_config[key] != reference.plot_config[key]:
                raise ValueError(
                    f"Incompatible {key}: {job_data.job_dir.name} has "
                    f"{job_data.plot_config[key]}, but {reference.job_dir.name} has "
                    f"{reference.plot_config[key]}"
                )

    # Check selective config compatibility (sampled parameters must match)
    ref_params = set(reference.selective_config['sample_params'])
    for job_data in job_data_list[1:]:
        job_params = set(job_data.selective_config['sample_params'])
        if job_params != ref_params:
            raise ValueError(
                f"Incompatible sampled parameters: {job_data.job_dir.name} samples "
                f"{job_params}, but {reference.job_dir.name} samples {ref_params}"
            )


def get_method_colors() -> Dict[str, str]:
    """
    Get consistent color mapping for different methods.

    Returns
    -------
    Dict[str, str]
        Method name to color mapping
    """
    return {
        'SFMPE': '#1f77b4',      # Blue
        'MCMC-slice': '#ff7f0e',  # Orange
        'MCMC-nuts': '#2ca02c',   # Green
        'MCMC-ess': '#d62728',    # Red
    }


def prepare_posterior_data(
    job_data: JobData,
    sample_params: List[str],
    fixed_params: Dict[str, Array],
    n_sites: int
) -> Dict[str, Array]:
    """
    Prepare posterior data for a single method.

    Parameters
    ----------
    job_data : JobData
        Job data containing posterior samples
    sample_params : List[str]
        List of sampled parameter names
    fixed_params : Dict[str, Array]
        Fixed parameter values
    n_sites : int
        Number of sites

    Returns
    -------
    Dict[str, Array]
        Reconstructed posterior dictionary filtered to sampled parameters
    """
    post_dict = reconstruct_selective_theta_dict(
        job_data.mcmc_posterior_samples, sample_params, fixed_params, n_sites
    )
    return {k: v for k, v in post_dict.items() if k in sample_params}


def generate_trajectories(
    posterior_samples: Array,
    sample_params: List[str],
    fixed_params: Dict[str, Array],
    plot_config: Dict,
    n_sites: int,
    n_ppc_samples: int = 0,
    key: Optional[Array] = None
) -> List[Array]:
    """
    Generate posterior trajectories for all sites.

    Parameters
    ----------
    posterior_samples : Array
        Posterior samples in flat format
    sample_params : List[str]
        List of sampled parameter names
    fixed_params : Dict[str, Array]
        Fixed parameter values
    plot_config : Dict
        Plot configuration
    n_sites : int
        Number of sites
    n_ppc_samples : int
        Number of samples to use (0 = use all)
    key : Optional[Array]
        Random key for sampling

    Returns
    -------
    List[Array]
        List of trajectory arrays, one per site
    """
    n_timesteps = plot_config['n_timesteps']
    n_warmup = plot_config['n_warmup']
    population = plot_config['population']
    I0_prop = plot_config['I0_prop']

    # Create dense time grids
    t_dense_full = jnp.linspace(0, n_warmup + n_timesteps, (n_warmup + n_timesteps) * 4)

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

    # Reconstruct posterior
    post_dict = reconstruct_selective_theta_dict(
        posterior_samples, sample_params, fixed_params, n_sites
    )

    # Flatten first two dimensions
    flattened_post = tree.map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
        post_dict
    )

    n_total_samples = flattened_post['beta_0'].shape[0]
    if n_ppc_samples == 0:
        n_ensemble = n_total_samples
    else:
        n_ensemble = min(n_ppc_samples, n_total_samples)

    if n_ensemble == n_total_samples:
        ensemble_indices = jnp.arange(n_total_samples)
    else:
        if key is None:
            key = jr.PRNGKey(42)
        ensemble_indices = jr.choice(key, n_total_samples, shape=(n_ensemble,), replace=False)

    # Sample parameters
    sampled_post = tree.map(
        lambda x: x[ensemble_indices],
        flattened_post
    )

    # Generate trajectories for each site
    posterior_trajectories = []
    for site_idx in range(n_sites):
        site_trajectories = []
        for sample_idx in range(n_ensemble):
            params_dict = {
                'beta_0': sampled_post['beta_0'][sample_idx, 0],
                'alpha': sampled_post['alpha'][sample_idx, 0],
                'sigma': sampled_post['sigma'][sample_idx, 0],
                'A': sampled_post['A'][sample_idx, site_idx],
                'T_season': sampled_post['T_season'][sample_idx, site_idx],
                'phi': sampled_post['phi'][sample_idx, site_idx]
            }
            traj = solve_trajectory_post_warmup(params_dict)
            site_trajectories.append(traj)
        posterior_trajectories.append(jnp.array(site_trajectories))

    return posterior_trajectories


def generate_true_trajectories(
    theta_truth: Dict[str, Array],
    plot_config: Dict,
    n_sites: int
) -> List[Array]:
    """
    Generate true trajectories for all sites.

    Parameters
    ----------
    theta_truth : Dict[str, Array]
        Ground truth parameters
    plot_config : Dict
        Plot configuration
    n_sites : int
        Number of sites

    Returns
    -------
    List[Array]
        List of true trajectory arrays, one per site
    """
    n_timesteps = plot_config['n_timesteps']
    n_warmup = plot_config['n_warmup']
    population = plot_config['population']
    I0_prop = plot_config['I0_prop']

    # Create dense time grids
    t_dense_full = jnp.linspace(0, n_warmup + n_timesteps, (n_warmup + n_timesteps) * 4)

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

    return true_trajectories


def plot_posterior_predictive_checks(
    job_data_list: List[JobData],
    theta_truth: Dict[str, Array],
    y_observed: Dict[str, Array],
    f_in: Dict[str, Array],
    plot_config: Dict,
    out_dir: Path,
    selective_config: Dict,
    title_prefix: str = "Posterior",
    filename_prefix: str = "seir_ppc",
    n_ppc_samples: int = 0,
    key: Optional[Array] = None
) -> None:
    """
    Generate posterior predictive check plots showing incidence trajectories.
    Supports both single and multiple methods.

    Creates separate plots for each site showing:
    - True trajectory (solid black line) from theta_truth
    - 95% credible bands from each method (different colors)
    - Observed data points (crosses) at observation times
    - Legend identifying all methods (if multiple)

    Parameters
    ----------
    job_data_list : List[JobData]
        List of job data (single method or multiple for comparison)
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
    selective_config : Dict
        Configuration specifying which parameters were sampled
    title_prefix : str
        Prefix for plot titles
    filename_prefix : str
        Prefix for saved filenames
    n_ppc_samples : int
        Number of samples to use for predictive checks (0 = use all samples)
    key : Optional[Array]
        Random key for posterior sampling
    """
    n_sites = plot_config['n_sites']
    n_timesteps = plot_config['n_timesteps']
    population = plot_config['population']
    method_colors = get_method_colors()

    # Create plotting grid (post-warmup only)
    t_dense_plot = jnp.linspace(0, n_timesteps, n_timesteps * 4)

    # Generate true trajectories using helper function
    true_trajectories = generate_true_trajectories(theta_truth, plot_config, n_sites)

    # Generate posterior trajectories for each method
    method_trajectories = {}
    sample_params = selective_config['sample_params']

    for job_data in job_data_list:
        fixed_params = {k: jnp.array(v) for k, v in job_data.selective_config['fixed_params'].items()}

        posterior_trajectories = generate_trajectories(
            job_data.mcmc_posterior_samples,
            sample_params,
            fixed_params,
            plot_config,
            n_sites,
            n_ppc_samples,
            key
        )
        method_trajectories[job_data.method_label] = posterior_trajectories

    # Create plots for each site
    for site_idx in range(n_sites):
        plt.figure(figsize=(12, 8) if len(job_data_list) > 1 else (10, 6))

        # Convert to cases per 100,000
        scale_factor = 100000.0 / population

        # Plot true trajectory
        true_scaled = true_trajectories[site_idx] * scale_factor
        plt.plot(t_dense_plot, true_scaled, 'k-', linewidth=3,
                label='True trajectory', zorder=10)

        if len(job_data_list) > 1:
            # Multi-method: plot credible bands for each method
            for method_label, trajectories in method_trajectories.items():
                color = method_colors.get(method_label, '#000000')
                post_traj = trajectories[site_idx]
                post_scaled = post_traj * scale_factor
                percentiles = jnp.percentile(
                    post_scaled,
                    jnp.array([2.5, 97.5]),
                    axis=0
                )
                plt.fill_between(
                    t_dense_plot, percentiles[0], percentiles[1],
                    alpha=0.3, color=color, label=f'{method_label} (95% CI)'
                )
        else:
            # Single method: use blue credible band
            method_label = list(method_trajectories.keys())[0]
            post_traj = method_trajectories[method_label][site_idx]
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
                   linewidth=2, label='Observations', zorder=10)

        plt.xlabel('Time since warmup (days)')
        plt.ylabel('Incidence per 100,000')

        if len(job_data_list) > 1:
            plt.title(f'Comparative {title_prefix} Predictive Check - Site {site_idx + 1} (Post-warmup)')
        else:
            plt.title(f'{title_prefix} Predictive Check - Site {site_idx + 1} (Post-warmup)')

        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # Save plot
        if len(job_data_list) > 1:
            save_filename = f"seir_comparative_{filename_prefix}_site_{site_idx + 1}.png"
        else:
            save_filename = f"{filename_prefix}_site_{site_idx + 1}.png"

        plt.savefig(out_dir / save_filename, dpi=300, bbox_inches='tight')
        plt.close()


def plot_prior_posterior_comparison(
    inference_data: az.InferenceData,
    out_dir: Path
) -> None:
    """
    Create prior/posterior comparison plots using ArviZ.
    
    Parameters
    ----------
    inference_data : az.InferenceData
        ArviZ InferenceData containing both prior and posterior
    out_dir : Path
        Output directory for saving plots
    """
    az.plot_dist_comparison(inference_data)
    plt.savefig(out_dir / "seir_prior_posterior_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_prior_predictive_checks(
    job_data_list: List[JobData],
    theta_truth: Dict[str, Array],
    y_observed: Dict[str, Array],
    f_in: Dict[str, Array],
    plot_config: Dict,
    out_dir: Path,
    selective_config: Dict,
    n_ppc_samples: int = 0,
    key: Optional[Array] = None
) -> None:
    """
    Generate prior predictive check plots showing incidence trajectories.
    """
    # Create job data with prior samples instead of posterior samples
    prior_job_data_list = []
    for job_data in job_data_list:
        prior_job_data = JobData(
            job_dir=job_data.job_dir,
            method_label=job_data.method_label,
            theta_truth=job_data.theta_truth,
            y_observed=job_data.y_observed,
            f_in=job_data.f_in,
            mcmc_posterior_samples=job_data.prior_samples,  # Use prior samples
            prior_samples=job_data.prior_samples,
            plot_config=job_data.plot_config,
            selective_config=job_data.selective_config
        )
        prior_job_data_list.append(prior_job_data)

    plot_posterior_predictive_checks(
        job_data_list=prior_job_data_list,
        theta_truth=theta_truth,
        y_observed=y_observed,
        f_in=f_in,
        plot_config=plot_config,
        out_dir=out_dir,
        selective_config=selective_config,
        title_prefix="Prior",
        filename_prefix="seir_prior_ppc",
        n_ppc_samples=n_ppc_samples,
        key=key
    )


def plot_pairplot_with_reference(
    job_data_list: List[JobData],
    theta_truth: Dict[str, Array],
    selective_config: Dict,
    plot_config: Dict,
    out_dir: Path,
    filename: str = "seir_mcmc_pairplot.png",
    n_pair_samples: int = 100
) -> None:
    """
    Create pairplot with KDE density plots and reference values from truth.
    Supports both single and multiple methods.

    Creates a pairplot showing:
    - KDE density plots on the lower triangle
    - Marginal density distributions on the diagonal
    - Reference values from theta_truth overlaid as markers
    - Different colors for different methods (if multiple)

    Parameters
    ----------
    job_data_list : List[JobData]
        List of job data (single method or multiple for comparison)
    theta_truth : Dict[str, Array]
        Ground truth parameters for reference values
    selective_config : Dict
        Configuration specifying which parameters were sampled
    plot_config : Dict
        Configuration parameters including n_sites
    out_dir : Path
        Output directory for saving plots
    filename : str
        Filename for saved plot
    n_pair_samples : int
        Number of samples to use for pairplot (default: 100)
    """
    sample_params = selective_config['sample_params']
    n_sites = plot_config['n_sites']
    method_colors = get_method_colors()

    # Build combined dataset
    all_data = []
    reference_values = {}

    # Set random seed for reproducible sub-sampling across jobs
    rng_key = jr.PRNGKey(42)

    for job_data in job_data_list:
        # Prepare posterior data for this method
        fixed_params = {k: jnp.array(v) for k, v in job_data.selective_config['fixed_params'].items()}
        post_dict_filtered = prepare_posterior_data(job_data, sample_params, fixed_params, n_sites)

        # Create flat data dictionary for this method
        flat_data = {'method': job_data.method_label}

        for param in sample_params:
            param_data = post_dict_filtered[param]  # Shape: (chains, draws, ...)

            if param in ['beta_0', 'alpha', 'sigma']:
                # Global parameters
                flattened = param_data.reshape(-1)
                flat_data[param] = flattened

                # Set reference value (only once)
                if param not in reference_values:
                    reference_values[param] = float(theta_truth[param][0, 0, 0])

            elif param in ['A', 'T_season', 'phi']:
                # Site-specific parameters
                for site_idx in range(n_sites):
                    col_name = f"{param}_site_{site_idx + 1}"
                    site_data = param_data[:, :, site_idx, 0]
                    flattened = site_data.reshape(-1)
                    flat_data[col_name] = flattened

                    # Set reference value (only once)
                    if col_name not in reference_values:
                        reference_values[col_name] = float(
                            theta_truth[param][0, site_idx, 0]
                        )

        # Sub-sample to ensure equal sample sizes across jobs
        n_samples = len(list(flat_data.values())[1])  # Length of first parameter array
        n_target_samples = min(n_pair_samples, n_samples)

        if n_target_samples < n_samples:
            # Generate sub-sampling indices
            rng_key, subkey = jr.split(rng_key)
            sample_indices = jr.choice(subkey, n_samples, shape=(n_target_samples,), replace=False)

            # Apply sub-sampling to all parameter arrays
            for param_name in flat_data:
                if param_name != 'method':
                    flat_data[param_name] = flat_data[param_name][sample_indices]

        # Update n_samples after sub-sampling
        n_samples = n_target_samples

        # Convert to long format
        for i in range(n_samples):
            row = {'method': job_data.method_label}
            for key, values in flat_data.items():
                if key != 'method':
                    row[key] = float(values[i])
            all_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    if len(job_data_list) > 1:
        # Multi-method: use hue for different colors
        methods_present = df['method'].unique()
        palette = {method: method_colors.get(method, '#000000') for method in methods_present}

        g = sns.pairplot(
            df,
            hue='method',
            palette=palette,
            kind='kde',
            diag_kind='kde',
            plot_kws={'alpha': 0.6},
            diag_kws={'alpha': 0.7}
        )
    else:
        # Single method: no hue needed
        param_columns = [col for col in df.columns if col != 'method']
        df_single = df[param_columns]

        g = sns.pairplot(
            df_single,
            kind='kde',
            diag_kind='kde',
            plot_kws={'alpha': 0.6},
            diag_kws={'alpha': 0.7}
        )

    # Add reference values as red X markers
    param_columns = [col for col in df.columns if col != 'method']
    for i, var1 in enumerate(param_columns):
        for j, var2 in enumerate(param_columns):
            if i != j:  # Off-diagonal plots
                ax = g.axes[i, j]
                if i > j:  # Lower triangle (where KDE plots are)
                    ax.scatter(
                        reference_values[var2],
                        reference_values[var1],
                        marker='x',
                        color='red',
                        s=100,
                        linewidths=3,
                        zorder=10,
                        label='Truth' if i == 1 and j == 0 else ""
                    )
            else:  # Diagonal plots
                ax = g.axes[i, j]
                ax.axvline(
                    reference_values[var1],
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.8,
                    label='Truth' if i == 0 else ""
                )

    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()



def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Generate SEIR MCMC visualization plots')
    parser.add_argument('job_dirs', nargs='*', type=str,
                        help='Hydra output directories containing saved results. '
                             'For single directory: provide one path. '
                             'For comparison: provide multiple paths.')
    parser.add_argument('--plot_dir', type=str, default='examples/outputs',
                        help='Directory to save plots (default: examples/outputs)')
    parser.add_argument('--n_ppc_samples', type=int, default=0,
                        help='Number of samples to use for predictive checks (0 = use all samples)')
    parser.add_argument('--n_pair_samples', type=int, default=100,
                        help='Number of samples to use for pairplot (default: 100)')
    args = parser.parse_args()

    # Handle backward compatibility: if no job_dirs provided, look for old-style positional arg
    if not args.job_dirs:
        parser.error("At least one job directory must be provided")

    # Convert to absolute paths
    job_dirs = [Path(job_dir).resolve() for job_dir in args.job_dirs]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    out_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {len(job_dirs)} job directories")

    # Load all job data
    job_data_list = []
    for job_dir in job_dirs:
        try:
            job_data = load_job_data(job_dir)
            logger.info(f"Loaded {job_data.method_label} from {job_dir.name}")
            job_data_list.append(job_data)
        except Exception as e:
            logger.error(f"Failed to load data from {job_dir}: {e}")
            raise

    # Validate compatibility
    try:
        validate_job_compatibility(job_data_list)
        logger.info("All job configurations are compatible")
    except Exception as e:
        logger.error(f"Job compatibility check failed: {e}")
        raise

    # Determine if this is single or multi-method analysis
    is_comparative = len(job_data_list) > 1
    method_labels = [job_data.method_label for job_data in job_data_list]

    if is_comparative:
        logger.info(f"Performing comparative analysis of methods: {', '.join(method_labels)}")
    else:
        logger.info(f"Performing single-method analysis of: {method_labels[0]}")

    # Use the first job's data for reference (all should have same truth/observations)
    reference_job = job_data_list[0]

    # Extract shared data for convenience
    theta_truth = reference_job.theta_truth
    y_observed = reference_job.y_observed
    f_in = reference_job.f_in
    plot_config = reference_job.plot_config
    selective_config = reference_job.selective_config

    logger.info(f"Using reference configuration from {reference_job.job_dir.name}")
    logger.info(f"Sampling parameters: {selective_config['sample_params']}")

    # Generate plots based on whether this is single or multi-method
    if is_comparative:
        # Multi-method comparative analysis

        # Generate ArviZ summary for reference job
        logger.info("Creating posterior distribution summary (reference method)")
        sample_params = selective_config['sample_params']
        fixed_params = {k: jnp.array(v) for k, v in selective_config['fixed_params'].items()}
        n_sites = plot_config['n_sites']

        # Use reference job for ArviZ analysis
        post_dict = reconstruct_selective_theta_dict(
            reference_job.mcmc_posterior_samples, sample_params, fixed_params, n_sites
        )
        post_dict_filtered = {k: v for k, v in post_dict.items() if k in sample_params}
        inference_data = az.from_dict(posterior=post_dict_filtered)
        print("Reference method summary:")
        print(az.summary(inference_data))

        # Generate comparative plots
        logger.info("Generating comparative pairplot")
        plot_pairplot_with_reference(
            job_data_list=job_data_list,
            theta_truth=theta_truth,
            selective_config=selective_config,
            plot_config=plot_config,
            out_dir=out_dir,
            filename="seir_comparative_pairplot.png",
            n_pair_samples=args.n_pair_samples
        )

        logger.info("Generating comparative posterior predictive checks")
        plot_posterior_predictive_checks(
            job_data_list=job_data_list,
            theta_truth=theta_truth,
            y_observed=y_observed,
            f_in=f_in,
            plot_config=plot_config,
            out_dir=out_dir,
            selective_config=selective_config,
            title_prefix="Posterior",
            filename_prefix="seir_ppc",
            n_ppc_samples=args.n_ppc_samples
        )

        # Generate individual method plots for comparison
        for job_data in job_data_list:
            method_dir = out_dir / f"individual_{job_data.method_label.replace('-', '_')}"
            method_dir.mkdir(exist_ok=True)

            # Individual ArviZ plots
            job_post_dict = reconstruct_selective_theta_dict(
                job_data.mcmc_posterior_samples, sample_params, fixed_params, n_sites
            )
            job_prior_dict = reconstruct_selective_theta_dict(
                job_data.prior_samples, sample_params, fixed_params, n_sites
            )
            job_post_filtered = {k: v for k, v in job_post_dict.items() if k in sample_params}
            job_prior_filtered = {k: v for k, v in job_prior_dict.items() if k in sample_params}
            job_inference_data = az.from_dict(
                posterior=job_post_filtered,
                prior=job_prior_filtered
            )

            # Individual posterior plots
            global_sampled = [p for p in sample_params if p in ['beta_0', 'alpha', 'sigma']]
            if global_sampled:
                ref_vals = [float(theta_truth[param][0, 0, 0]) for param in global_sampled]
                az.plot_posterior(job_inference_data, var_names=global_sampled, ref_val=ref_vals)
                plt.savefig(method_dir / f"{job_data.method_label}_posterior.png", dpi=300)
                plt.close()

            # Individual trace plot
            az.plot_trace(job_inference_data)
            plt.savefig(method_dir / f"{job_data.method_label}_trace.png", dpi=300)
            plt.close()

            # Individual pairplot
            plot_pairplot_with_reference(
                job_data_list=[job_data],
                theta_truth=theta_truth,
                selective_config=selective_config,
                plot_config=plot_config,
                out_dir=method_dir,
                filename="seir_mcmc_pairplot.png",
                n_pair_samples=args.n_pair_samples
            )

    else:
        # Single-method analysis - use original workflow
        logger.info("Performing single-method analysis")
        job_data = job_data_list[0]

        sample_params = selective_config['sample_params']
        fixed_params = {k: jnp.array(v) for k, v in selective_config['fixed_params'].items()}
        n_sites = plot_config['n_sites']

        # Reconstruct posterior and prior
        post_dict = reconstruct_selective_theta_dict(
            job_data.mcmc_posterior_samples, sample_params, fixed_params, n_sites
        )
        prior_dict = reconstruct_selective_theta_dict(
            job_data.prior_samples, sample_params, fixed_params, n_sites
        )

        # Filter to sampled parameters
        post_dict_filtered = {k: v for k, v in post_dict.items() if k in sample_params}
        prior_dict_filtered = {k: v for k, v in prior_dict.items() if k in sample_params}

        # Create ArviZ InferenceData
        inference_data = az.from_dict(
            posterior=post_dict_filtered,
            prior=prior_dict_filtered
        )

        # Generate summary statistics
        print(az.summary(inference_data))

        # Create posterior plots
        global_sampled = [p for p in sample_params if p in ['beta_0', 'alpha', 'sigma']]
        if global_sampled:
            ref_vals = [float(theta_truth[param][0, 0, 0]) for param in global_sampled]
            az.plot_posterior(inference_data, var_names=global_sampled, ref_val=ref_vals)
            plt.savefig(out_dir / "seir_mcmc_posterior.png", dpi=300)
            plt.close()

        # Generate prior/posterior comparison plots
        logger.info("Generating prior/posterior comparison plots")
        plot_prior_posterior_comparison(inference_data, out_dir)

        # Generate trace plot
        logger.info("Generating trace plot")
        az.plot_trace(inference_data)
        plt.savefig(out_dir / "seir_mcmc_trace.png", dpi=300)
        plt.close()

        # Generate pairplot
        logger.info("Generating pairplot with reference values")
        plot_pairplot_with_reference(
            job_data_list=[job_data],
            theta_truth=theta_truth,
            selective_config=selective_config,
            plot_config=plot_config,
            out_dir=out_dir,
            filename="seir_mcmc_pairplot.png",
            n_pair_samples=args.n_pair_samples
        )

        # Generate posterior predictive checks
        logger.info("Generating posterior predictive check plots")
        plot_posterior_predictive_checks(
            job_data_list=[job_data],
            theta_truth=theta_truth,
            y_observed=y_observed,
            f_in=f_in,
            plot_config=plot_config,
            out_dir=out_dir,
            selective_config=selective_config,
            title_prefix="Posterior",
            filename_prefix="seir_ppc",
            n_ppc_samples=args.n_ppc_samples,
            key=jr.PRNGKey(42)
        )

        # Generate prior predictive checks
        logger.info("Generating prior predictive check plots")
        plot_prior_predictive_checks(
            job_data_list=[job_data],
            theta_truth=theta_truth,
            y_observed=y_observed,
            f_in=f_in,
            plot_config=plot_config,
            out_dir=out_dir,
            selective_config=selective_config,
            n_ppc_samples=args.n_ppc_samples,
            key=jr.PRNGKey(43)
        )

    logger.info("SEIR visualization completed successfully!")
    logger.info(f"Plots saved to: {out_dir}")
    if is_comparative:
        logger.info(f"Comparative plots: seir_comparative_pairplot.png, seir_comparative_ppc_site_*.png")
        logger.info(f"Individual method plots saved in subdirectories")


if __name__ == "__main__":
    main()
