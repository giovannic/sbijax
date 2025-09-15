"""
Shared utilities for SEIR model implementation.

This module contains common functions used by both SEIR MCMC estimation
and visualization scripts, including:
- SEIR dynamics and simulation
- Prior distributions
- Parameter transformation utilities
- Data preprocessing functions
"""

from typing import Callable, Dict
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from jax import random as jr, vmap, tree
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, ForwardMode
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
from sfmpe.pytree_bijector import (
    PyTreeBijector,
    create_zscaling_bijector_tree
)
import numpyro
import numpyro.distributions as dist


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
    t_season_spread = 1./7.
    return tfd.JointDistributionNamed(
        dict(
            # Global parameters (independent of obs by exchangeability)
            beta_0 = tfd.Uniform(jnp.full((1, 1), 0.1), jnp.full((1, 1), 2.0)),
            alpha = tfd.Uniform(jnp.full((1, 1), 1/30), jnp.full((1, 1), 1/7)),
            sigma = tfd.Uniform(jnp.full((1, 1), 1/21), jnp.full((1, 1), 1/7)),

            # Local parameters are independent of global parameters
            A = tfd.Uniform(jnp.full((n, 1), .2), jnp.full((n, 1), .5)),
            T_season = tfd.Gamma(
                jnp.full((n, 1), 365.0 * t_season_spread),
                jnp.full((n, 1), t_season_spread)
            ),
            phi = tfd.Uniform(
                jnp.zeros((n, 1)),
                jnp.full((n, 1), jnp.pi)
            )
        ),
        batch_ndims=1,
    )


def create_selective_prior_fn(
    n_sites: int,
    sample_params: list[str],
    fixed_params: Dict[str, Array]
) -> Callable:
    """
    Create prior distribution that only samples specified parameters.

    Args:
        n_sites: Number of observation sites
        sample_params: List of parameter names to sample
        fixed_params: Dictionary of fixed parameter values

    Returns:
        Prior function that returns distribution over sampled parameters only
    """
    t_season_spread = 1./7.

    def selective_prior_fn(n):
        prior_dict = {}

        # Only include distributions for parameters we want to sample
        if 'beta_0' in sample_params:
            prior_dict['beta_0'] = tfd.Uniform(
                jnp.full((1, 1), 0.1),
                jnp.full((1, 1), 2.0)
            )
        if 'alpha' in sample_params:
            prior_dict['alpha'] = tfd.Uniform(
                jnp.full((1, 1), 1/30),
                jnp.full((1, 1), 1/7)
            )
        if 'sigma' in sample_params:
            prior_dict['sigma'] = tfd.Uniform(
                jnp.full((1, 1), 1/21),
                jnp.full((1, 1), 1/7)
            )
        if 'A' in sample_params:
            prior_dict['A'] = tfd.Uniform(
                jnp.full((n, 1), .2),
                jnp.full((n, 1), .5)
            )
        if 'T_season' in sample_params:
            prior_dict['T_season'] = tfd.Gamma(
                jnp.full((n, 1), 365.0 * t_season_spread),
                jnp.full((n, 1), t_season_spread)
            )
        if 'phi' in sample_params:
            prior_dict['phi'] = tfd.Uniform(
                jnp.zeros((n, 1)),
                jnp.full((n, 1), jnp.pi)
            )

        return tfd.JointDistributionNamed(prior_dict, batch_ndims=1)

    return selective_prior_fn


def p_local(g, n):
    """Local prior distribution for site-specific parameters (independent of global)."""
    n_sims = g['beta_0'].shape[0]
    t_season_spread = 1./7.
    return tfd.JointDistributionNamed(
        dict(
            # Site-specific seasonal parameters  
            A = tfd.Uniform(jnp.full((n_sims, n, 1), .2), jnp.full((n_sims, n, 1), .5)),
            T_season = tfd.Gamma(
                jnp.full((n_sims, n, 1), 365.0 * t_season_spread), 
                jnp.full((n_sims, n, 1), t_season_spread)
            ),
            phi = tfd.Uniform(
                jnp.zeros((n_sims, n, 1)), 
                jnp.full((n_sims, n, 1), jnp.pi)
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

            def vector_field(t, y, args):
                return seir_dynamics(y, t, params_dict)
            
            # Solve ODE at observation times for this site
            # Add warmup offset to observation times
            t_eval = jnp.concatenate([jnp.array([0.]), obs_times_single[:, 0] + n_warmup])  # Extract times (n_obs,)
            sort_indices = jnp.argsort(t_eval)
            t_eval_sorted = t_eval[sort_indices]
            
            # Set up diffrax solver
            term = ODETerm(vector_field)
            solver = Dopri5()
            saveat = SaveAt(ts=t_eval_sorted)
            
            solution = diffeqsolve(
                term,
                solver, 
                t0=t_eval_sorted[0],
                t1=t_eval_sorted[-1], 
                dt0=0.1,
                y0=initial_state, 
                saveat=saveat,
                adjoint=ForwardMode(),
                max_steps=None
            )
            
            # Reorder solution to match original time sequence
            reorder_indices = jnp.argsort(sort_indices)
            solution_reordered = solution.ys[reorder_indices]
            
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


def f_in_fn(n_obs: int, n_sites: int, n_timesteps: int):
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


def f_in_fn_observed(n_obs: int, n_sites: int, f_in):
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


def flatten_theta_dict(theta_dict: Dict[str, Array]) -> Array:
    """Flatten theta dictionary to 1D array for MCMC/FMPE."""
    batch_shape = theta_dict['A'].shape[:-2]
    n_sites = theta_dict['A'].shape[-2]
    flattened_parts = []

    # Global parameters (3 parameters, 1 each)
    for param_name in ['beta_0', 'alpha', 'sigma']:
        flattened_parts.append(
            theta_dict[param_name].reshape(batch_shape + (1,))
        )

    # Site-specific parameters (3 parameters, n_sites each)
    for param_name in ['A', 'T_season', 'phi']:
        flattened_parts.append(
            theta_dict[param_name].reshape(batch_shape + (n_sites,))
        )

    return jnp.concatenate(flattened_parts, axis=len(batch_shape))


def flatten_selective_theta_dict(
    theta_dict: Dict[str, Array],
    sample_params: list[str]
) -> Array:
    """
    Flatten only sampled parameters from theta dictionary.

    Args:
        theta_dict: Full theta dictionary
        sample_params: List of parameter names to include

    Returns:
        Flattened array containing only sampled parameters
    """
    # Get dimensions from any site-specific parameter
    if 'A' in theta_dict:
        batch_shape = theta_dict['A'].shape[:-2]
        n_sites = theta_dict['A'].shape[-2]
    elif 'T_season' in theta_dict:
        batch_shape = theta_dict['T_season'].shape[:-2]
        n_sites = theta_dict['T_season'].shape[-2]
    elif 'phi' in theta_dict:
        batch_shape = theta_dict['phi'].shape[:-2]
        n_sites = theta_dict['phi'].shape[-2]
    else:
        # Only global parameters
        batch_shape = theta_dict['beta_0'].shape[:-2]
        n_sites = 1  # Not used for global-only case

    flattened_parts = []

    # Process parameters in consistent order
    param_order = ['beta_0', 'alpha', 'sigma', 'A', 'T_season', 'phi']

    for param_name in param_order:
        if param_name not in sample_params or param_name not in theta_dict:
            continue

        if param_name in ['beta_0', 'alpha', 'sigma']:
            # Global parameters
            flattened_parts.append(
                theta_dict[param_name].reshape(batch_shape + (1,))
            )
        else:
            # Site-specific parameters
            flattened_parts.append(
                theta_dict[param_name].reshape(batch_shape + (n_sites,))
            )

    return jnp.concatenate(flattened_parts, axis=len(batch_shape))


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


def create_selective_flat_bijector(
    repr_theta: Dict[str, Array],
    bijector_specs: Dict[str, tfb.Bijector],
    n_sites: int,
    sample_params: list[str]
) -> tfb.Bijector:
    """
    Create blockwise bijector for selective parameter sampling.

    Args:
        repr_theta: Representative theta samples for Z-scaling
        bijector_specs: Bijector specifications for each parameter
        n_sites: Number of observation sites
        sample_params: List of parameter names to sample

    Returns:
        Blockwise bijector for sampled parameters only
    """
    individual_bijectors = []
    block_sizes = []

    # Process parameters in the same order as flattening
    param_order = ['beta_0', 'alpha', 'sigma', 'A', 'T_season', 'phi']

    for param in param_order:
        if param not in sample_params:
            continue

        base_bij = bijector_specs[param]

        if param in ['beta_0', 'alpha', 'sigma']:
            # Global parameters
            param_data = repr_theta[param].reshape(-1, 1)
            mean_val = jnp.mean(base_bij.forward(param_data))
            std_val = jnp.std(base_bij.forward(param_data))
            block_size = 1
        else:
            # Site-specific parameters
            param_data = repr_theta[param].reshape(-1, n_sites)
            mean_val = jnp.mean(base_bij.forward(param_data), axis=0)
            std_val = jnp.std(base_bij.forward(param_data), axis=0)
            block_size = n_sites

        z_scaled_bij = tfb.Chain([
            tfb.Scale(1.0 / jnp.maximum(std_val, 1e-8)),
            tfb.Shift(-mean_val),
            base_bij
        ])
        individual_bijectors.append(z_scaled_bij)
        block_sizes.append(block_size)

    # Create blockwise bijector
    return tfb.Blockwise(
        bijectors=individual_bijectors,
        block_sizes=block_sizes
    )


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


def reconstruct_selective_theta_dict(
    theta_flat: Array,
    sample_params: list[str],
    fixed_params: Dict[str, Array],
    n_sites: int
) -> Dict[str, Array]:
    """
    Reconstruct full theta dictionary from selective samples + fixed values.

    Args:
        theta_flat: Flattened array containing only sampled parameters
        sample_params: List of parameter names that were sampled
        fixed_params: Dictionary of fixed parameter values
        n_sites: Number of observation sites

    Returns:
        Full theta dictionary with sampled and fixed parameters
    """
    theta_dict = {}
    idx = 0

    # Process parameters in consistent order
    param_order = ['beta_0', 'alpha', 'sigma', 'A', 'T_season', 'phi']

    for param_name in param_order:
        if param_name in sample_params:
            # Extract sampled parameter from flattened array
            if param_name in ['beta_0', 'alpha', 'sigma']:
                # Global parameters
                theta_dict[param_name] = theta_flat[..., idx:idx+1, None]
                idx += 1
            else:
                # Site-specific parameters
                theta_dict[param_name] = theta_flat[..., idx:idx+n_sites, None]
                idx += n_sites
        else:
            # Use fixed parameter value
            if param_name in fixed_params:
                # Broadcast fixed value to match batch dimensions of sampled params
                fixed_val = fixed_params[param_name]
                if theta_flat.ndim > 1:
                    # Add batch dimensions to match theta_flat
                    batch_shape = theta_flat.shape[:-1]
                    if param_name in ['beta_0', 'alpha', 'sigma']:
                        theta_dict[param_name] = jnp.broadcast_to(
                            fixed_val,
                            batch_shape + (1, 1)
                        )
                    else:
                        theta_dict[param_name] = jnp.broadcast_to(
                            fixed_val,
                            batch_shape + (n_sites, 1)
                        )
                else:
                    theta_dict[param_name] = fixed_val

    return theta_dict


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


def get_standard_bijector_specs() -> Dict[str, tfb.Bijector]:
    """Get standard bijector specifications for SEIR parameters."""
    return {
        'beta_0': tfb.Invert(tfb.Sigmoid(low=0.1, high=2.0)),
        'alpha': tfb.Invert(tfb.Sigmoid(low=1/30, high=1/7)),
        'sigma': tfb.Invert(tfb.Sigmoid(low=1/21, high=1/7)),
        'A': tfb.Invert(tfb.Sigmoid(low=0.2, high=0.5)),
        'T_season': tfb.Invert(tfb.Softplus()),
        'phi': tfb.Invert(tfb.Sigmoid(low=0.0, high=jnp.pi)),
    }


def get_y_bijector_specs() -> Dict[str, tfb.Bijector]:
    """Get bijector specifications for observation data."""
    return {
        'obs': tfb.Invert(tfb.Softplus())  # Positive observations to unconstrained
    }


def create_pytree_bijectors(
    repr_theta: Dict[str, Array],
    repr_y: Dict[str, Array],
    theta_bijector_specs: Dict[str, tfb.Bijector],
    y_bijector_specs: Dict[str, tfb.Bijector]
) -> tuple[PyTreeBijector, PyTreeBijector]:
    """Create PyTreeBijectors for SFMPE with Z-scaling."""

    # Create Z-scaled bijector maps and PyTreeBijectors
    theta_bijector_map = create_zscaling_bijector_tree(repr_theta, repr_theta, theta_bijector_specs)
    sfmpe_theta_bijector = PyTreeBijector(theta_bijector_map, repr_theta)

    y_bijector_map = create_zscaling_bijector_tree(repr_y, repr_y, y_bijector_specs)
    sfmpe_y_bijector = PyTreeBijector(y_bijector_map, repr_y)

    return sfmpe_theta_bijector, sfmpe_y_bijector


def create_numpyro_seir_model(
    simulator_fn: Callable,
    n_sites: int,
    f_in: Dict[str, Array]
) -> Callable:
    """
    Create a NumPyro model for SEIR inference using native NumPyro distributions.

    Args:
        simulator_fn: Function that simulates SEIR dynamics
        n_sites: Number of observation sites
        f_in: Functional input data containing observation indices

    Returns:
        NumPyro model function compatible with NUTS/ESS kernels
    """
    def seir_model(y_observed: Dict[str, Array] = None):
        # Global parameters (shared across sites)
        beta_0 = numpyro.sample(
            'beta_0',
            dist.Uniform(jnp.array(0.1), jnp.array(2.0))
        )
        alpha = numpyro.sample(
            'alpha',
            dist.Uniform(jnp.array(1/30), jnp.array(1/7))
        )
        sigma = numpyro.sample(
            'sigma',
            dist.Uniform(jnp.array(1/21), jnp.array(1/7))
        )

        # Site-specific parameters
        # Seasonal amplitude
        A = numpyro.sample(
            'A',
            dist.Uniform(
                jnp.full((n_sites,), 0.2),
                jnp.full((n_sites,), 0.5)
            )
        )

        # Seasonal period (using Gamma distribution like TFP version)
        t_season_spread = 1./7.
        T_season = numpyro.sample(
            'T_season',
            dist.Gamma(
                jnp.full((n_sites,), 365.0 * t_season_spread),
                jnp.full((n_sites,), t_season_spread)
            )
        )

        # Seasonal phase
        phi = numpyro.sample(
            'phi',
            dist.Uniform(
                jnp.zeros((n_sites,)),
                jnp.full((n_sites,), jnp.pi)
            )
        )

        # Construct theta dictionary in the expected format
        theta = {
            'beta_0': beta_0[None, None],  # Shape: (1, 1)
            'alpha': alpha[None, None],    # Shape: (1, 1)
            'sigma': sigma[None, None],    # Shape: (1, 1)
            'A': A[:, None],               # Shape: (n_sites, 1)
            'T_season': T_season[:, None], # Shape: (n_sites, 1)
            'phi': phi[:, None]            # Shape: (n_sites, 1)
        }

        # Add batch dimension for simulator compatibility
        theta = tree.map(lambda x: x[None, ...], theta)

        # Simulate observations using the existing simulator
        y_pred = simulator_fn(numpyro.prng_key(), theta, f_in)

        # Extract predicted observations and ensure positive values
        obs_pred = jnp.maximum(y_pred['obs'][0], 0.1)  # Remove batch dim, ensure positive

        # Likelihood: Poisson observations
        numpyro.sample(
            'obs',
            dist.Independent(
                dist.Poisson(obs_pred),
                reinterpreted_batch_ndims=1
            ),
            obs=y_observed['obs'][0] if y_observed is not None else None
        )

    return seir_model


def create_selective_numpyro_seir_model(
    simulator_fn: Callable,
    n_sites: int,
    f_in: Dict[str, Array],
    sample_params: list[str],
    fixed_params: Dict[str, Array]
) -> Callable:
    """
    Create a NumPyro model that only samples specified parameters.

    Args:
        simulator_fn: Function that simulates SEIR dynamics
        n_sites: Number of observation sites
        f_in: Functional input data containing observation indices
        sample_params: List of parameter names to sample
        fixed_params: Dictionary of fixed parameter values

    Returns:
        NumPyro model function that samples only specified parameters
    """
    def selective_seir_model(y_observed: Dict[str, Array] = None):
        # Initialize theta dictionary with fixed values
        theta = {}

        # Add fixed parameters
        for param_name, param_value in fixed_params.items():
            theta[param_name] = param_value

        # Sample only specified parameters
        if 'beta_0' in sample_params:
            theta['beta_0'] = numpyro.sample(
                'beta_0',
                dist.Uniform(jnp.array(0.1), jnp.array(2.0))
            )[None, None]
        if 'alpha' in sample_params:
            theta['alpha'] = numpyro.sample(
                'alpha',
                dist.Uniform(jnp.array(1/30), jnp.array(1/7))
            )[None, None]
        if 'sigma' in sample_params:
            theta['sigma'] = numpyro.sample(
                'sigma',
                dist.Uniform(jnp.array(1/21), jnp.array(1/7))
            )[None, None]
        if 'A' in sample_params:
            theta['A'] = numpyro.sample(
                'A',
                dist.Uniform(
                    jnp.full((n_sites,), 0.2),
                    jnp.full((n_sites,), 0.5)
                )
            )[:, None]
        if 'T_season' in sample_params:
            t_season_spread = 1./7.
            theta['T_season'] = numpyro.sample(
                'T_season',
                dist.Gamma(
                    jnp.full((n_sites,), 365.0 * t_season_spread),
                    jnp.full((n_sites,), t_season_spread)
                )
            )[:, None]
        if 'phi' in sample_params:
            theta['phi'] = numpyro.sample(
                'phi',
                dist.Uniform(
                    jnp.zeros((n_sites,)),
                    jnp.full((n_sites,), jnp.pi)
                )
            )[:, None]

        # Add batch dimension for simulator compatibility
        theta = tree.map(lambda x: x[None, ...], theta)

        # Simulate observations using the existing simulator
        y_pred = simulator_fn(numpyro.prng_key(), theta, f_in)

        # Extract predicted observations and ensure positive values
        obs_pred = jnp.maximum(y_pred['obs'][0], 0.1)  # Remove batch dim

        # Likelihood: Poisson observations
        numpyro.sample(
            'obs',
            dist.Independent(
                dist.Poisson(obs_pred),
                reinterpreted_batch_ndims=1
            ),
            obs=y_observed['obs'][0] if y_observed is not None else None
        )

    return selective_seir_model
