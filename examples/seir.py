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
from typing import Tuple, Callable, Dict
from jaxtyping import PyTree, Array
from flax import nnx
import optax
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from jax import numpy as jnp, random as jr, tree, vmap
from jax.experimental.ode import odeint
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.bottom_up import train_bottom_up
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.cnf import CNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.mlp import MLPVectorField
from sfmpe.train_rounds import train_fmpe_rounds
from sfmpe.pytree_bijector import (
    PyTreeBijector, 
    create_bijector_from_distribution, 
    create_manual_bijector_tree
)
from tensorflow_probability.substrates.jax import bijectors as tfb
from sfmpe.metrics.lc2st import (
    train_lc2st_classifiers,
    evaluate_lc2st,
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier
)

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
    
    # Derivatives
    dS = -s_to_e
    dE = s_to_e - e_to_i
    dI = e_to_i - i_to_r
    dR = i_to_r
    
    return jnp.array([dS, dE, dI, dR])


def p_local(g, n):
    """Local prior distribution for site-specific parameters (independent of global)."""
    n_sims = g['beta_0'].shape[0]
    return tfd.JointDistributionNamed(
        dict(
            # Site-specific seasonal parameters  
            A = tfd.Uniform(jnp.zeros((n_sims, n, 1)), jnp.ones((n, 1))),
            T_season = tfd.Normal(
                jnp.full((n_sims, n, 1), 365.0), 
                jnp.full((n_sims, n, 1), 50.0)
            ),
            phi = tfd.Uniform(
                jnp.zeros((n_sims, n, 1)), 
                jnp.full((n_sims, n, 1), 2*jnp.pi)
            )
        ),
        batch_ndims=1,
    )


def prior_fn(n):
    """Global prior distribution."""
    return tfd.JointDistributionNamed(
        dict(
            # Global parameters (independent of obs by exchangeability)
            beta_0 = tfd.Uniform(jnp.full((1, 1), 0.1), jnp.full((1, 1), 2.0)),
            alpha = tfd.Uniform(jnp.full((1, 1), 1/30), jnp.full((1, 1), 1/7)),
            sigma = tfd.Uniform(jnp.full((1, 1), 1/21), jnp.full((1, 1), 1/7)),
            
            # Local parameters are independent of global parameters
            A = tfd.Uniform(jnp.zeros((n, 1)), jnp.ones((n, 1))),
            T_season = tfd.Normal(
                jnp.full((n, 1), 365.0), 
                jnp.full((n, 1), 50.0)
            ),
            phi = tfd.Uniform(
                jnp.zeros((n, 1)), 
                jnp.full((n, 1), 2*jnp.pi)
            )
        ),
        batch_ndims=1,
    )


def create_simulator_fn(
    n_timesteps: int,
    dt: float = 1.0,
    population: int = 10000,
    I0_prop: float = 0.001
) -> Callable:
    """Create simulator function for SVEIR dynamics."""
    
    def simulator_fn(key: Array, theta: Dict[str, Array], f_in: dict) -> Dict[str, Array]:
        """
        Simulate SVEIR dynamics and return indexed observations.
        
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
            # Keep as JAX arrays - don't convert to Python floats
            params_dict = {
                # Global parameters  
                'beta_0': params_single['beta_0'][0, 0],
                'alpha': params_single['alpha'][0, 0],
                'sigma': params_single['sigma'][0, 0],
                # Site-specific parameters
                'A': params_single['A'][site_idx, 0],
                'T_season': params_single['T_season'][site_idx, 0],
                'phi': params_single['phi'][site_idx, 0]
            }
            
            def ode_func(state, t):
                return seir_dynamics(state, t, params_dict)
            
            # Solve ODE at observation times for this site
            # Sort times for ODE solver, then reorder results
            t_eval = obs_times_single[:, 0]  # Extract times (n_obs,)
            sort_indices = jnp.argsort(t_eval)
            t_eval_sorted = t_eval[sort_indices]
            solution = odeint(ode_func, initial_state, t_eval_sorted)
            
            # Reorder solution to match original time sequence
            reorder_indices = jnp.argsort(sort_indices)
            solution_reordered = solution[reorder_indices]
            
            # Extract incidence - since we're solving at the observation times directly,
            # we return the infection rate (new infections per day) at those times
            # For incidence, we use the infection rate α*E at observation times  
            exposed = solution_reordered[:, 1]  # E compartment (index 1 in SEIR)
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
        
        # Add Poisson noise to incidence
        key_poisson = jr.split(key, batch_size * n_sites)
        key_poisson = key_poisson.reshape((batch_size, n_sites, 2))
        
        obs = vmap(vmap(
            lambda k, rate: tfd.Poisson(jnp.maximum(rate, 0.1)).sample(seed=k)
        ))(key_poisson, incidence_batch)
        
        # Add final dimension to match expected shape (batch_size, n_sites, n_obs, 1)
        obs = obs[..., None]
        
        return {'obs': obs}
    
    return simulator_fn


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
            
            # Functional observation times - Exponential(1) distributed per site
            obs = tfd.Exponential(
                jnp.ones((n_sites, n_obs, 1))  # Exponential(1) for observation times
            )
        ),
        batch_ndims=1
    )


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


def create_prior_bijector(n_sites: int) -> PyTreeBijector:
    """Create bijector for transforming prior samples to unconstrained space."""
    # Create bijector from distribution default bijectors
    return create_bijector_from_distribution(prior_fn(n_sites))


def create_local_bijector(n_sites: int) -> PyTreeBijector:
    """Create bijector for transforming local prior samples to unconstrained space."""
    # Create bijector from p_local distribution default bijectors
    dummy_global = { 'beta_0': jnp.zeros((1, 1, 1)) }  # p_local doesn't depend on global parameters in our case
    return create_bijector_from_distribution(p_local(dummy_global, n_sites))


def create_simulator_bijector(obs_sample: Dict[str, Array]) -> PyTreeBijector:
    """Create bijector for transforming simulator outputs to unconstrained space."""
    # For incidence data (positive values), use Softplus transformation
    bijector_specs = {
        'obs': tfb.Softplus()  # Maps R -> R+ for positive incidence values
    }
    
    return create_manual_bijector_tree(obs_sample, bijector_specs)


def run(cfg: DictConfig) -> None:
    """Main execution function."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running SEIR with n_simulations={cfg.n_simulations}, "
                f"n_rounds={cfg.n_rounds}, n_epochs={cfg.n_epochs}")
    
    # Extract parameters
    n_timesteps = cfg.n_timesteps
    n_obs = cfg.n_obs
    n_sites = cfg.n_sites
    n_simulations = cfg.n_simulations
    n_rounds = cfg.n_rounds
    n_epochs = cfg.n_epochs
    n_post_samples = cfg.n_post_samples
    
    # Independence structure for structured inference
    independence = {
        'local': ['obs'],  # Observations independent across time/sites
        'cross': [
            ('beta_0', 'obs'),  # Global parameters independent by exchangeability
            ('alpha', 'obs'),
            ('sigma', 'obs')
        ],
        'cross_local': [  # Site-specific parameters connect to their observations
            ('A', 'obs', (0, 0)),
            ('T_season', 'obs', (0, 0)),
            ('phi', 'obs', (0, 0))
        ]
    }
    
    key = jr.PRNGKey(cfg.seed)
    
    # Create functions
    simulator_fn = create_simulator_fn(n_timesteps, cfg.dt, cfg.population, cfg.I0_prop)
    
    # Generate ground truth and observations
    theta_key, obs_key, f_in_key, key = jr.split(key, 4)
    
    theta_truth = prior_fn(n_sites).sample((1,), seed=theta_key)
    f_in = f_in_fn(n_obs, n_sites).sample((1,), seed=f_in_key)
    y_observed = simulator_fn(obs_key, theta_truth, f_in)
    
    # Apply dequantization  
    deq_key, key = jr.split(key)
    y_processed = apply_dequantization(y_observed, deq_key)
    
    # Transform prior samples and observations to unconstrained space
    simulator_bijector = create_simulator_bijector(y_processed)
    y_unconstrained = simulator_bijector.forward(y_processed)
    
    # Create wrapped functions for train_bottom_up
    def wrapped_prior_fn(n):
        """Prior function that returns TransformedDistribution."""
        return prior_fn(n)
        base_prior = prior_fn(n)
        bijector = create_prior_bijector(n)
        return tfd.TransformedDistribution(
            base_prior, 
            bijector,
            name="transformed_prior"
        )
    
    def wrapped_p_local(g, n):
        """Local prior function that returns TransformedDistribution."""
        return p_local(g, n)
        base_local = p_local(g, n)
        bijector = create_local_bijector(n)
        return tfd.TransformedDistribution(
            base_local,
            bijector, 
            name="transformed_local"
        )
    
    def wrapped_simulator_fn(seed, theta, f_in_sample):
        """Simulator function that handles bijector transformations."""
        output = simulator_fn(seed, theta, f_in_sample)
        return apply_dequantization(output, seed)
        # Transform parameters back to constrained space
        n_sites = theta['A'].shape[-2]
        prior_bijector = create_prior_bijector(n_sites)
        theta_constrained = prior_bijector.inverse(theta)
        
        # Apply original simulator
        y_constrained = simulator_fn(seed, theta_constrained, f_in_sample)
        simulator_bijector = create_simulator_bijector(y_constrained)
        
        # Transform outputs to unconstrained space
        y_unconstrained_sim = simulator_bijector.forward(y_constrained)
        
        return y_unconstrained_sim
    
    logger.info("Generated ground truth and processed observations")
    logger.info(f"Observation times shape: {f_in['obs'].shape}")
    logger.info(f"Truth incidence shape: {y_observed['obs'].shape}")
    logger.info(f"Processed incidence range: {jnp.min(y_processed['obs'])} to {jnp.max(y_processed['obs'])}")
    
    # SFMPE Implementation
    rngs = nnx.Rngs(key)
    transformer_config = {
        'latent_dim': cfg.sfmpe.transformer.latent_dim,
        'label_dim': cfg.sfmpe.transformer.label_dim,
        'index_out_dim': cfg.sfmpe.transformer.index_out_dim,
        'n_encoder': cfg.sfmpe.transformer.n_encoder,
        'n_decoder': cfg.sfmpe.transformer.n_decoder,
        'n_heads': cfg.sfmpe.transformer.n_heads,
        'n_ff': cfg.sfmpe.transformer.n_ff,
        'dropout': cfg.sfmpe.transformer.dropout,
        'activation': nnx.relu,
    }

    nn = Transformer(
        transformer_config,
        value_dim=1,
        n_labels=7,  # 3 global + 3 site-specific parameters + obs
        index_dim=1,  # Temporal indexing
        rngs=rngs
    )

    model = StructuredCNF(nn, rngs=rngs)
    estim = SFMPE(model, rngs=rngs)

    train_key, key = jr.split(key)
    logger.info("Starting SFMPE bottom-up training")
    start_time = time.time()
    
    labels, slices, masks = train_bottom_up(
        train_key,
        estim,
        wrapped_prior_fn,
        wrapped_p_local,
        wrapped_simulator_fn,
        ['beta_0', 'alpha', 'sigma'],  # Global parameters
        ['A', 'T_season', 'phi'],  # Local parameters
        n_sites,
        n_rounds,
        n_simulations,
        n_epochs,
        y_unconstrained,  # Use unconstrained data
        independence,
        optimiser=optax.adam(cfg.training.learning_rate),
        batch_size=int(n_simulations * cfg.training.batch_size_fraction),
        f_in=f_in_fn,
        f_in_args=(n_obs, 1),
        f_in_args_global=(n_obs, n_sites),
        f_in_target=f_in
    )
    logger.info(f"SFMPE bottom-up training completed in {time.time() - start_time:.2f} seconds")

    logger.info("Sampling SFMPE posterior")
    start_time = time.time()
    # Create flattened f_in index for posterior sampling
    f_in_flattened = flatten_f_in(f_in)
    posterior = estim.sample_posterior(
        _flatten(y_processed)[..., None],
        labels, #type:ignore
        slices,
        masks=masks,
        n_samples=n_post_samples,
        index=f_in_flattened
    )

    print('posterior shapes:')
    print(tree.map(lambda x: x.shape, posterior))
    print(tree.map(lambda x: x[:10], posterior))
    logger.info(f"SFMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")

    logger.info('Starting FMPE training')
    
    # Create proxy functions for FMPE training
    def fmpe_prior_fn(key: Array, n_samples: int) -> Array:
        """Prior function compatible with FMPE interface"""
        theta_samples = prior_fn(n_sites).sample((n_samples,), seed=key)
        # Flatten: combine all parameters
        flattened_parts = []
        for param_name in ['beta_0', 'alpha', 'sigma']:
            flattened_parts.append(theta_samples[param_name].reshape((n_samples, 1)))
        for param_name in ['A', 'T_season', 'phi']:
            flattened_parts.append(theta_samples[param_name].reshape((n_samples, n_sites)))
        return jnp.concatenate(flattened_parts, axis=1)

    def fmpe_simulator_fn(key: Array, theta_flat: Array) -> Array:
        """Simulator function compatible with FMPE interface"""
        # Reconstruct structured theta from flat representation
        theta_dict = {}
        idx = 0
        
        # Global parameters (3 parameters, 1 each)
        theta_dict['beta_0'] = theta_flat[:, idx:idx+1, None]  # Add singleton for batch
        idx += 1
        theta_dict['alpha'] = theta_flat[:, idx:idx+1, None]
        idx += 1
        theta_dict['sigma'] = theta_flat[:, idx:idx+1, None]
        idx += 1
        
        # Site-specific parameters (3 parameters, n_sites each for SEIR)
        for param_name in ['A', 'T_season', 'phi']:
            theta_dict[param_name] = theta_flat[:, idx:idx+n_sites, None]
            idx += n_sites
        
        # Run simulator
        n_simulations = theta_flat.shape[0]
        f_in_matched = tree.map(
            lambda leaf: jnp.repeat(leaf, n_simulations, axis=0),
            f_in
        )
        y_samples = simulator_fn(key, theta_dict, f_in_matched)
        # Apply same dequantization as observations
        y_deq = apply_dequantization(y_samples, key)
        return y_deq['obs'].reshape((theta_flat.shape[0], -1))

    # Create FMPE model
    total_params = 3 + 3 * n_sites  # 3 global + 3 * n_sites local parameters (SEIR model)
    fmpe_nn = MLPVectorField(
        theta_dim=total_params,
        context_dim=n_obs * n_sites,  # flattened observations
        latent_dim=cfg.fmpe.nn.latent_dim,
        n_layers=cfg.fmpe.nn.n_layers,
        dropout=cfg.fmpe.nn.dropout,
        activation=nnx.relu
    )
    
    fmpe_model = CNF(fmpe_nn)
    fmpe_estim = FMPE(fmpe_model, rngs=rngs)
    
    # Flatten observed data for FMPE
    fmpe_y_observed = y_processed['obs'].reshape(-1)
    
    # Train using round-based approach
    train_key, key = jr.split(key)
    logger.info("Starting FMPE round-based training")
    start_time = time.time()
    
    def sim_corrected_fmpe_prior_fn(key, n_samples):
        n = n_samples // n_sites
        return fmpe_prior_fn(key, n)
    
    fmpe_estim = train_fmpe_rounds(
        train_key,
        fmpe_estim,
        sim_corrected_fmpe_prior_fn,
        fmpe_simulator_fn,
        fmpe_y_observed,
        theta_shape=(total_params,),
        n_rounds=n_rounds,
        n_simulations=n_simulations,
        n_epochs=n_epochs,
        optimizer=optax.adam(cfg.training.learning_rate),
        batch_size=int((n_simulations // n_sites) * cfg.training.batch_size_fraction)
    )

    logger.info(f"FMPE round-based training completed in {time.time() - start_time:.2f} seconds")

    # Sample from FMPE posterior
    logger.info("Sampling FMPE posterior")
    start_time = time.time()
    fmpe_posterior_samples = fmpe_estim.sample_posterior(
        fmpe_y_observed[None, ...],
        theta_shape=(total_params,),
        n_samples=n_post_samples
    )

    logger.info(f'Ground truth shape: {fmpe_y_observed.shape}')
    logger.info(f'FMPE posterior mean: {jnp.mean(fmpe_posterior_samples, axis=0)}')
    logger.info(f"FMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")
    
    # LC2ST Analysis
    n_cal = cfg.analysis.n_cal
    logger.info(f"Creating calibration dataset with {n_cal} samples")
    start_time = time.time()

    def sample_single_sfmpe_posterior(key, x):
        theta_0 = jr.normal(key, (1, total_params, 1))
        x = tree.map(lambda leaf: leaf[None, ...], x)
        context = _flatten(x)[..., None]
        posterior = estim.sample_posterior(
            context,
            labels, #type:ignore
            slices,
            masks=masks,
            n_samples=1,
            theta_0 = theta_0,
            index=f_in_flattened
        )
        return _flatten(posterior)[...,0,:]

    def sample_single_fmpe_posterior(key, x):
        dim = total_params
        theta_0 = jr.normal(key, (1, dim))
        return fmpe_estim.sample_posterior(
            x[None, ...],
            theta_shape = (dim,),
            n_samples=1,
            theta_0 = theta_0
        ).reshape((dim,))
    
    n_cal_epochs = cfg.analysis.n_cal_epochs
    analyse_key, key = jr.split(key)
    
    # Use Hydra's output directory
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.runtime.output_dir)

    # Preprocess posterior samples for SFMPE
    sfmpe_posterior_flat = _flatten(posterior)
    
    logger.info("Starting C2ST-NF analysis for SFMPE")
    start_time = time.time()
    cal_f_in = tree.map(lambda leaf: jnp.repeat(leaf, n_cal, axis=0), f_in)
    null_stats, main_stat, p_value = apply_lc2st(
        analyse_key,
        create_sfmpe_calibration_dataset,
        sample_single_sfmpe_posterior,
        lambda: prior_fn(n_sites),
        lambda seed, theta: apply_dequantization(
            simulator_fn(seed, theta, cal_f_in),
            seed
        ),
        y_processed['obs'].reshape(-1),
        sfmpe_posterior_flat,
        n_cal_epochs,
        n_cal,
        cfg.analysis.classifier.n_layers,
        cfg.analysis.n_null,
        cfg.analysis.classifier.latent_dim,
        rngs
    )
    save_lc2st_results(null_stats, main_stat, p_value, out_dir/'sfmpe')
    logger.info(f"SFMPE C2ST-NF analysis completed in {time.time() - start_time:.2f} seconds")
    
    logger.info("Starting C2ST-NF analysis for FMPE")
    start_time = time.time()
    null_stats, main_stat, p_value = apply_lc2st(
        analyse_key,
        create_fmpe_calibration_dataset,
        sample_single_fmpe_posterior,
        fmpe_prior_fn,
        fmpe_simulator_fn,
        y_processed['obs'].reshape(-1),
        fmpe_posterior_samples,
        n_cal_epochs,
        n_cal,
        cfg.analysis.classifier.n_layers,
        cfg.analysis.n_null,
        cfg.analysis.classifier.latent_dim,
        rngs
    )
    save_lc2st_results(null_stats, main_stat, p_value, out_dir/'fmpe')
    logger.info(f"FMPE C2ST-NF analysis completed in {time.time() - start_time:.2f} seconds")
    
    logger.info("SEIR experiment completed successfully!")


def apply_lc2st(
    key: jnp.ndarray,
    create_calibration_dataset: Callable,
    sample_single_posterior: Callable[[Array, Array], Array],
    prior_fn: Callable,
    simulator_fn: Callable,
    observation: jnp.ndarray,
    posterior_samples: jnp.ndarray,
    n_epochs: int,
    n_cal: int,
    n_layers: int,
    n_null: int,
    latent_dim: int,
    rngs: nnx.Rngs
    ):
    """Apply l-c2st pipeline"""
    cal_key, key = jr.split(key)
    d = create_calibration_dataset(
        cal_key,
        sample_single_posterior,
        prior_fn,
        simulator_fn,
        n_cal
    )

    theta_cal, x_cal = d[0], d[1]
    classifier_dim = theta_cal.shape[-1] + x_cal.shape[-1]
    
    # Train main classifier
    train_key, key = jr.split(key)
    activation = nnx.relu

    main = BinaryMLPClassifier(
        dim=classifier_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs
    )

    null_cls = MultiBinaryMLPClassifier(
        dim=classifier_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        n=n_null,
        rngs=rngs
    )
    logger = logging.getLogger(__name__)
    logger.info(f'Training classifiers with {n_epochs} epochs')
    train_lc2st_classifiers(
        train_key,
        d,
        main,
        null_cls,
        n_epochs
    )
    
    null_stats, main_stat, p_value = evaluate_lc2st(
        observation,
        posterior_samples,
        main,
        null_cls
    )

    logger.info(f'null_stats: {null_stats}')
    logger.info(f'main_stat: {main_stat}')
    logger.info(f'p-value: {p_value}')
    return null_stats, main_stat, p_value


def save_lc2st_results(
    null_stats: jnp.ndarray,
    main_stat: jnp.ndarray,
    p_value: jnp.ndarray,
    out_dir: Path
    ):
    """Save LC2ST results and create plots."""
    # Create output directory and make quant plot
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write out stats json with serialize
    stats = {
        'main_stat': float(main_stat),
        'null_stats': null_stats.tolist(),
        'p_value': float(p_value),
        'reject': bool(p_value < 0.05)
    }
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f)


def create_sfmpe_calibration_dataset(
    key: jnp.ndarray,
    sample_posterior: Callable[[Array, PyTree], PyTree],
    prior_fn: Callable[[], tfd.Distribution],
    simulator_fn: Callable[[Array, PyTree], PyTree],
    n: int
    ) -> Tuple[Array, Array, Array]:
    """Create calibration dataset for SFMPE."""
    prior_key, post_key, sim_key = jr.split(key, 3)
    prior = prior_fn().sample((n,), seed=prior_key)
    y = simulator_fn(sim_key, prior)
    post_estimate = vmap(
        sample_posterior,
        in_axes=[0, tree.map(lambda _: 0, y)]
    )(jr.split(post_key, n), y)
    x = _flatten(prior)
    y = _flatten(y)
    return y, x, post_estimate


def create_fmpe_calibration_dataset(
    key: jnp.ndarray,
    sample_posterior: Callable[[Array, Array], Array],
    prior_fn: Callable[[Array, int], PyTree],
    simulator_fn: Callable[[Array, Array], PyTree],
    n: int
    ) -> Tuple[Array, Array, Array]:
    """Create calibration dataset for FMPE."""
    prior_key, post_key, sim_key = jr.split(key, 3)
    prior = prior_fn(prior_key, n)
    y = simulator_fn(sim_key, prior)
    post_estimate = vmap(sample_posterior)(jr.split(post_key, n), y)
    return y, prior, post_estimate


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

@hydra.main(version_base=None, config_path="conf", config_name="seir_config")
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
