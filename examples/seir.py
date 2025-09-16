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
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.bottom_up import train_bottom_up
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.cnf import CNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.mlp import MLPVectorField
from sfmpe.train_rounds import train_fmpe_rounds
from sfmpe.metrics.lc2st import (
    train_lc2st_classifiers,
    evaluate_lc2st,
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier
)
from seir_utils import (
    prior_fn, create_simulator_dist, create_simulator_fn, apply_dequantization,
    f_in_fn, f_in_fn_observed, _flatten, flatten_f_in,
    get_standard_bijector_specs, get_y_bijector_specs, create_pytree_bijectors,
    create_selective_prior_fn, create_selective_sfmpe_functions,
    flatten_selective_theta_dict, reconstruct_selective_theta_dict,
    create_selective_flat_bijector
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

    # Set up parameters for unified selective approach
    # Full inference is just selective inference with all parameters sampled
    all_params = ['beta_0', 'alpha', 'sigma', 'A', 'T_season', 'phi']

    if (hasattr(cfg, 'inference') and
        cfg.inference is not None and
        cfg.inference.sample_params is not None):
        sample_params = cfg.inference.sample_params
        logger.info(f"Using selective inference: sampling {sample_params}")
    else:
        sample_params = all_params
        logger.info("Using full inference: sampling all parameters")

    fixed_param_names = [p for p in all_params if p not in sample_params]
    if fixed_param_names:
        logger.info(f"Fixed parameters: {fixed_param_names}")
    else:
        logger.info("No fixed parameters")
    
    key = jr.PRNGKey(cfg.seed)
    
    # Create functions
    simulator_dist = create_simulator_dist(n_timesteps, cfg.dt, cfg.population, cfg.I0_prop, n_warmup)
    simulator_fn = create_simulator_fn(simulator_dist)
    
    # Generate ground truth and observations
    theta_key, obs_key, f_in_key, key = jr.split(key, 4)

    theta_truth = prior_fn(n_sites).sample((1,), seed=theta_key)

    # For selective inference, broadcast fixed local parameters to be identical across sites
    for param_name in fixed_param_names:
        if param_name in ['A', 'T_season', 'phi']:
            # Take the first site's value and broadcast it across all sites
            single_value = theta_truth[param_name][0, 0:1]  # Shape: (1, 1)
            broadcasted_value = jnp.broadcast_to(single_value, (n_sites, 1))
            theta_truth[param_name] = theta_truth[param_name].at[0].set(broadcasted_value)

    f_in = f_in_fn(n_obs, n_sites, n_timesteps).sample((1,), seed=f_in_key)
    y_observed = simulator_fn(obs_key, theta_truth, f_in)

    # Extract fixed parameter values
    fixed_params = {}
    for param_name in fixed_param_names:
        fixed_params[param_name] = theta_truth[param_name][0]  # Remove batch dim
    if fixed_params:
        logger.info(f"Fixed parameter values: {tree.map(lambda x: jnp.squeeze(x), fixed_params)}")
    
    # Apply dequantization  
    deq_key, key = jr.split(key)
    y_processed = apply_dequantization(y_observed, deq_key)
    
    # Generate representative data for consistent Z-scaling across all bijectors
    repr_key, key = jr.split(key)

    # Always use selective functions for representative data generation
    selective_prior_fn, selective_local_fn, wrapped_simulator_fn, global_names, local_names = create_selective_sfmpe_functions(
        n_sites, sample_params, fixed_params, simulator_fn
    )
    # Sample only the selected parameters for representative data
    repr_theta = selective_prior_fn(n_sites).sample((1000,), seed=repr_key)

    # For representative data, use the same f_in for all samples
    repr_f_in = tree.map(lambda leaf: jnp.repeat(leaf, 1000, axis=0), f_in)
    repr_y_raw = wrapped_simulator_fn(repr_key, repr_theta, repr_f_in)
    repr_y = apply_dequantization(repr_y_raw, deq_key)
    
    # Create Z-scaled bijector maps and PyTreeBijectors
    # Filter bijector specs to only include sampled parameters
    theta_bijector_specs = get_standard_bijector_specs()
    selective_theta_specs = {k: v for k, v in theta_bijector_specs.items() if k in sample_params}
    y_bijector_specs = get_y_bijector_specs()
    sfmpe_theta_bijector, sfmpe_y_bijector = create_pytree_bijectors(repr_theta, repr_y, selective_theta_specs, y_bijector_specs)
    
    # Transform observations to unconstrained space
    y_unconstrained = sfmpe_y_bijector.forward(y_processed)
    
    # Create wrapped functions for train_bottom_up
    def wrapped_prior_fn(n):
        """Prior function that returns TransformedDistribution."""
        base_prior = selective_prior_fn(n)
        return tfd.TransformedDistribution(
            base_prior,
            sfmpe_theta_bijector,
            name="transformed_prior"
        )

    def wrapped_p_local(g, n):
        """Local prior function that returns TransformedDistribution."""
        base_local = selective_local_fn(g, n)
        return tfd.TransformedDistribution(
            base_local,
            sfmpe_theta_bijector,
            name="transformed_local"
        )
    
    def wrapped_simulator_fn_for_training(seed, theta, f_in_sample):
        """Simulator function that handles bijector transformations."""
        # Transform parameters back to constrained space
        theta_constrained = sfmpe_theta_bijector.inverse(theta)

        # Apply wrapped simulator (handles parameter reconstruction if needed)
        y_constrained = wrapped_simulator_fn(seed, theta_constrained, f_in_sample)
        y_deq = apply_dequantization(y_constrained, seed)

        # Transform outputs to unconstrained space
        return sfmpe_y_bijector.forward(y_deq)
    
    logger.info("Generated ground truth and processed observations")
    logger.info(f"Observation times shape: {f_in['obs'].shape}")
    logger.info(f"Truth incidence shape: {y_observed['obs'].shape}")
    logger.info(f"Processed incidence range: {jnp.min(y_processed['obs'])} to {jnp.max(y_processed['obs'])}")
    logger.info(f"Transformed incidence range: {jnp.min(y_unconstrained['obs'])} to {jnp.max(y_unconstrained['obs'])}")
    
    # Independence structure for structured inference (dynamic based on sampled parameters)
    local_independence = ['obs'] + local_names
    cross_local_connections = [(param, 'obs', (0, 0)) for param in local_names]
    independence = {
        'local': local_independence,  # Observations independent across time/sites
        'cross': [],
        'cross_local': cross_local_connections
    }

    # SFMPE Neural Network Setup (dynamic n_labels)
    rngs = nnx.Rngs(key)
    n_labels = len(global_names) + len(local_names) + 1  # sampled parameters + obs
    logger.info(f"Using {n_labels} labels: {len(global_names)} global + {len(local_names)} local + 1 obs")

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
        n_labels=n_labels,
        index_dim=1,  # Temporal indexing
        rngs=rngs
    )

    model = StructuredCNF(nn, rngs=rngs)
    estim = SFMPE(model, rngs=rngs)

    # Training
    train_key, key = jr.split(key)
    logger.info("Starting SFMPE bottom-up training")

    # Set up f_in function arguments based on configuration
    if cfg.f_in_sample == 'observed':
        f_in_fn_train = f_in_fn_observed
        f_in_args = (n_obs, 1, f_in)
        f_in_args_global = (n_obs, n_sites, f_in)
    elif cfg.f_in_sample == 'prior':
        f_in_fn_train = f_in_fn
        f_in_args = (n_obs, 1, n_timesteps)
        f_in_args_global = (n_obs, n_sites, n_timesteps)
    else:
        raise ValueError(f"Invalid f_in_sample: {cfg.f_in_sample}")

    start_time = time.time()
    labels, slices, masks = train_bottom_up(
        train_key,
        estim,
        wrapped_prior_fn,
        wrapped_p_local,
        wrapped_simulator_fn_for_training,  # Use training-specific wrapped simulator
        global_names,  # Dynamic global parameters
        local_names,   # Dynamic local parameters
        n_sites,
        n_rounds,
        n_simulations,
        n_epochs,
        y_unconstrained,  # Use unconstrained data
        independence,
        optimiser=optax.adam(cfg.training.learning_rate),
        batch_size=int(n_simulations * cfg.training.batch_size_fraction),
        f_in=f_in_fn_train,
        f_in_args=f_in_args,
        f_in_args_global=f_in_args_global,
        f_in_target=f_in
    )
    logger.info(f"SFMPE bottom-up training completed in {time.time() - start_time:.2f} seconds")

    # Sample SFMPE posterior
    logger.info("Sampling SFMPE posterior")
    start_time = time.time()

    # Create flattened f_in index for posterior sampling
    f_in_flattened = flatten_f_in(f_in, sample_params=sample_params)
    posterior = estim.sample_posterior(
        _flatten(y_processed)[..., None],
        labels,
        slices,
        masks=masks,
        n_samples=n_post_samples,
        index=f_in_flattened
    )

    # Transform posterior samples back into constrained space
    posterior = sfmpe_theta_bijector.inverse(posterior)

    logger.info(f'SFMPE posterior mean: {jnp.mean(_flatten(posterior), axis=0)}')
    logger.info(f"SFMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")

    # Create FMPE bijector
    fmpe_theta_bijector = create_selective_flat_bijector(repr_theta, theta_bijector_specs, n_sites, sample_params)
    
    logger.info('Starting FMPE training')
    
    # Create proxy functions for FMPE training
    def fmpe_prior_fn(key: Array, n_samples: int) -> Array:
        """Prior function compatible with FMPE interface"""
        selective_prior = create_selective_prior_fn(n_sites, sample_params, fixed_params)
        theta_samples = selective_prior(n_sites).sample((n_samples,), seed=key)
        # Flatten and transform to unconstrained space
        theta_flat = flatten_selective_theta_dict(theta_samples, sample_params)
        return fmpe_theta_bijector.forward(theta_flat)

    def fmpe_simulator_fn(key: Array, theta_flat: Array) -> Array:
        """Simulator function compatible with FMPE interface"""
        # Inverse transform to constrained space
        theta_constrained_flat = fmpe_theta_bijector.inverse(theta_flat)

        # Reconstruct structured theta from flat representation
        theta_dict = reconstruct_selective_theta_dict(theta_constrained_flat, sample_params, fixed_params, n_sites)

        # Run simulator
        n_simulations = theta_flat.shape[0]
        f_in_matched = tree.map(
            lambda leaf: jnp.repeat(leaf, n_simulations, axis=0),
            f_in
        )
        y_samples = simulator_fn(key, theta_dict, f_in_matched)
        y_deq = apply_dequantization(y_samples, key)

        # Transform observations to unconstrained space (consistent with SFMPE)
        y_transformed = sfmpe_y_bijector.forward(y_deq)
        return y_transformed['obs'].reshape((theta_flat.shape[0], -1))

    # Create FMPE model
    # Calculate parameter dimensions based on sampled parameters
    n_global_params = len([p for p in sample_params if p in ['beta_0', 'alpha', 'sigma']])
    n_local_params = len([p for p in sample_params if p in ['A', 'T_season', 'phi']])
    total_params = n_global_params + n_local_params * n_sites

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
    
    # Flatten observed data for FMPE (use transformed observations)
    fmpe_y_observed = y_unconstrained['obs'].reshape(-1)
    
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

    logger.info(f'FMPE posterior mean: {jnp.mean(fmpe_posterior_samples, axis=0)}')
    logger.info(f"FMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")
    
    # LC2ST Analysis
    n_cal = cfg.analysis.n_cal
    logger.info(f"Creating calibration dataset with {n_cal} samples")
    start_time = time.time()

    def sample_single_sfmpe_posterior(key, x):
        # Use the correct dimensions for selective inference
        n_selective_params = len(global_names) + len(local_names) * n_sites
        theta_0 = jr.normal(key, (1, n_selective_params, 1))
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
        lambda: wrapped_prior_fn(n_sites),  # Use wrapped (transformed) function
        lambda seed, theta: wrapped_simulator_fn_for_training(seed, theta, cal_f_in),  # Use training-specific wrapped simulator
        y_unconstrained['obs'].reshape(-1), # Use transformed observations
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
        fmpe_prior_fn,        # Already transformed
        fmpe_simulator_fn,    # Already transformed
        fmpe_y_observed,      # Use same transformed observations as training
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
