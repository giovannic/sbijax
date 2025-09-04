import json
import logging
import time
from pathlib import Path
from typing import Tuple, Callable
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
from sfmpe.metrics.lc2stnf import lc2st_quant_plot

def run(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Running with n_simulations={cfg.n_simulations}, n_rounds={cfg.n_rounds}, n_epochs={cfg.n_epochs}")
    
    # Extract parameters from config
    n_theta = cfg.n_theta
    n_obs = cfg.n_obs
    n_simulations = cfg.n_simulations
    n_rounds = cfg.n_rounds
    n_epochs = cfg.n_epochs
    n_post_samples = cfg.n_post_samples
    var_mu = cfg.var_mu
    var_theta = cfg.var_theta
    obs_rate = cfg.obs_rate

    independence = {
        'local': ['obs', 'theta'],  # observations and theta are independent of each other
        'cross': [('mu', 'obs'), ('obs', 'mu')],  # mu is independent of observations
        'cross_local': [('theta', 'obs', (0, 0))]  # theta[i] connects to y[i]
    }

    # make prior distribution
    def p_local(g, n):
        return tfd.JointDistributionNamed(
            dict(
                theta = tfd.Independent(
                    tfd.Normal(
                        jnp.repeat(g['mu'], n, axis=-2),
                        var_theta
                    ),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=1,
        )

    def prior_fn(n):
        return tfd.JointDistributionNamed(
            dict(
                mu = tfd.Normal(jnp.zeros((1, 1)), jnp.full((1, 1), var_mu)), #type:ignore
                theta = lambda mu: tfd.Independent(
                    tfd.Normal(
                        jnp.repeat(mu, n, axis=-2),
                        var_theta
                    ),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=1,
        )

    # make simulator function
    def simulator_fn(seed, theta, f_in: dict):
        # theta['theta'] has shape (n_simulations, n_theta, 1)
        # f_in['obs'] has shape (n_simulations, n_theta, n_obs, 1)
        # We want n_theta means each with n_obs different variances
        # Expand theta to (n_simulations, n_theta, n_obs, 1)
        theta_expanded = jnp.expand_dims(theta['theta'], -2)  # (n_simulations, n_theta, 1, 1)
        theta_expanded = jnp.broadcast_to(theta_expanded, (*theta_expanded.shape[:-2], n_obs, 1))  # (n_simulations, n_theta, n_obs, 1)
        
        obs = tfd.Independent(
            tfd.Normal(theta_expanded, f_in['obs']),
            reinterpreted_batch_ndims=2
        ).sample(seed=seed)
        return {
            'obs': obs  # Shape: (n_simulations, n_theta, n_obs, 1)
        }

    # make function input sampler
    def f_in_fn(n_obs, n_theta):
        return tfd.JointDistributionNamed(
            dict(
                mu = tfd.Deterministic(jnp.zeros((1, 1))), # dummy f_in for indexing purposes
                theta = tfd.Deterministic(jnp.zeros((n_theta, 1))), # dummy f_in for indexing purposes
                obs = tfd.Exponential(jnp.full((n_theta, n_obs, 1), obs_rate)),
            ),
            batch_ndims=1
        )

    # def f_in_fn_train(times):
        # logits = jnp.log(jnp.ones((n_theta,)) / n_theta)
        # return tfd.JointDistributionNamed(
            # dict(
                # mu = tfd.Deterministic(jnp.zeros((1, 1))), # dummy f_in for indexing purposes
                # theta = tfd.Deterministic(jnp.zeros((1, 1))), # dummy f_in for indexing purposes
                # theta_index = tfd.Categorical(logits=logits),
                # obs = lambda theta_index: tfd.Deterministic(
                    # times[0, theta_index, ...].reshape(
                        # theta_index.shape[:1] + (1, n_obs, 1)
                    # )
                # ),
            # ),
            # batch_ndims=1
        # )

    key = jr.PRNGKey(cfg.seed)

    theta_key, y_key, f_in_key, key = jr.split(key, 4)
    theta_truth = prior_fn(n_theta).sample((1,), seed=theta_key)
    f_in: dict = f_in_fn(n_obs, n_theta).sample((1,), seed=f_in_key) #type:ignore
    y_observed = simulator_fn(y_key, theta_truth, f_in)

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
        n_labels=3,
        index_dim=1,  # Enable indexing for f_in
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
        prior_fn,
        p_local,
        simulator_fn,
        ['mu'],
        ['theta'],
        n_theta,
        n_rounds,
        n_simulations,
        n_epochs,
        y_observed, # type: ignore
        independence,
        optimiser=optax.adam(cfg.training.learning_rate),
        batch_size=int(n_simulations * cfg.training.batch_size_fraction),
        f_in=f_in_fn,
        f_in_args=(n_obs, 1),
        f_in_args_global=(n_obs, n_theta),
        f_in_target=f_in
    )
    logger.info(f"SFMPE bottom-up training completed in {time.time() - start_time:.2f} seconds")

    logger.info("Sampling SFMPE posterior")
    start_time = time.time()
    # Create flattened f_in index for posterior sampling
    f_in_flattened = flatten_f_in(f_in)
    posterior = estim.sample_posterior(
        _flatten(y_observed)[..., None],
        labels, #type:ignore
        slices,
        masks=masks,
        n_samples=n_post_samples,
        index=f_in_flattened
    )
    logger.info(f"SFMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")

    logger.info('Starting internal FMPE training')

    # Create proxy functions for FMPE training
    def fmpe_prior_fn(key, n_samples):
        """Prior function compatible with FMPE interface"""
        theta_samples = prior_fn(n_theta).sample((n_samples,), seed=key)
        # Flatten: combine mu and theta
        mu_flat = theta_samples['mu'].reshape((n_samples, 1)) #type:ignore
        theta_flat = theta_samples['theta'].reshape((n_samples, n_theta)) #type:ignore 
        return jnp.concatenate([mu_flat, theta_flat], axis=1)

    def sim_corrected_fmpe_prior_fn(key, n_samples):
        n = n_samples // n_theta
        return fmpe_prior_fn(key, n)
    
    def fmpe_simulator_fn(key, theta_flat):
        """Simulator function compatible with FMPE interface"""
        # Reconstruct structured theta from flat representation
        mu = theta_flat[..., :1, None]  # Add singleton dim for simulator
        theta = theta_flat[..., 1:, None]  # Add singleton dim for simulator
        theta_dict = {'mu': mu, 'theta': theta}
        
        # Run simulator
        y_samples = simulator_fn(key, theta_dict, f_in)
        return y_samples['obs'].reshape((theta_flat.shape[0], -1)) # type: ignore
    
    # Create FMPE model
    fmpe_nn = MLPVectorField(
        theta_dim=1 + n_theta,  # mu + theta
        context_dim=n_obs * n_theta,  # flattened observations
        latent_dim=cfg.fmpe.nn.latent_dim,
        n_layers=cfg.fmpe.nn.n_layers,
        dropout=cfg.fmpe.nn.dropout,
        activation=nnx.relu
    )
    
    fmpe_model = CNF(fmpe_nn)
    fmpe_estim = FMPE(fmpe_model, rngs=rngs)
    
    # Flatten observed data for FMPE
    fmpe_y_observed = y_observed['obs'].reshape(-1) # type: ignore
    
    # Train using round-based approach
    train_key, key = jr.split(key)
    logger.info("Starting FMPE round-based training")
    start_time = time.time()
    fmpe_estim = train_fmpe_rounds(
        train_key,
        fmpe_estim,
        sim_corrected_fmpe_prior_fn,
        fmpe_simulator_fn,
        fmpe_y_observed,
        theta_shape=(1 + n_theta,),
        n_rounds=n_rounds,
        n_simulations=n_simulations,
        n_epochs=n_epochs,
        optimizer=optax.adam(cfg.training.learning_rate),
        batch_size=int((n_simulations // n_theta) * cfg.training.batch_size_fraction)
    )

    logger.info(f"FMPE round-based training completed in {time.time() - start_time:.2f} seconds")

    # Sample from posterior
    logger.info("Sampling FMPE posterior")
    start_time = time.time()
    fmpe_posterior_samples = fmpe_estim.sample_posterior(
        fmpe_y_observed[None, ...],
        theta_shape=(1 + n_theta,),
        n_samples=n_post_samples
    )

    logger.info(f"FMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")
    
    n_cal = cfg.analysis.n_cal
    logger.info(f"Creating calibration dataset with {n_cal} samples")
    start_time = time.time()

    def sample_single_sfmpe_posterior(key, x):
        theta_0 = jr.normal(key, (1, 1 + n_theta, 1))
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
        return jnp.concatenate([
            posterior['mu'].reshape(-1),
            posterior['theta'].reshape(-1)
        ])

    def sample_single_fmpe_posterior(key, x):
        dim = (1 + n_theta)
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
    #TODO: figure out design to generalise this and put it into lc2st.py
    null_stats, main_stat, p_value = apply_lc2st(
        analyse_key,
        create_sfmpe_calibration_dataset,
        sample_single_sfmpe_posterior,
        lambda: prior_fn(n_theta),
        lambda key, theta: simulator_fn(key, theta, f_in),
        y_observed['obs'].reshape(-1), # type: ignore
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
        y_observed['obs'].reshape(-1), # type: ignore
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
    """
    Apply l-c2st pipeline

    Input:
        key: JAX PRNG key.
        sample_single_posterior: Function that samples from a posterior.
        prior_fn: Prior function.
        simulator_fn: Simulator function.
        observation: Observation.
        posterior_samples: Posterior samples.
        n_epochs: Number of epochs.
        n_cal: Number of calibration samples.
        rngs: nnx.Rngs.
    """
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

    # Create output directory and make quant plot
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lc2st_quant_plot(
        T_data = main_stat,
        T_null = null_stats,
        p_value = p_value,
        reject = bool(p_value < 0.05),
        conf_alpha = 0.05,
        out_path = out_dir / 'quant_plot.jpg'
    )

    # Write out stats json with serialize
    stats = {
        'main_stat': float(main_stat),
        'null_stats': null_stats.tolist(),
        'p_value': float(p_value),
        'reject': bool(p_value < 0.05)
    }
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f)

def _flatten(x: PyTree) -> jnp.ndarray:
    """
    Flatten a batched SFMPE PyTree into a 2D array.

    SFMPE PyTrees are dicts with the (name, samples) items for each variable.
    Samples are shaped as (batch_dims, event_dims, sample_dim). Sample dims are always 1. In this example batch dims are 1. And so the remaining dims to be flattened are the event dims.
    """
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
    theta_keys = ['mu', 'theta']  # global and local parameters
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

def create_sfmpe_calibration_dataset(
    key: jnp.ndarray,
    sample_posterior: Callable[[Array, PyTree], PyTree],
    prior_fn: Callable[[], tfd.Distribution],
    simulator_fn: Callable[[Array, PyTree], PyTree],
    n: int
    ) -> Tuple[Array, Array, Array]:
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
    prior_key, post_key, sim_key = jr.split(key, 3)
    prior = prior_fn(prior_key, n)
    y = simulator_fn(sim_key, prior)
    post_estimate = vmap(sample_posterior)(jr.split(post_key, n), y)
    return y, prior, post_estimate

@hydra.main(version_base=None, config_path="conf", config_name="brownian_config")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration management."""
    # Setup logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting hierarchical Brownian experiment")
    
    # Run the experiment
    run(cfg)


if __name__ == "__main__":
    main()
