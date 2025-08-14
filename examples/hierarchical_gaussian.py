import json
import argparse
import logging
import time
from pathlib import Path
from typing import Tuple, Callable
from jaxtyping import PyTree, Array
from flax import nnx
import optax

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

def run(n_theta: int, n_obs: int, n_simulations=1_000, n_rounds=2, n_epochs=1000):
    logger = logging.getLogger(__name__)
    logger.info(f"Running with n_simulations={n_simulations}, n_rounds={n_rounds}, n_epochs={n_epochs}")
    n_post_samples = 1_000
    var_mu = 1.
    var_theta = 1e-1
    var_obs = 1e-2

    independence = {
        'local': ['obs'],  # y observations independent of each other
        'cross': [('mu', 'obs')],  # mu is independent of observations
        'cross_local': [('theta', 'obs', (0, 0))]  # theta[i] connects to y[i]
    }

    # make prior distribution
    def prior_fn(n):
        prior = tfd.JointDistributionNamed(
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
        return prior

    # make simulator function
    def simulator_fn(seed, theta):
        obs = tfd.Independent(
            tfd.Normal(theta['theta'], var_obs),
            reinterpreted_batch_ndims=1
        ).sample((n_obs,), seed=seed)
        obs = jnp.transpose(obs, (1, 2, 0, 3)) #type:ignore
        return {
            'obs': obs
        }

    key = jr.PRNGKey(42)

    theta_key, y_key, key = jr.split(key, 3)
    theta_truth = prior_fn(n_theta).sample((1,), seed=theta_key)
    y_observed = simulator_fn(y_key, theta_truth)

    rngs = nnx.Rngs(key)
    config = {
        'latent_dim': 64,
        'label_dim': 8,
        'index_out_dim': 0,
        'n_encoder': 1,
        'n_decoder': 1,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }

    nn = Transformer(
        config,
        value_dim=1,
        n_labels=3,
        index_dim=0,
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
        simulator_fn,
        ['mu'],
        ['theta'],
        n_theta,
        n_rounds,
        n_simulations,
        n_epochs,
        y_observed,
        independence,
        optimiser=optax.adam(3e-4),
        batch_size=n_simulations // 10
    )
    logger.info(f"SFMPE bottom-up training completed in {time.time() - start_time:.2f} seconds")

    logger.info("Sampling SFMPE posterior")
    start_time = time.time()
    posterior = estim.sample_posterior(
        _flatten(y_observed)[..., None],
        labels, #type:ignore
        slices,
        masks=masks,
        n_samples=n_post_samples
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
        y_samples = simulator_fn(key, theta_dict)
        return y_samples['obs'].reshape((theta_flat.shape[0], -1))
    
    # Create FMPE model
    fmpe_nn = MLPVectorField(
        theta_dim=1 + n_theta,  # mu + theta
        context_dim=n_obs * n_theta,  # flattened observations
        latent_dim=64,
        n_layers=2,
        dropout=.1,
        activation=nnx.relu
    )
    
    fmpe_model = CNF(fmpe_nn)
    fmpe_estim = FMPE(fmpe_model, rngs=rngs)
    
    # Flatten observed data for FMPE
    fmpe_y_observed = y_observed['obs'].reshape(-1)
    
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
        optimizer=optax.adam(3e-4),
        batch_size=(n_simulations // n_theta) // 10
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

    print(f'truth: {fmpe_y_observed}')
    print(f'posterior mean: {jnp.mean(fmpe_posterior_samples, axis=0)}')
    logger.info(f"FMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")
    
    n_cal = 1_000
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
            theta_0 = theta_0
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
    
    n_cal_epochs = 100
    analyse_key, key = jr.split(key)
    out_dir = Path(__file__).parent/'outputs'/'hierarchical_gaussian'/f'{n_simulations}_sims_{n_rounds}_rounds_{n_epochs}_epochs'

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
        simulator_fn,
        y_observed['obs'].reshape(-1),
        sfmpe_posterior_flat,
        n_cal_epochs,
        n_cal,
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
        y_observed['obs'].reshape(-1),
        fmpe_posterior_samples,
        n_cal_epochs,
        n_cal,
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
    n_layers = 2
    n_null = 100
    activation = nnx.relu

    main = BinaryMLPClassifier(
        dim=classifier_dim,
        latent_dim=16,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs
    )

    null_cls = MultiBinaryMLPClassifier(
        dim=classifier_dim,
        latent_dim=16,
        n_layers=n_layers,
        activation=activation,
        n=n_null,
        rngs=rngs
    )
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Gaussian Flow Matching Example")
    parser.add_argument("--n_simulations", type=int, default=1_000, help="Number of simulations per round")
    parser.add_argument("--n_rounds", type=int, default=1, help="Number of training rounds (default: 1)")
    parser.add_argument("--n_epochs", type=int, default=1_000, help="Number of epochs per round (default: 1000)")
    parser.add_argument("--n_theta", type=int, default=2, help="Number of theta parameters (default: 2)")
    parser.add_argument("--n_obs", type=int, default=5, help="Number of theta parameters (default: 5)")
    
    args = parser.parse_args()
    
    # Setup logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running with n_rounds={args.n_rounds}, n_epochs={args.n_epochs}")
    run(
        n_theta=args.n_theta,
        n_obs=args.n_obs,
        n_simulations=args.n_simulations,
        n_rounds=args.n_rounds,
        n_epochs=args.n_epochs
    )
