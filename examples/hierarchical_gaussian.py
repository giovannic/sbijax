import argparse
from pathlib import Path
from typing import Tuple, Callable
from jaxtyping import PyTree

from jax import numpy as jnp, random as jr, tree
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.bottom_up import train_bottom_up, get_posterior
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.cnf import CNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.mlp import MLPVectorField
from sfmpe.train_rounds import train_fmpe_rounds
from sfmpe.c2stnf import (
    train_c2st_nf_main_classifier,
    precompute_c2st_nf_null_classifiers,
    evaluate_c2st_nf,
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier
)

from flax import nnx
import optax

def run(n_rounds=2, n_epochs=1000):
    n_simulations = 1_000
    n_post_samples = 10
    var_theta = 1. #TODO: change to 10
    var_mu = 1. #TODO: change to 10
    var_obs = 1.
    n_obs = 10 #TODO: scale up
    n_theta = 5

    independence = {
        'local': ['theta', 'obs'], # Shouldn't this be commented out?
        'cross_local': [('theta', 'obs', (0, 0))],
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

    key = jr.PRNGKey(0)

    theta_key, y_key, key = jr.split(key, 3)
    theta_truth = prior_fn(n_theta).sample((1,), seed=theta_key)
    y_observed = simulator_fn(y_key, theta_truth)

    rngs = nnx.Rngs(0)
    config = {
        'latent_dim': 64,
        'label_dim': 8,
        'index_out_dim': 0,
        'n_encoder': 1,
        'n_decoder': 2,
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
        optimiser=optax.adam(3e-4)
    )

    post_key, key = jr.split(key)
    posterior = get_posterior(
        post_key,
        estim,
        labels,
        slices,
        masks,
        n_post_samples,
        theta_truth,
        y_observed,
        independence
    )

    print('internal FMPE')

    # Create proxy functions for FMPE training
    def fmpe_prior_fn(key, n_samples):
        """Prior function compatible with FMPE interface"""
        theta_samples = prior_fn(n_theta).sample((n_samples,), seed=key)
        # Flatten: combine mu and theta
        mu_flat = theta_samples['mu'].reshape((n_samples, 1))
        theta_flat = theta_samples['theta'].reshape((n_samples, n_theta))
        return jnp.concatenate([mu_flat, theta_flat], axis=1)
    
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
    fmpe_estim = train_fmpe_rounds(
        train_key,
        fmpe_estim,
        fmpe_prior_fn,
        fmpe_simulator_fn,
        fmpe_y_observed,
        theta_shape=(1 + n_theta,),
        n_rounds=n_rounds,
        n_simulations=n_simulations,
        n_epochs=n_epochs,
        optimizer=optax.adam(3e-4),
        batch_size=100
    )

    # Sample from posterior
    post_key, key = jr.split(key)
    fmpe_posterior_samples = fmpe_estim.sample_posterior(
        fmpe_y_observed[None, ...],
        theta_shape=(1 + n_theta,),
        n_samples=n_post_samples
    )
    
    fmpe_theta_hat = jnp.mean(
        fmpe_posterior_samples[..., :1],
        keepdims=True,
        axis=0
    )
    fmpe_z_hat = jnp.mean(
        fmpe_posterior_samples[..., 1:],
        keepdims=True,
        axis=0
    )
    theta_hat = jnp.mean(posterior['mu'], keepdims=True, axis=0)
    z_hat = jnp.mean(posterior['theta'], keepdims=True, axis=0)

    cal_key, key = jr.split(key)
    n_cal = 10_000
    d = create_calibration_dataset(
        cal_key,
        prior_fn,
        simulator_fn,
        n_theta,
        n_cal,
    )

    print('d shapes', tree.map(lambda x: x.shape, d))

    def inverse(theta, x):
        print(theta.shape, x.shape)
        # Use SFMPE's sample_base_dist method like in the test
        # theta and x should be flat arrays with singleton dimension added
        z_samples = estim.sample_base_dist(
            theta[..., None],  # Add singleton dimension for theta
            x[..., None],      # Add singleton dimension for context
            labels,
            slices,
            masks=masks
        )
        
        # Flatten the structured output
        z = jnp.concatenate([
            z_samples['mu'].reshape((theta.shape[0], -1)),
            z_samples['theta'].reshape((theta.shape[0], -1))
        ], axis=1)
        return z


    def inverse_fmpe(theta, x):
        """Inverse function for FMPE model"""
        z_samples = fmpe_estim.sample_base_dist(
            theta,
            x,
            theta_shape=(1 + n_theta,)
        )
        return z_samples

    n_cal_epochs = 100
    analyse_key, key = jr.split(key)
    out_dir = Path(__file__).parent/'outputs'/'hierarchical_gaussian'
    # Preprocess posterior samples for SFMPE
    sfmpe_posterior_flat = _flatten(posterior)
    
    analyse_c2stnf(
        analyse_key,
        d,
        inverse,
        y_observed['obs'].reshape(-1),
        sfmpe_posterior_flat,
        n_cal_epochs,
        n_cal,
        out_dir/'sfmpe'
    )
    analyse_c2stnf(
        analyse_key,
        d,
        inverse_fmpe,
        y_observed['obs'].reshape(-1),
        fmpe_posterior_samples.reshape(fmpe_posterior_samples.shape[0], -1),
        n_cal_epochs,
        n_cal,
        out_dir/'fmpe'
    )

    print("SFMPE results:")
    print(f"theta_truth: {theta_truth}")
    print(f"theta_hat: {theta_hat}")
    print(f"z_hat: {z_hat}")
    
    print("FMPE results:")
    print(f"fmpe_theta_hat: {fmpe_theta_hat}")
    print(f"fmpe_z_hat: {fmpe_z_hat}")

def analyse_c2stnf(
    key: jnp.ndarray,
    d: Tuple[jnp.ndarray, jnp.ndarray],
    inverse: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    observation: jnp.ndarray,
    posterior_samples: jnp.ndarray,
    n_epochs: int,
    n_cal: int,
    out_dir: Path
    ):
    """
    Analyse C2ST-NF

    Input:
        key: JAX PRNG key for reproducibility.
        d: Tuple of (Theta_n, X_n) sampled from p(theta, x). Expects a flat dataset
        inverse: Callable that applies the inverse NF transformation.
        observation: Observation.
        n_cal: Number of calibration samples.
        out_dir: Output directory.
    """
    # Generate z samples using inverse transformation
    theta_cal, x_cal = d
    z_samples = inverse(theta_cal, x_cal)
    
    # Train main classifier
    main_key, null_key, key = jr.split(key, 3)
    n_layers = 1
    n_null = 100
    activation = nnx.relu
    latent_dim = z_samples.shape[-1]
    
    main = BinaryMLPClassifier(
        dim=latent_dim,
        latent_dim=16,
        n_layers=n_layers,
        activation=activation,
        rngs=nnx.Rngs(main_key),
    )
    print(f'Training main classifier with {n_epochs} epochs')
    train_c2st_nf_main_classifier(
        main_key,
        main,
        z_samples,
        n_epochs
    )

    null_cls = MultiBinaryMLPClassifier(
        dim=latent_dim,
        latent_dim=16,
        n_layers=n_layers,
        activation=activation,
        n=n_null,
        rngs=nnx.Rngs(null_key)
    )
    print(f'Training {n_null} null classifiers with {n_epochs} epochs')
    precompute_c2st_nf_null_classifiers(
        null_key,
        null_cls,
        latent_dim=latent_dim,
        N_cal=n_cal,
        num_epochs=n_epochs
    )

    # Generate posterior samples at observation using provided posterior samples
    z_posterior = inverse(
        posterior_samples,
        jnp.repeat(observation[None, :], posterior_samples.shape[0], axis=0)
    )
    
    ev_key, key = jr.split(key)
    null_stats, main_stat, p_value = evaluate_c2st_nf(
        ev_key,
        z_posterior,
        main,
        null_cls,
        latent_dim=latent_dim
    )

    # Create output directory and make quant plot
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the quant plot function from lc2stnf (still available there)
    from sfmpe.lc2stnf import lc2st_quant_plot
    lc2st_quant_plot(
        T_data = main_stat,
        T_null = null_stats,
        p_value = p_value,
        reject = bool(p_value < 0.05),
        conf_alpha = 0.05,
        out_path = out_dir / 'quant_plot.jpg'
    )

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

def create_calibration_dataset(
    key: jnp.ndarray,
    prior_fn: Callable[..., PyTree],
    simulator_fn: Callable[..., PyTree],
    n_theta: int,
    n: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    prior_key, sim_key = jr.split(key)
    #TODO: update to flatten sfmpe version
    prior = prior_fn(n_theta).sample((n,), seed=prior_key)
    x = _flatten(prior)
    y = simulator_fn(sim_key, prior)
    y = _flatten(y)
    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Gaussian Flow Matching Example")
    parser.add_argument("--n_rounds", type=int, default=2, help="Number of training rounds (default: 2)")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs per round (default: 1000)")
    
    args = parser.parse_args()
    
    print(f"Running with n_rounds={args.n_rounds}, n_epochs={args.n_epochs}")
    run(n_rounds=args.n_rounds, n_epochs=args.n_epochs)
