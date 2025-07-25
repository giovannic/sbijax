import pytest
from pathlib import Path
from typing import Tuple, Callable
from jaxtyping import PyTree

from jax import numpy as jnp, random as jr, vmap
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.fmpe import SFMPE
from sfmpe.bottom_up import train_bottom_up, get_posterior
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.lc2stnf import (
    train_l_c2st_nf_main_classifier,
    precompute_null_distribution_nf_classifiers,
    evaluate_l_c2st_nf,
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    lc2st_quant_plot
)

from sbijax import FMPE
from sbijax.nn import make_cnf

from flax import nnx
import optax

def run():
    tol = 1e-3
    n_rounds =  2 #TODO: change to 10
    n_simulations = 1_000
    n_epochs = 1_000
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

    model = CNF(transform=nn)

    estim = SFMPE(model)

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
        n_early_stopping_patience=500,
        n_early_stopping_delta=1e-3,
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

    print('sbijax')

    # flatten y_observed
    sbijax_y_observed = y_observed['obs'].reshape((1, -1)) # type: ignore
    sbijax_nn = make_cnf(1 + n_theta)
    def sbijax_prior_fn():
        prior = tfd.JointDistributionNamed(
            dict(
                mu = tfd.Normal(0., var_mu),
                theta = lambda mu: tfd.Independent(
                    tfd.Normal(
                        jnp.full((n_theta,), mu),
                        var_theta
                    ),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=1,
            use_vectorized_map=True
        )

        return tfd.JointDistributionNamed(
            dict(
                theta=tfd.Blockwise(prior)
            ),
            batch_ndims=1,
        )

    def sbijax_simulator_fn(seed, theta):
        arg = {
            'mu': theta['theta'][...,:1, None],
            'theta': theta['theta'][...,1:, None]
        }
        n_samples = arg['mu'].shape[0]
        obs = simulator_fn(seed, arg)['obs']
        obs = obs.reshape((n_samples, -1))
        return obs # type:ignore

    sbijax_estim = FMPE(
        (sbijax_prior_fn, sbijax_simulator_fn),
        sbijax_nn,
    )

    data, params = None, {}
    for _ in range(n_rounds):
        sample_key, key = jr.split(key)
        data, params = sbijax_estim.simulate_data_and_possibly_append(
            sample_key,
            params=params,
            observable=sbijax_y_observed,
            data=data,
            n_simulations=n_simulations // n_theta,
        )
        train_key, key = jr.split(key)
        params, _ = sbijax_estim.fit(
            train_key,
            data=data,
            n_iter=n_epochs
        )

    post_key, key = jr.split(key)
    sbijax_posterior, _ = sbijax_estim.sample_posterior(
        rng_key=post_key,
        params=params,
        observable=sbijax_y_observed,
        n_samples=n_post_samples,
    )
    sbijax_theta_hat_array = jnp.array(sbijax_posterior.posterior.theta).reshape( # type: ignore
        (n_post_samples, -1)
    )
    sbijax_theta_hat = jnp.mean(
        sbijax_theta_hat_array[..., :1],
        keepdims=True,
        axis=0
    )
    sbijax_z_hat = jnp.mean(
        sbijax_theta_hat_array[..., 1:],
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

    dummy_key = jr.PRNGKey(0)

    def inverse(theta, x):
        theta, x = theta[..., None], x[..., None]
        def sample_pair(theta, x):
            return estim.sample_structured_posterior(
                key,
                x[None, ...],
                labels,
                slices,
                masks=masks,
                n_samples=1,
                theta_0=theta[None, ...],
                direction='backward'
            )

        z_struct = vmap(sample_pair)(theta, x)
        z = jnp.concatenate([
            z_struct['mu'].reshape((x.shape[0], -1)),
            z_struct['theta'].reshape((x.shape[0], -1))
        ], axis=1)
        return z


    def inverse_sbijax(theta, x):
        #TODO: this is going to fail
        import warnings
        warnings.warn('this is going to fail')
        z_data = sbijax_estim.sample_posterior(
            dummy_key,
            params,
            x,
            n_samples=1,
            theta_0=theta,
            direction='backward'
        )
        z = jnp.concatenate([
            jnp.ndarray(z_data.posterior.mu.reshape((x.shape[0], -1))), #type:ignore
            jnp.ndarray(z_data.posterior.theta).reshape((x.shape[0], -1)) #type:ignore
        ])
        return z

    n_cal_epochs = 100
    analyse_key, key = jr.split(key)
    out_dir = Path(__file__).parent/'outputs'/'hierarchical_gaussian'
    analyse_lc2stnf(
        analyse_key,
        d,
        inverse,
        y_observed['obs'].reshape(-1),
        n_cal_epochs,
        n_cal,
        out_dir/'sfmpe'
    )
    analyse_lc2stnf(
        analyse_key,
        d,
        inverse_sbijax,
        y_observed['obs'],
        n_cal,
        n_cal_epochs,
        out_dir/'sbijax'
    )

    print(theta_truth)
    print(theta_hat)
    print(z_hat)
    print(sbijax_theta_hat)
    print(sbijax_z_hat)
    assert sbijax_theta_hat[None,...] == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert theta_hat == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert sbijax_z_hat[None,...] == pytest.approx(theta_truth['z'], tol) # type: ignore
    assert z_hat == pytest.approx(theta_truth['z'], tol) # type: ignore

def analyse_lc2stnf(
    key: jnp.ndarray,
    d: Tuple[jnp.ndarray, jnp.ndarray],
    inverse: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    observation: jnp.ndarray,
    n_epochs: int,
    n_cal: int,
    out_dir: Path
    ):
    """
    Analyse LC2ST-NF

    Input:
        key: JAX PRNG key for reproducibility.
        d: Tuple of (Theta_n, X_n) sampled from p(theta, x). Expects a flat dataset
        inverse: Callable that applies the inverse NF transformation .
        observation: Observation.
        n_cal: Number of calibration samples.
        out_dir: Output directory.
    """
    # Train main classifier
    main_key, null_key, key = jr.split(key, 3)
    n_layers = 1
    n_null = 100
    activation = nnx.relu
    dim = d[0].shape[-1] + d[1].shape[-1]
    main = BinaryMLPClassifier(
        dim=dim,
        n_layers=n_layers,
        activation=activation,
        rngs=nnx.Rngs(key),
    )
    print(f'Training main classifier with {n_epochs} epochs')
    train_l_c2st_nf_main_classifier(
        main_key,
        main,
        d,
        inverse,
        n_epochs
    )

    null_key, key = jr.split(null_key)
    null_cls = MultiBinaryMLPClassifier(
        dim=dim,
        n_layers=n_layers,
        activation=activation,
        n=n_null,
        rngs=nnx.Rngs(null_key)
    )
    train_key, key = jr.split(key)
    print(f'Training {n_null} null classifiers with {n_epochs} epochs')
    precompute_null_distribution_nf_classifiers(
        train_key,
        d,
        null_cls,
        n_epochs
    )

    ev_key, key = jr.split(key)
    null_stats, main_stat, p_value = evaluate_l_c2st_nf(
        ev_key,
        observation,
        main,
        null_cls,
        latent_dim=d[0].shape[-1],
        Nv=n_cal
    )

    # Make quant plot
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
    run()
