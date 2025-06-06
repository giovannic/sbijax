import pytest
from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.fmpe import SFMPE
from sfmpe.bottom_up import train_bottom_up, get_posterior
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer

from sbijax import FMPE
from sbijax.nn import make_cnf

from flax import nnx

def test_hierarchical_parameters():
    tol = 1e-3
    n_rounds = 2
    global_noise = 1e-1
    local_noise = 1
    measurement_noise = 1e-2
    n_simulations = 1_000
    n_epochs = 1_000
    n_post_samples = 1_000
    n_z = 4

    independence = {
        'local': ['z', 'obs'],
        'cross_local': [('z', 'obs', None)],
    }

    # make prior distribution
    def prior_fn(n_z):
        prior = tfd.JointDistributionNamed(
            dict(
                theta=tfd.Normal(jnp.zeros((1, 1)), global_noise),
                z=lambda theta: tfd.Independent(
                    tfd.Normal(
                        jnp.repeat(theta, n_z, axis=-2),
                        local_noise
                    ),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=1,
        )
        return prior

    # make simulator function
    def simulator_fn(seed, theta):
        return {
            'obs': tfd.Normal(theta['z'], measurement_noise).sample(
                seed=seed
            )
        }

    key = jr.PRNGKey(0)

    theta_key, y_key, key = jr.split(key, 3)
    theta_truth = prior_fn(n_z).sample((1,), seed=theta_key)
    y_observed = simulator_fn(y_key, theta_truth)

    rngs = nnx.Rngs(0)
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

    model = CNF(transform=nn)

    estim = SFMPE(model)

    train_key, key = jr.split(key)
    labels, slices, masks = train_bottom_up(
        train_key,
        estim,
        prior_fn,
        simulator_fn,
        ['theta'],
        ['z'],
        n_z,
        n_rounds,
        n_simulations,
        n_epochs,
        y_observed,
        independence
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

    sbijax_nn = make_cnf(5)
    def sbijax_prior_fn():
        prior = tfd.JointDistributionNamed(
            dict(
                theta=tfd.Normal(5., global_noise),
                z=lambda theta: tfd.Independent(
                    tfd.Normal(
                        jnp.full((n_z,), theta),
                        local_noise
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
        arg = { 'theta': theta['theta'][...,:1], 'z': theta['theta'][...,1:] }
        return simulator_fn(seed, arg)['obs']

    sbijax_estim = FMPE(
        (sbijax_prior_fn, sbijax_simulator_fn),
        sbijax_nn,
    )

    data, params = None, {}
    for _ in range(n_rounds):
        data, params = sbijax_estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=sbijax_y_observed,
            data=data,
            n_simulations=n_simulations // n_z,
        )
        params, _ = sbijax_estim.fit(
            jr.PRNGKey(2),
            data=data,
            n_iter=n_epochs
        )

    sbijax_posterior, _ = sbijax_estim.sample_posterior(
        rng_key=jr.PRNGKey(3),
        params=params,
        observable=sbijax_y_observed,
        n_samples=n_post_samples,
    )
    sbijax_theta_hat_array = jnp.array(sbijax_posterior.posterior.theta).reshape( # type: ignore
        (n_post_samples, (1 + n_z))
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
    theta_hat = jnp.mean(posterior['theta'], keepdims=True, axis=0)
    z_hat = jnp.mean(posterior['z'], keepdims=True, axis=0)
    assert sbijax_theta_hat[None,...] == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert theta_hat == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert sbijax_z_hat[None,...] == pytest.approx(theta_truth['z'], tol) # type: ignore
    assert z_hat == pytest.approx(theta_truth['z'], tol) # type: ignore
