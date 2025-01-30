import pytest
from functools import reduce
from jax import numpy as jnp, random as jr, vmap
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.fmpe import SFMPE
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer

from sbijax import FMPE
from sbijax.nn import make_cnf

from flax import nnx

def prod(x):
    return reduce(lambda x, y: x * y, x)

def test_hierarchical_parameters():
    tol = 1e-3
    n_rounds = 5
    global_noise = 1e2
    local_noise = 1e-1
    measurement_noise = 1e-1
    n_simulations = 100
    n_epochs = 1_000
    n_post_samples = 1_000
    n_z = 4

    theta_index = {
        'theta': jnp.full((1, 1, 1), 0),
        'z': jnp.full((1, n_z, 1), 0),
    }

    y_index = {
        'obs': jnp.full((1, n_z, 1), 0),
    }

    def prior_fn(**kwargs):
        n_z = kwargs["n_z"]
        prior = tfd.JointDistributionNamed(
            dict(
                theta=tfd.Normal(5., global_noise),
                z=lambda theta: tfd.Independent(
                    tfd.Normal(
                        jnp.broadcast_to(theta, (n_z,)),
                        jnp.broadcast_to(local_noise, (n_z,))
                    )
                )
            ),
            batch_ndims=1,
        )
        return prior
       
    def simulator_fn(seed, theta, **_):
        sample_size= theta["theta"].shape[0]


        # sample from guassian mixture
        def sample_row(seed, z):
            return tfd.Normal(z, measurement_noise).sample(seed=seed)

            
        # vmap over sample size
        keys = jr.split(seed, sample_size)
        samples = vmap(sample_row, in_axes=(0, 0, 0))(
            keys,
            theta["z"],
        )

        return { 'obs' : samples }

    theta_key, y_key = jr.split(jr.PRNGKey(0))

    theta_truth = prior_fn(n_z=n_z).sample((1,), seed=theta_key)
    y_observed = simulator_fn(y_key, theta_truth)

    rngs = nnx.Rngs(0)
    config = {
        'latent_dim': 64,
        'label_dim': 64,
        'index_out_dim': 64,
        'n_encoder': 1,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }
    theta_value_dim = 1
    nn = Transformer(
        config,
        context_value_dim=1,
        n_context_labels=1,
        context_index_dim=1,
        theta_value_dim=theta_value_dim,
        n_theta_labels=2,
        theta_index_dim=1,
        rngs=rngs
    )
    model = CNF(
        theta_shape=(n_z, theta_value_dim),
        transform=nn
    )

    data_batch_ndims = {
        'theta': {'s': 1},
        'y': {'obs': 1}
    }

    theta_batch_shapes = {
        'theta': (1,),
        'z': (1,)
    }

    estim = SFMPE(
        (prior_fn, simulator_fn),
        model,
        theta_batch_shapes
    )

    data = None
    for _ in range(n_rounds):
        #TODO: simulate from n_z=1
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            first_round=data is None,
            observable=y_observed,
            context_index=y_index,
            theta_index=theta_index,
            data=data,
            n_simulations=n_simulations,
            n_z=1
        )
        #TODO: fit to n_z=1
        estim.fit(
            jr.PRNGKey(2),
            data=data,
            n_iter=n_epochs,
            data_batch_ndims=data_batch_ndims,
            theta_event_sizes={
                'theta': 1,
                'z': n_z
            }
        )
        #TODO: calibrate theta to fit to n_z=n_z

    print('sbijax')

    # flatten y_observed
    sbijax_y_observed = y_observed['obs'].reshape((1, -1)) # type: ignore

    sbijax_nn = make_cnf(6)
    def sbijax_prior_fn():
        prior = prior_fn(n_z=n_z)

        return tfd.JointDistributionNamed(
            dict(
                theta=tfd.Blockwise(prior)
            ),
            batch_ndims=1,
        )

    def sbijax_simulator_fn(seed, theta):
        arg = { 'theta': theta[...,:1], 'z': theta[...,1:] }
        return simulator_fn(seed, arg)

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
            n_simulations=n_simulations,
        )
        params, _ = sbijax_estim.fit(
            jr.PRNGKey(2),
            data=data,
            n_iter=n_epochs
        )

    posterior, _ = estim.sample_posterior( 
        jr.PRNGKey(3),
        y_observed,
        context_index=y_index,
        theta_index=theta_index,
        n_samples=n_post_samples,
    )

    sbijax_posterior, _ = sbijax_estim.sample_posterior(
        rng_key=jr.PRNGKey(3),
        params=params,
        observable=sbijax_y_observed,
        n_samples=n_post_samples,
    )
    sbijax_theta_hat = jnp.mean(jnp.array(sbijax_posterior.posterior.theta), keepdims=True, axis=0) # type: ignore
    sbijax_z_hat = jnp.mean(jnp.array(sbijax_posterior.posterior.z), keepdims=True, axis=0) # type: ignore
    theta_hat = jnp.mean(jnp.array(posterior.posterior.theta), keepdims=True, axis=0) # type: ignore
    z_hat = jnp.mean(jnp.array(posterior.posterior.z), keepdims=True, axis=0) # type: ignore
    theta_truth = theta_truth['theta'] # type: ignore
    assert sbijax_theta_hat == pytest.approx(theta_truth['theta'], tol)
    assert theta_hat == pytest.approx(theta_truth['theta'], tol)
    assert sbijax_z_hat == pytest.approx(theta_truth['z'], tol)
    assert z_hat == pytest.approx(theta_truth['z'], tol)
    assert False
