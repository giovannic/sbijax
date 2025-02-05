import pytest
from functools import reduce
from jax import numpy as jnp, random as jr, vmap, tree
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.fmpe import SFMPE
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.util.dataloader import (
    flatten_structured,
    structured_as_batch_iterators,
    pad_multidim_event,
)

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

    independence = {
        'local': ['z', 'obs'],
        'cross_local': [('z', 'obs', None)],
    }

    # make prior distribution
    def prior_fn(n_z):
        prior = tfd.JointDistributionNamed(
            dict(
                theta=tfd.Normal([[5.]], global_noise),
                z=lambda theta: tfd.Independent(
                    tfd.Normal(
                        jnp.broadcast_to(theta, (theta.shape[0], n_z, 1)),
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
        'label_dim': 64,
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
        n_labels=1,
        index_dim=0,
        rngs=rngs
    )
    model = CNF(
        transform=nn
    )

    batch_shapes = {
        'theta': (1,),
        'z': (1,),
        'obs': (1,)
    }

    estim = SFMPE(model, batch_shapes)

    data, data_slices = (None, {})
    for _ in range(n_rounds):
        # Fit p(z|theta, y)
        theta_key, obs_key, key = jr.split(key, 3)
        if data is not None and data_slices:
            (
                flat_data,
                labels,
                masks,
                _
            ) = flatten_structured(
                data,
                independence=independence
            )

            choice_key, theta_key = jr.split(theta_key)
            theta_samples = vmap(
                lambda key, obs: tree.map(
                    lambda leaf: leaf[0], #TODO: clean up
                    estim.sample_structured_posterior(
                        key,
                        jnp.expand_dims(obs, 0),
                        labels,
                        masks,
                        data_slices['theta'], #type: ignore
                        n_samples=1
                    )
                )
            )(jr.split(theta_key, n_simulations), flat_data['y'])
            # z_choice = jr.choice(choice_key, n_z, (n_simulations,))
            theta_samples = {
                'theta': theta_samples['theta'],
                'z': jr.choice(choice_key, theta_samples['z'], shape=(1,), axis=1)
            }
        else:
            theta_samples = prior_fn(1).sample(seed=theta_key, sample_shape=(n_simulations,))

        y_samples= simulator_fn(obs_key, theta_samples)

        z_data = {
            'theta': {
                'z': theta_samples['z'], #type: ignore
            },
            'y': {
                'theta': theta_samples['theta'], #type: ignore
                'obs': y_samples['obs']
            }
        }

        itr_key, key = jr.split(key)
        (
            train_iter,
            val_iter,
            labels,
            masks,
            _
        ) = structured_as_batch_iterators(
            itr_key,
            z_data,
            independence=independence
        )

        fit_key, key = jr.split(key)

        estim.fit(
            fit_key,
            train_iter,
            val_iter,
            labels,
            masks,
            n_iter=n_epochs
        )

        # simulate from p(z_vec|theta, y_vec)
        z_sim = z_data.copy()
        sim_key, key = jr.split(key)
        z_sim['y']['obs'] = vmap(
            lambda k: jr.choice(k, z_data['y']['obs'], shape=(n_z,)) # type: ignore
        )(jr.split(sim_key, n_simulations))
        # pad z to n_z
        z_sim['theta']['z'] = pad_multidim_event(
            z_sim['theta']['z'],
            1,
            (n_z,)
        )

        (
            flat_z_sim,
            labels,
            masks,
            z_sim_slices
        ) = flatten_structured(
            z_sim,
            independence=independence
        )

        sample_key, key = jr.split(key)
        z_vec = vmap(
            lambda key, obs: tree.map(
                lambda leaf: leaf[0], #TODO: clean up
                estim.sample_structured_posterior(
                    key,
                    jnp.expand_dims(obs, 0),
                    labels,
                    masks,
                    z_sim_slices['theta'],
                    n_samples=1
                )
            )
        )(jr.split(sample_key, n_simulations), flat_z_sim['y'])

        # fit p(theta,z_vec|y_vec)
        data = {
            'theta': {
                'z': z_vec['z'],
                'theta': z_sim['y']['theta']
            },
            'y': {
                'obs': z_sim['y']['obs']
            }
        }

        train_key, itr_key, key = jr.split(key, 3)
        (
            train_iter,
            val_iter,
            labels,
            masks,
            data_slices
        ) = structured_as_batch_iterators(
            itr_key,
            data,
            independence=independence
        )

        estim.fit(
            train_key,
            train_iter,
            val_iter,
            labels,
            masks,
            n_iter=n_epochs
        )

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
        data_slices['theta'],
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
