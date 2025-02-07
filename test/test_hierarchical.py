import pytest
from jax import numpy as jnp, random as jr, vmap, tree
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.fmpe import SFMPE
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.mlp import MLPVectorField
from sfmpe.util.dataloader import (
    flatten_structured,
    flat_as_batch_iterators,
    pad_multidim_event,
    combine_data
)

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
    # nn = MLPVectorField(
        # config,
        # n_labels=3,
        # in_dim=2,
        # value_dim=1,
        # out_dim=1,
        # rngs=rngs
    # )

    model = CNF(transform=nn)

    estim = SFMPE(model)

    flat_data, data_slices = (None, {})
    train_data = None
    for _ in range(n_rounds):
        # Fit p(z|theta, y)
        theta_key, obs_key, key = jr.split(key, 3)
        if flat_data is not None and data_slices:
            choice_key, theta_key = jr.split(theta_key)
            theta_samples = vmap(
                lambda key, obs: tree.map(
                    lambda leaf: leaf[0], #TODO: clean up
                    estim.sample_structured_posterior(
                        key,
                        jnp.expand_dims(obs, 0),
                        flat_data['labels'], #type: ignore
                        data_slices['theta'], #type: ignore
                        masks=flat_data['masks'], #type: ignore
                        n_samples=1
                    )
                )
            )(jr.split(theta_key, n_simulations), flat_data['data']['y'])
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

        z_flat, _ = flatten_structured(
            z_data,
            independence=independence
        )

        if train_data is None:
            train_data = z_flat
        else:
            train_data = combine_data(train_data, z_flat)

        #TODO: remove
        # del train_data['masks']

        itr_key, key = jr.split(key)
        train_iter, val_iter = flat_as_batch_iterators(
            itr_key,
            train_data
        )

        fit_key, key = jr.split(key)

        estim.fit(
            fit_key,
            train_iter,
            val_iter,
            n_iter=n_epochs
        )

        # simulate from p(z_vec|theta, y_vec)
        #TODO: this seems broken
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
            z_sim_slices
        ) = flatten_structured(
            z_sim,
            independence=independence
        )

        sample_key, key = jr.split(key)
        # print(flat_z_sim['data']['y'][:1])
        # print(estim.sample_structured_posterior(
                    # sample_key,
                    # flat_z_sim['data']['y'][:1],
                    # flat_z_sim['labels'],
                    # z_sim_slices['theta'],
                    # masks=flat_z_sim['masks'],
                    # n_samples=10
                # ))


        z_vec = vmap(
            lambda key, obs: tree.map(
                lambda leaf: leaf[0], #TODO: clean up
                estim.sample_structured_posterior(
                    key,
                    jnp.expand_dims(obs, 0),
                    flat_z_sim['labels'],
                    z_sim_slices['theta'],
                    masks=flat_z_sim['masks'],
                    n_samples=1
                )
            )
        )(jr.split(sample_key, n_simulations), flat_z_sim['data']['y'])

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
        print(tree.map(lambda leaf: leaf[:10], data))

        flat_data, data_slices = flatten_structured(
            data,
            independence=independence
        )

        train_data = combine_data(train_data, flat_data)

        train_key, itr_key, key = jr.split(key, 3)
        train_iter, val_iter = flat_as_batch_iterators(
            itr_key,
            train_data
        )

        estim.fit(
            train_key,
            train_iter,
            val_iter,
            n_iter=n_epochs
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

    obs_flat, _ = flatten_structured(
        { 'theta': theta_truth, 'y': y_observed },
        independence=independence
    )
    posterior = estim.sample_structured_posterior( 
        jr.PRNGKey(3),
        obs_flat['data']['y'],
        flat_data['labels'], #type: ignore
        data_slices['theta'], #type: ignore
        masks=flat_data['masks'], #type: ignore
        n_samples=n_post_samples,
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
    print('theta')
    print(theta_truth['theta']) # type: ignore
    print(theta_hat)
    print(sbijax_theta_hat)
    print('z')
    print(theta_truth['z']) # type: ignore
    print(z_hat)
    print(sbijax_z_hat)
    assert sbijax_theta_hat[None,...] == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert theta_hat == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert sbijax_z_hat[None,...] == pytest.approx(theta_truth['z'], tol) # type: ignore
    assert z_hat == pytest.approx(theta_truth['z'], tol) # type: ignore
    assert False
