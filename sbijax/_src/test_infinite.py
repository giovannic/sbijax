import pytest
from jax import numpy as jnp
from jax import random as jr
from jax import tree
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import FMPE
from sbijax._src.nn.make_continuous_flow import CNF
from sbijax._src.infinite import index_theta
from .nn.transformer.transformer import Transformer
from .infinite import IndexMap

from flax import nnx

# TODO:
# - ~generate infinite dim prior~
# - ~generate infinite dim observations~
# - ~create indices (either continuous or set based)~
# - ~integrate indices into FMPE, SBI and NE~
# - ~when are these indices variably sized? During training?~
# - ~translate CNF to flax
# - ~integrate indices into transformer~
#   ~- init~
#   ~- vector_field~
#   ~- __call__~
# - generate realistic y_observed
# - generate appropriate test

def infinite_prior_fn(**kwargs):
    x, y = kwargs["x"][0], kwargs["y"][0]
    # return prior distribution of gaussian random walk at positions x and y
    prior = tfd.JointDistributionNamed(
        dict(
            x=tfd.Normal(x, 1.),
            y=tfd.Normal(y, 1.)
        ),
        batch_ndims=1,
    )
    return prior

# perform outer product of a batch of vectors
def batch_mul(x, y):
    return jnp.einsum('...i,...j->...ij', x, y)
    
def infinite_simulator_fn(seed, theta, **kwargs):
    t = kwargs['obs'][:, :, 0, 0]
    batch_size = theta["x"].shape[0]
    space_size = theta["x"].shape[1]
    time_size = t.shape[1]
    noise= tfd.Normal(
        jnp.zeros((batch_size, time_size, space_size)),
        1.
    )
    x_seed, y_seed = jr.split(seed)
    obs = jnp.stack([
        batch_mul(t, theta["x"]) + noise.sample(seed=x_seed),
        batch_mul(t, theta["y"]) + noise.sample(seed=y_seed),
    ], axis=-1)
    return {
        'obs': obs
    }

def sample_index(key, shape):
    points = jnp.cumsum(
        jr.lognormal(key, shape=(3,) + shape),
        axis=0
    )
    return {
        'x': points[0],
        'y': -points[1],
        't': points[2]
    }

def test_data_can_be_indexed_in_time_and_space():
    n = 10
    n_dim = 2

    indices = {
        's': jnp.concatenate([
            jnp.full((n, 1), 0),
            jnp.full((n, 1), 1),
        ], axis=-1),
        't': jnp.full((n, 1), 2),
    }

    theta= {
        'x': jnp.zeros((n, n_dim)),
        'y': jnp.zeros((n, n_dim))
    }

    index_map: IndexMap = {
        'x': 's',
        'y': 't'
    }

    x_index = index_theta(theta, indices, index_map)

    assert all(tree.map(lambda leaf: leaf.shape == (n, n_dim, 3), x_index))

    # assert index is set
    assert jnp.all(x_index['x'][..., 0] == 0)
    assert jnp.all(x_index['x'][..., 1] == 1)
    assert jnp.all(x_index['y'][..., 2] == 2)

    # assert others are nan
    assert jnp.all(jnp.isnan(x_index['x'][..., 2]))
    assert jnp.all(jnp.isnan(x_index['y'][..., 0:2]))

def test_data_can_be_partially_indexed():
    n = 10
    n_dim = 2

    indices = {
        's': jnp.concatenate([
            jnp.full((n, 1), 0),
            jnp.full((n, 1), 1),
        ], axis=-1)
    }

    theta= {
        'x': jnp.zeros((n, n_dim)),
        'y': jnp.zeros((n, n_dim))
    }

    index_map: IndexMap = {
        'x': 's'
    }

    x_index = index_theta(theta, indices, index_map)

    assert all(tree.map(lambda leaf: leaf.shape == (n, n_dim, 2), x_index).values())

    # assert index is set
    assert jnp.all(x_index['x'][..., 0] == 0)
    assert jnp.all(x_index['x'][..., 1] == 1)

    # assert others are nan
    assert jnp.all(jnp.isnan(x_index['y']))

def test_event_shape_can_be_indexed():
    n = 10
    event_shape = (3,)
    sample_shape = (4,)

    indices = {
        't': jnp.array([-1, 0, 1])[jnp.newaxis, :, jnp.newaxis]
    }

    theta= {
        'x': jnp.zeros((n,) + event_shape + sample_shape),
        'y': jnp.zeros((n,) + event_shape + sample_shape)
    }

    index_map: IndexMap = {
        'x': 't',
        'y': 't'
    }

    x_index = index_theta(theta, indices, index_map)

    assert all([
        leaf.shape == (n,) + event_shape + sample_shape + (1,)
        for leaf in tree.leaves(x_index)
    ])

    assert all([
        jnp.all(leaf == indices['t'])
        for leaf in tree.leaves(x_index)
    ])


def test_indexer_ignores_unreferenced_index():
    # create a continuous flow with a linear transform
    n = 10
    n_dim = 2

    indices = {
        's': jnp.concatenate([
            jnp.full((n, 1), 0),
            jnp.full((n, 1), 1),
        ], axis=-1),
        't': jnp.full((n, 1), 2),
    }

    theta= {
        'x': jnp.zeros((n, n_dim)),
        'y': jnp.zeros((n, n_dim))
    }

    index_map: IndexMap = {
        'x': 's'
    }

    x_index = index_theta(theta, indices, index_map)

    assert all(tree.map(lambda leaf: leaf.shape == (n, n_dim, 2), x_index))

def test_infinite_parameters():
    tol: float = 1e-3
    y_observed = jnp.array([
        [3., -3.],
        [8., -8.],
        [15., -15.],
    ])
    theta_indices = {
        'x': jnp.array([1., 2., 3.])[jnp.newaxis, :],
        'y': jnp.array([-1., -2., -3.])[jnp.newaxis, :],
    }
    y_indices = {
        'obs': jnp.array([3, 4, 5, 6])[
        jnp.newaxis, #batch
        :, #time
        jnp.newaxis, #distance
        jnp.newaxis #x/y
        ],
    }
    rngs = nnx.Rngs(0)
    config = {
        'latent_dim': 12,
        'label_dim': 2,
        'index_out_dim': 4,
        'n_encoder': 2,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }
    nn = Transformer(
        config,
        context_value_dim=2,
        n_context_labels=2,
        context_index_dim=3,
        theta_value_dim=2,
        n_theta_labels=2,
        theta_index_dim=2,
        rngs=rngs
    )
    theta_index_size= 3
    model = CNF(theta_dim=theta_index_size*2, transform=nn)

    estim = FMPE(
        (infinite_prior_fn, infinite_simulator_fn),
        model
    )
    data, params = None, {}
    for _ in range(2):
        # TODO: params will never be created!
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            context_index=y_indices,
            theta_index=theta_indices,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        print(tree.map(lambda leaf: leaf.shape, data))
        estim.fit(jr.PRNGKey(2), data=data, n_iter=2)
    #TODO: theta needs to be specified here too
    posterior, _ = estim.sample_posterior( 
        jr.PRNGKey(3),
        y_observed,
        observed_index=y_indices,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
    assert posterior.posterior.mean() == pytest.approx(1., tol) # type: ignore
