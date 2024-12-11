import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import FMPE
from sbijax._src.nn.make_continuous_flow import CNF
from sbijax._src.infinite import index_data
from .nn.transformer.transformer import Transformer

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
    x, y = kwargs["x"], kwargs["y"]
    # return prior distribution of gaussian random walk at positions x and y
    prior = tfd.JointDistributionNamed(
        dict(
            x=tfd.Normal(x, 1.),
            y=tfd.Normal(y, 1.),
        ),
        batch_ndims=1,
    )
    return prior

def infinite_simulator_fn(seed, theta, **kwargs):
    t = kwargs['t']
    p_x = tfd.Normal(jnp.zeros_like(theta["x"]), 1.)
    p_y = tfd.Normal(jnp.zeros_like(theta["y"]), 1.)
    x_seed, y_seed = jr.split(seed)
    obs = jnp.stack([
        theta["x"] @ t + p_x.sample(seed=x_seed),
        theta["y"] @ t + p_y.sample(seed=y_seed),
    ], axis=-1)
    return obs.reshape(theta["x"].shape[0], -1)

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

    x= jnp.zeros((n, n_dim))

    index_map = {
        's': 0,
        't': 1
    }

    x_index = index_data(x, indices, index_map)

    assert x_index.shape == (n, n_dim, 3)

    # assert index is set
    assert jnp.all(x_index[..., 0, 0] == 0)
    assert jnp.all(x_index[..., 0, 1] == 1)
    assert jnp.all(x_index[..., 1, 2] == 2)

    # assert others are nan
    assert jnp.all(jnp.isnan(x_index[..., 0, 2]))
    assert jnp.all(jnp.isnan(x_index[..., 2, 0:1]))

def test_data_can_be_partially_indexed():
    n = 10
    n_dim = 4

    indices = {
        's': jnp.concatenate([
            jnp.full((n, 1), 0),
            jnp.full((n, 1), 1),
        ], axis=-1),
        't': jnp.full((n, 1), 2),
    }

    x= jnp.zeros((n, n_dim))

    index_map = {
        's': 0,
        't': 3
    }

    x_index = index_data(x, indices, index_map)

    assert x_index.shape == (n, n_dim, 3)

    # assert index is set
    assert jnp.all(x_index[..., 0, 0] == 0)
    assert jnp.all(x_index[..., 0, 1] == 1)
    assert jnp.all(x_index[..., 3, 2] == 2)

    # assert others are nan
    assert jnp.all(jnp.isnan(x_index[..., 0, 2]))
    assert jnp.all(jnp.isnan(x_index[..., 1:3, :]))
    assert jnp.all(jnp.isnan(x_index[..., 3, :2]))

#TODO: remove because unnecessary
def test_cnf_can_be_initialised_with_an_index():
    # create a continuous flow with a linear transform
    rngs = nnx.Rngs(0, base_dist=0)
    n = 10
    n_context = 5
    n_dim = 2

    indices = {
        's': jnp.concatenate([
            jnp.linspace(0, 1, n)[:, jnp.newaxis],
            jnp.linspace(0, 1, n)[:, jnp.newaxis],
        ], axis=-1),
        't': jnp.linspace(0, 1, n)[:, jnp.newaxis]
    }

    dummy_theta = jnp.zeros((n, n_dim))
    dummy_context = jnp.zeros((n, n_context))

    theta_index = index_data(
        dummy_theta,
        indices,
        {
            's': 0,
            't': 1
        }
    )
    context_index = index_data(
        dummy_context,
        indices,
        {
            's': 0,
            't': 1
        }
    )

    class DummyTransform(nnx.Module):
        def __init__(self, n_context, n_dim, rngs):
            self.linear = nnx.Linear(n_context, n_dim, rngs=rngs)
        def __call__(self, theta, time, context, **kwargs):
            return self.linear(context)

    transform = DummyTransform(n_context, n_dim, rngs)
    cnf = CNF(n_dim, transform)

    assert cnf.sample(
            rngs,
            dummy_context,
            theta_index=theta_index,
            context_index=context_index
        ).shape == (n, n_dim)
    assert cnf.vector_field(
            dummy_theta, # theta
            .5, # time
            dummy_context, # context
            theta_index=theta_index,
            context_index=context_index
        ).shape == (n, n_dim)

def test_infinite_parameters():
    tol: float = 1e-3
    y_observed = jnp.array([
        [3., -3.],
        [8., -8.],
        [15., -15.],
    ])
    y_indices = {
        'x': jnp.array([1, 2, 3])[:, jnp.newaxis],
        'y': jnp.array([-1, -2, -3])[:, jnp.newaxis],
        't': jnp.array([3, 4, 5])[:, jnp.newaxis],
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
        n_context_labels=2,
        context_index_dim=3,
        context_z_stats=(
            jnp.mean(y_observed, axis=0),
            jnp.std(y_observed, axis=0)
        ),
        n_theta_labels=2,
        theta_index_dim=2,
        rngs=rngs
    )

    estim = FMPE(
        (infinite_prior_fn, infinite_simulator_fn),
        nn,
        sample_context_index=sample_index,
        context_index_shape=(1,),
        sample_theta_index=sample_index,
        theta_index_shape=(3,),
    )
    data, params = None, {}
    for _ in range(2):
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            context_index=y_indices,
            theta_index=y_indices,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        params, _ = estim.fit(jr.PRNGKey(2), data=data, n_iter=2)
    posterior, _ = estim.sample_posterior(
        jr.PRNGKey(3),
        params,
        y_observed,
        observed_index=y_indices,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
    assert posterior.posterior.mean() == pytest.approx(1., tol) # type: ignore
