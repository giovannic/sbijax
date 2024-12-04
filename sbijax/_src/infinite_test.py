import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import FMPE
from sbijax.nn import make_cnf

# TODO:
# - ~generate infinite dim prior~
# - ~generate infinite dim observations~
# - ~create indices (either continuous or set based)~
# - ~integrate indices into FMPE, SBI and NE~
# - ~when are these indices variably sized? During training?~
# - integrate indices into transformer
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
    return jnp.stack([
        t * theta["x"] + p_x.sample(seed=seed),
        t * theta["y"] + p_y.sample(seed=seed),
    ], axis=-1)

def sample_index(key, shape):
    points = jnp.cumsum(jr.lognormal(key, shape=(3,) + shape))
    return {
        'x': points[0],
        'y': -points[1],
        't': points[2]
    }

def test_infinte_parameters():
    tol: float = 1e-3
    y_observed = jnp.array([
        [3., -3.],
        [8., -8.],
        [15., -15.],
    ])
    y_indices = {
        'x': jnp.array([1, 2, 3]),
        'y': jnp.array([-1, -2, -3]),
        't': jnp.array([3, 4, 5]),
    }
    estim = FMPE(
        (infinite_prior_fn, infinite_simulator_fn),
        make_cnf(2),
        sample_index=sample_index,
        index_shape=(3,)
    )
    data, params = None, {}
    for _ in range(2):
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
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
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
    assert posterior.posterior.mean() == pytest.approx(1., tol) # type: ignore
