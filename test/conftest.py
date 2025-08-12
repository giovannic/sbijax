import pytest
from tensorflow_probability.substrates.jax import distributions as tfd
from jax import numpy as jnp

def prior_fn():
    prior = tfd.JointDistributionNamed(
        dict(
            theta=tfd.Normal(jnp.zeros(2), 1.0),
        ),
        batch_ndims=0,
    )
    return prior

def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["theta"]), 1.0)
    y = theta["theta"] + p.sample(seed=seed)
    return y

@pytest.fixture
def prior_simulator_tuple():
    return (prior_fn, simulator_fn)
