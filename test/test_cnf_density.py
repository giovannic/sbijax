import pytest
from jax import numpy as jnp, random as jr
from jax.scipy.stats import norm
from flax import nnx

from sfmpe.cnf import CNF
from sfmpe.nn.mlp import VectorFieldModel


@pytest.fixture
def constant_cnf_setup():
    class DummyTransform(VectorFieldModel):
        """Dummy transform implementation for testing."""

        def __init__(self, n_context, n_dim, rngs):
            super().__init__()
            self.n_context = n_context
            self.n_dim = n_dim

        def __call__(self, theta, time, context) -> jnp.ndarray:
            return jnp.zeros_like(theta)

    """Basic CNF setup for testing."""
    rngs = nnx.Rngs(0, base_dist=0)
    n_context = 5
    n_dim = 2

    transform = DummyTransform(n_context, n_dim, rngs)
    cnf = CNF(transform, rngs=rngs)
    return cnf, rngs, n_context, n_dim

@pytest.fixture
def doubling_cnf_setup():
    """Basic CNF setup for testing."""

    class Transform(VectorFieldModel):
        """Dummy transform implementation for testing."""

        def __init__(self, n_context, n_dim, rngs):
            super().__init__()
            self.n_context = n_context
            self.n_dim = n_dim

        def __call__(self, theta, time, context) -> jnp.ndarray:
            return jnp.log(2) * theta

    rngs = nnx.Rngs(0, base_dist=0)
    n_context = 5
    n_dim = 2

    transform = Transform(n_context, n_dim, rngs)
    cnf = CNF(transform, rngs=rngs)
    return cnf, rngs, n_context, n_dim

def test_density_of_constant_cnf(constant_cnf_setup):
    # Generate samples from normal distribution
    samples = jr.normal(jr.PRNGKey(0), shape=(100, 2))

    # Compute density analytically for 2D independent Gaussian
    log_prob = jnp.sum(norm.logpdf(samples), axis=1)

    # Compute stats through constant cnf
    cnf, _, n_context, _ = constant_cnf_setup
    log_prob_cnf = cnf.log_prob(
        theta=samples,
        context=jnp.zeros((100, n_context))
    )

    # Compare results
    assert jnp.allclose(log_prob, log_prob_cnf)

def test_density_of_constant_cnf_with_transformed_data(doubling_cnf_setup):
    # Generate samples from normal distribution
    samples = jr.normal(jr.PRNGKey(0), shape=(100, 2))

    # Compute density analytically for 2D independent Gaussian
    # The doubling transform produces y ~ N(0, 4) from x ~ N(0, 1)
    # So log p(y) should be computed directly as N(y; 0, 2^2)
    log_prob = jnp.sum(norm.logpdf(samples, scale=2), axis=1)

    # Compute stats through constant cnf
    cnf, _, n_context, _ = doubling_cnf_setup
    log_prob_cnf = cnf.log_prob(
        theta=samples,
        context=jnp.zeros((100, n_context)),
        n_epsilon=20
    )

    # Compare results
    # Use moderate tolerance due to stochastic nature of FFJORD trace estimation
    assert jnp.allclose(log_prob, log_prob_cnf, rtol=0.1, atol=0.5)