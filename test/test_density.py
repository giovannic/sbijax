import pytest
from jax import numpy as jnp, random as jr, jit
from jax.scipy.stats import norm
from flax import nnx

from sfmpe.structured_cnf import StructuredCNF
from sfmpe.svf import StructuredVectorFieldModel

@pytest.fixture
def constant_cnf_setup():
    class DummyTransform(StructuredVectorFieldModel):
        """Dummy transform implementation for testing."""
        
        def __init__(self, config, n_labels, in_dim, value_dim, out_dim, rngs):
            super().__init__()
            self.config = config
            self.n_labels = n_labels
            self.in_dim = in_dim
            self.value_dim = value_dim
            self.out_dim = out_dim
            
        def __call__(
            self,
            context,
            context_label,
            context_index,
            context_mask,
            theta,
            theta_label,
            theta_index,
            theta_mask,
            cross_mask,
            time,
        ) -> jnp.ndarray:
            return jnp.zeros_like(theta)


    """Basic CNF setup for testing."""
    rngs = nnx.Rngs(0, base_dist=0)
    n_context = 5
    n_dim = 2
    
    config = {
        'latent_dim': 16,
        'label_dim': 8,
        'index_out_dim': 8,
        'n_ff': 2,
        'dropout': 0.0,
        'activation': 'relu'
    }
    
    transform = DummyTransform(
        config=config,
        n_labels=2,
        in_dim=n_context + n_dim,
        value_dim=n_dim,
        out_dim=n_dim,
        rngs=rngs
    )
    cnf = StructuredCNF(transform)
    return cnf, rngs, n_context, n_dim

@pytest.fixture
def doubling_cnf_setup():
    """Basic CNF setup for testing."""

    class Transform(StructuredVectorFieldModel):
        """Dummy transform implementation for testing."""
        
        def __init__(self, config, n_labels, in_dim, value_dim, out_dim, rngs):
            super().__init__()
            self.config = config
            self.n_labels = n_labels
            self.in_dim = in_dim
            self.value_dim = value_dim
            self.out_dim = out_dim
            
        def __call__(
            self,
            context,
            context_label,
            context_index,
            context_mask,
            theta,
            theta_label,
            theta_index,
            theta_mask,
            cross_mask,
            time,
        ) -> jnp.ndarray:
            return jnp.log(2) * theta


    rngs = nnx.Rngs(0, base_dist=0)
    n_context = 5
    n_dim = 2
    
    config = {
        'latent_dim': 16,
        'label_dim': 8,
        'index_out_dim': 8,
        'n_ff': 2,
        'dropout': 0.0,
        'activation': 'relu'
    }
    
    transform = Transform(
        config=config,
        n_labels=2,
        in_dim=n_context + n_dim,
        value_dim=n_dim,
        out_dim=n_dim,
        rngs=rngs
    )
    cnf = StructuredCNF(transform)
    return cnf, rngs, n_context, n_dim

def test_density_of_constant_cnf(constant_cnf_setup):
    # Generate samples from normal distribution
    samples = jr.normal(jr.PRNGKey(0), shape=(100, 2))

    # Compute density analytically for 2D independent Gaussian
    log_prob = jnp.sum(norm.logpdf(samples), axis=1)

    # Compute stats through constant cnf
    cnf, _, _, _ = constant_cnf_setup
    log_prob_cnf = cnf.log_prob(
        theta=samples,
        theta_label=jnp.zeros((100, 2)),
        theta_index=None,
        theta_mask=None,
        context=jnp.zeros((100, 2)),
        context_label=jnp.ones((100, 2)),
        context_index=None,
        context_mask=None,
        cross_mask=None
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
    cnf, _, _, _ = doubling_cnf_setup
    log_prob_cnf = cnf.log_prob(
        theta=samples,
        theta_label=jnp.zeros((100, 2)),
        theta_index=None,
        theta_mask=None,
        context=jnp.zeros((100, 2)),
        context_label=jnp.ones((100, 2)),
        context_index=None,
        context_mask=None,
        cross_mask=None,
        n_epsilon=20
    )

    # Compare results
    # Use moderate tolerance due to stochastic nature of FFJORD trace estimation
    assert jnp.allclose(log_prob, log_prob_cnf, rtol=0.1, atol=0.5)

def test_log_prob_jitability(constant_cnf_setup):
    """Test that log_prob can be jitted without tracing issues."""
    cnf, _, _, _ = constant_cnf_setup

    # Define a jitted version of log_prob
    @jit
    def jitted_log_prob(theta, context):
        return cnf.log_prob(
            theta=theta,
            theta_label=jnp.zeros_like(theta),
            theta_index=None,
            theta_mask=None,
            context=context,
            context_label=jnp.ones_like(context),
            context_index=None,
            context_mask=None,
            cross_mask=None
        )

    # Generate test data
    samples = jr.normal(jr.PRNGKey(0), shape=(10, 2))
    context = jnp.zeros((10, 2))

    # This should work without tracing errors
    log_prob_jitted = jitted_log_prob(samples, context)

    # Should return valid log probabilities
    assert log_prob_jitted.shape == (10,)
    assert jnp.all(jnp.isfinite(log_prob_jitted))
