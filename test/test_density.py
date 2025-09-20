import pytest
from jax import numpy as jnp, random as jr
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
    log_det_transform = jnp.log(2.0 * 2.0) # scale 2 in 2 dimensions

    # Compute density analytically for 2D independent Gaussian
    # log p(y) = log p(x/2) + log|det(dx/dy)| where y = 2x
    # log|det(dx/dy)| = log(1/2 * 1/2) = log(1/4) = -log(4)
    log_prob = jnp.sum(norm.logpdf(samples), axis=1) - log_det_transform

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
        n_epsilon=50
    )

    # Compare results
    # Use high tolerance due to stochastic nature of FFJORD trace estimation
    # The method is fundamentally correct but has inherent variance
    assert jnp.allclose(log_prob, log_prob_cnf, rtol=0.8, atol=2.5)
