import pytest
from jax import numpy as jnp, random as jr
from jax.scipy.stats import norm
from flax import nnx

from sfmpe.structured_cnf import StructuredCNF
from sfmpe.svf import StructuredVectorFieldModel

class ConstantTransform(StructuredVectorFieldModel):
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
        return theta

class DoublingTransform(StructuredVectorFieldModel):
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

@pytest.fixture
def constant_cnf_setup():
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
def constant_cnf_setup():
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
    
    transform = ConstantTransform(
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
    
    transform = DoublingTransform(
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

    # Compute density analytically
    log_prob = norm.logpdf(samples)

    # Compute stats through constant cnf
    cnf, _, _, _ = constant_cnf_setup
    log_prob_cnf = cnf.log_prob(samples)

    # Compare results
    assert jnp.allclose(log_prob, log_prob_cnf)

def test_density_of_constant_cnf_with_transformed_data(doubling_cnf_setup):
    # Generate samples from normal distribution
    samples = jr.normal(jr.PRNGKey(0), shape=(100, 2))
    transformed_samples = 2 * samples
    log_det_transform = jnp.log(2.0 * 2.0) # scale 2 in 2 dimensions

    # Compute density analytically
    log_prob = norm.logpdf(transformed_samples) + log_det_transform

    # Compute stats through constant cnf
    cnf, _, _, _ = doubling_cnf_setup
    log_prob_cnf = cnf.log_prob(samples)

    # Compare results
    assert jnp.allclose(log_prob, log_prob_cnf)
