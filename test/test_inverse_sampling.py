"""Tests for inverse sampling functionality in CNF."""

import pytest
import jax.numpy as jnp
from flax import nnx

from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transform import Transform

class DummyTransform(Transform):
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
        """Return stable vector field that contracts toward zero."""
        return jnp.full(theta.shape, 0.1)  # Contraction rate


@pytest.fixture
def basic_cnf_setup():
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
    cnf = CNF(transform)
    return cnf, rngs, n_context, n_dim

def test_cnf_direction_parameter_validation(basic_cnf_setup):
    """Test that CNF raises error for invalid direction parameter."""
    cnf, rngs, n_context, n_dim = basic_cnf_setup
    n_samples = 10
    
    # Mock context and other parameters
    context = jnp.zeros((1, n_context))
    context_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    context_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    context_mask = jnp.ones((1, 1, 1))
    theta_shape = (1, n_dim)
    theta_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    theta_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    theta_mask = jnp.ones((1, 1, 1))
    cross_mask = jnp.ones((1, 1, 1)) # fake
    
    with pytest.raises(ValueError, match="Unknown direction"):
        cnf.sample(
            rngs=rngs,
            context=context,
            context_label=context_label,
            context_index=context_index,
            context_mask=context_mask,
            theta_shape=theta_shape,
            theta_label=theta_label,
            theta_index=theta_index,
            theta_mask=theta_mask,
            cross_mask=cross_mask,
            sample_size=n_samples,
            direction='invalid'
        )

def test_cnf_backward_sampling_shape_consistency(basic_cnf_setup):
    """Test that backward sampling produces correct shapes."""
    cnf, rngs, n_context, n_dim = basic_cnf_setup
    n_samples = 10
    
    # Mock context and other parameters
    context = jnp.zeros((1, n_context))
    context_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    context_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    context_mask = jnp.ones((1, 1, 1))
    theta_shape = (1, n_dim)
    theta_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    theta_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    theta_mask = jnp.ones((1, 1, 1))
    cross_mask = jnp.ones((1, 1, 1)) # fake
   
    # Test backward sampling
    backward_samples = cnf.sample(
        rngs=rngs,
        context=context,
        context_label=context_label,
        context_index=context_index,
        context_mask=context_mask,
        theta_shape=theta_shape,
        theta_label=theta_label,
        theta_index=theta_index,
        theta_mask=theta_mask,
        cross_mask=cross_mask,
        sample_size=n_samples,
        direction='backward'
    )
    
    # Check shape consistency
    assert backward_samples.shape == (n_samples, 1, n_dim)
    assert backward_samples.dtype == jnp.float32

def test_cnf_integration_is_finite(basic_cnf_setup):
    """test that cnf integration bounds are set correctly for directions."""
    cnf, rngs, n_context, n_dim = basic_cnf_setup
    
    # test that the time bounds are correctly set internally
    # this is more of a unit test for the implementation logic
    
    # mock context and other parameters
    context = jnp.zeros((1, n_context))
    context_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    context_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    context_mask = jnp.ones((1, 1, 1))
    theta_shape = (1, n_dim)
    theta_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    theta_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    theta_mask = jnp.ones((1, 1, 1))
    cross_mask = jnp.ones((1, 1, 1)) # fake
   
    # test forward sampling (should not raise errors)
    forward_samples = cnf.sample(
        rngs=rngs,
        context=context,
        context_label=context_label,
        context_index=context_index,
        context_mask=context_mask,
        theta_shape=theta_shape,
        theta_label=theta_label,
        theta_index=theta_index,
        theta_mask=theta_mask,
        cross_mask=cross_mask,
        sample_size=2,
        direction='forward'
    )
    # if successful, check finite results
    assert jnp.all(jnp.isfinite(forward_samples))

def test_cnf_back_integration_is_finite(basic_cnf_setup):
    """test that cnf integration bounds are set correctly for directions."""
    cnf, rngs, n_context, n_dim = basic_cnf_setup
    
    # test that the time bounds are correctly set internally
    # this is more of a unit test for the implementation logic
    
    # mock context and other parameters
    context = jnp.zeros((1, n_context))
    context_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    context_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    context_mask = jnp.ones((1, 1, 1))
    theta_shape = (1, n_dim)
    theta_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    theta_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    theta_mask = jnp.ones((1, 1, 1))
    cross_mask = jnp.ones((1, 1, 1)) # fake

    # test backward sampling (should not raise errors about direction)
    backward_samples = cnf.sample(
        rngs=rngs,
        context=context,
        context_label=context_label,
        context_index=context_index,
        context_mask=context_mask,
        theta_shape=theta_shape,
        theta_label=theta_label,
        theta_index=theta_index,
        theta_mask=theta_mask,
        cross_mask=cross_mask,
        sample_size=2,
        direction='backward'
    )
    # if successful, check finite results
    print(backward_samples)
    assert jnp.all(jnp.isfinite(backward_samples))

def test_sample_size_parameter_consistency(basic_cnf_setup):
    """Test that sample_size parameter works consistently for both directions."""
    cnf, rngs, n_context, n_dim = basic_cnf_setup
    
    # Mock context and other parameters
    context = jnp.zeros((1, n_context))
    context_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    context_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    context_mask = jnp.ones((1, 1, 1))
    theta_shape = (1, n_dim)
    theta_label = jnp.ones((1, 1, 1), dtype=jnp.int32)
    theta_index = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    theta_mask = jnp.ones((1, 1, 1))
    cross_mask = jnp.ones((1, 1, 1)) # fake
   
    sample_sizes = [1, 5, 10]
    
    for sample_size in sample_sizes:
        # Test forward sampling
        forward_samples = cnf.sample(
            rngs=rngs,
            context=context,
            context_label=context_label,
            context_index=context_index,
            context_mask=context_mask,
            theta_shape=theta_shape,
            theta_label=theta_label,
            theta_index=theta_index,
            theta_mask=theta_mask,
            cross_mask=cross_mask,
            sample_size=sample_size,
            direction='forward'
        )
        
        # Test backward sampling
        backward_samples = cnf.sample(
            rngs=rngs,
            context=context,
            context_label=context_label,
            context_index=context_index,
            context_mask=context_mask,
            theta_shape=theta_shape,
            theta_label=theta_label,
            theta_index=theta_index,
            theta_mask=theta_mask,
            cross_mask=cross_mask,
            sample_size=sample_size,
            direction='backward'
        )
        
        # Check shape consistency
        assert forward_samples.shape == (sample_size, 1, n_dim)
        assert backward_samples.shape == (sample_size, 1, n_dim)

def test_cnf_can_recover_initial_parameters():
    """Test that CNF can recover initial parameters."""
    pass
