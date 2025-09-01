"""
Tests for PyTreeBijector class.
"""
import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import tree
from jaxtyping import PyTree, Array
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.pytree_bijector import (
    PyTreeBijector,
    create_bijector_from_distribution,
    create_manual_bijector_tree
)


class TestPyTreeBijector:
    """Test PyTreeBijector functionality."""
    
    def test_init(self):
        """Test PyTreeBijector initialization."""
        bijector_tree = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid()
        }
        example_tree = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([0.5, -0.5])
        }
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        assert pytree_bij.name == "pytree_bijector"
    
    def test_forward_transform(self):
        """Test forward transformation preserves PyTree structure."""
        bijector_tree = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid(),
            'c': {
                'nested': tfb.Exp()
            }
        }
        
        # Input PyTree
        x = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([0.5, -0.5]),
            'c': {
                'nested': jnp.array([0.0, 1.0])
            }
        }
        pytree_bij = PyTreeBijector(bijector_tree, x)
        
        # Forward transform
        y = pytree_bij.forward(x)
        
        # Check structure is preserved
        assert tree.structure(y) == tree.structure(x)
        
        # Check transformations
        assert jnp.allclose(y['a'], x['a'])  # Identity
        assert jnp.allclose(y['b'], tfb.Sigmoid().forward(x['b']))
        assert jnp.allclose(y['c']['nested'], tfb.Exp().forward(x['c']['nested']))
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        bijector_tree = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid()
        }
        
        x = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([0.2, 0.8])  # Valid sigmoid outputs
        }
        pytree_bij = PyTreeBijector(bijector_tree, x)
        
        # Forward then inverse should recover original
        y = pytree_bij.forward(x)
        x_recovered = pytree_bij.inverse(y)
        
        assert jnp.allclose(x_recovered['a'], x['a'])
        assert jnp.allclose(x_recovered['b'], x['b'], atol=1e-6)
    
    def test_log_det_jacobian(self):
        """Test log determinant Jacobian computation."""
        bijector_tree = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid()
        }
        
        x = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([0.5, -0.5])
        }
        pytree_bij = PyTreeBijector(bijector_tree, x)
        
        event_ndims = {
            'a': 0,
            'b': 0
        }
        
        # Compute forward log det jacobian
        fldj = pytree_bij.forward_log_det_jacobian(x, event_ndims)
        
        # Should be sum of individual log det jacobians
        expected_fldj = (
            tfb.Identity().forward_log_det_jacobian(x['a'], 0).sum() +
            tfb.Sigmoid().forward_log_det_jacobian(x['b'], 0).sum()
        )
        
        assert jnp.allclose(fldj, expected_fldj)
    
    def test_inverse_log_det_jacobian(self):
        """Test inverse log determinant Jacobian."""
        bijector_tree = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid()
        }
        
        y = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([0.2, 0.8])  # Valid sigmoid outputs
        }
        pytree_bij = PyTreeBijector(bijector_tree, y)
        
        event_ndims = {
            'a': 0,
            'b': 0
        }
        
        # Compute inverse log det jacobian
        ildj = pytree_bij.inverse_log_det_jacobian(y, event_ndims)
        
        # Should be sum of individual inverse log det jacobians
        expected_ildj = (
            tfb.Identity().inverse_log_det_jacobian(y['a'], 0).sum() +
            tfb.Sigmoid().inverse_log_det_jacobian(y['b'], 0).sum()
        )
        
        assert jnp.allclose(ildj, expected_ildj)
    
    def test_empty_tree(self):
        """Test with empty PyTree."""
        bijector_tree = {}
        x = {}
        pytree_bij = PyTreeBijector(bijector_tree, x)
        
        y = pytree_bij.forward(x)
        
        assert y == {}
    
    def test_nested_structure(self):
        """Test with deeply nested PyTree structure."""
        bijector_tree = {
            'level1': {
                'level2': {
                    'level3': tfb.Exp()
                },
                'other': tfb.Identity()
            }
        }
        
        x = {
            'level1': {
                'level2': {
                    'level3': jnp.array([0.0, 1.0])
                },
                'other': jnp.array([5.0])
            }
        }
        pytree_bij = PyTreeBijector(bijector_tree, x)
        
        y = pytree_bij.forward(x)
        x_recovered = pytree_bij.inverse(y)
        
        assert jnp.allclose(
            x_recovered['level1']['level2']['level3'],
            x['level1']['level2']['level3']
        )
        assert jnp.allclose(
            x_recovered['level1']['other'],
            x['level1']['other']
        )


class TestCreateBijectorFromDistribution:
    """Test creation of bijector from JointDistributionNamed."""
    
    def test_simple_joint_distribution(self):
        """Test with simple joint distribution."""
        prior = tfd.JointDistributionNamed({
            'a': tfd.Normal(0.0, 1.0),
            'b': tfd.Uniform(0.0, 1.0)
        })
        
        pytree_bij = create_bijector_from_distribution(prior)
        
        # Sample from prior
        key = jr.PRNGKey(42)
        samples = prior.sample(5, seed=key)
        
        # Transform samples
        transformed = pytree_bij.forward(samples)
        
        # Check structure is preserved
        assert tree.structure(transformed) == tree.structure(samples)
        
        # Inverse should recover samples
        recovered = pytree_bij.inverse(transformed)
        
        assert jnp.allclose(recovered['a'], samples['a'], atol=1e-5)
        assert jnp.allclose(recovered['b'], samples['b'], atol=1e-5)
    
    def test_complex_joint_distribution(self):
        """Test with more complex joint distribution."""
        prior = tfd.JointDistributionNamed({
            'positive': tfd.Gamma(2.0, 1.0),
            'bounded': tfd.Beta(2.0, 3.0),
            'normal': tfd.Normal(0.0, 2.0)
        })
        
        pytree_bij = create_bijector_from_distribution(prior)
        
        # Sample and transform
        key = jr.PRNGKey(123)
        samples = prior.sample(3, seed=key)
        transformed = pytree_bij.forward(samples)
        recovered = pytree_bij.inverse(transformed)
        
        # Check all keys are preserved
        assert set(recovered.keys()) == set(samples.keys())
        
        # Check values are recovered accurately
        for key in samples.keys():
            assert jnp.allclose(recovered[key], samples[key], atol=1e-4)


class TestCreateManualBijectorTree:
    """Test manual bijector tree creation."""
    
    def test_with_specifications(self):
        """Test with manual bijector specifications."""
        sample_tree = {
            'observations': jnp.array([1.0, 2.0, 3.0]),
            'parameters': jnp.array([0.5, 0.8])
        }
        
        bijector_specs = {
            'observations': tfb.Softplus(),  # For positive observations
            'parameters': tfb.Sigmoid()     # For bounded parameters
        }
        
        pytree_bij = create_manual_bijector_tree(sample_tree, bijector_specs)
        
        # Test transformations
        x = {
            'observations': jnp.array([0.5, 1.0, 1.5]),
            'parameters': jnp.array([0.0, -1.0])
        }
        
        y = pytree_bij.forward(x)
        x_recovered = pytree_bij.inverse(y)
        
        assert jnp.allclose(x_recovered['observations'], x['observations'])
        assert jnp.allclose(x_recovered['parameters'], x['parameters'], atol=1e-6)
    
    def test_default_identity(self):
        """Test that unspecified bijectors default to Identity."""
        sample_tree = {
            'specified': jnp.array([1.0]),
            'unspecified': jnp.array([2.0])
        }
        
        bijector_specs = {
            'specified': tfb.Exp()
        }
        
        pytree_bij = create_manual_bijector_tree(sample_tree, bijector_specs)
        
        x = {
            'specified': jnp.array([0.0]),
            'unspecified': jnp.array([5.0])
        }
        
        y = pytree_bij.forward(x)
        
        # Specified should be transformed
        assert jnp.allclose(y['specified'], jnp.exp(x['specified']))
        # Unspecified should be unchanged (Identity)
        assert jnp.allclose(y['unspecified'], x['unspecified'])
    
    def test_no_specifications(self):
        """Test with no bijector specifications (all Identity)."""
        sample_tree = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([3.0])
        }
        
        pytree_bij = create_manual_bijector_tree(sample_tree)
        
        x = sample_tree
        y = pytree_bij.forward(x)
        
        # All should be unchanged (Identity transformations)
        assert tree.all(tree.map(jnp.allclose, x, y))