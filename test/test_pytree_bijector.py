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
    create_bijector_tree,
    create_zscaling_bijector_tree
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
    
    def test_forward_dtype(self):
        """Test forward_dtype method returns correct dtype structure."""
        # Create a JointDistributionNamed to get realistic dtype structure
        prior = tfd.JointDistributionNamed({
            'float_param': tfd.Normal(0.0, 1.0),
            'bounded_param': tfd.Uniform(0.0, 1.0)
        })
        
        # Create PyTreeBijector from distribution
        pytree_bij = create_bijector_from_distribution(prior)
        
        # Get output dtype structure from the bijector
        output_dtype = pytree_bij.forward_dtype(prior.dtype)
        
        # Should preserve the structure and dtypes
        assert tree.structure(output_dtype) == tree.structure(prior.dtype)
        assert tree.leaves(output_dtype) == tree.leaves(prior.dtype)




class TestCreateBijectorTree:
    """Test bijector tree creation."""
    
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
        
        pytree_bij = create_bijector_tree(sample_tree, bijector_specs)
        
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
        
        pytree_bij = create_bijector_tree(sample_tree, bijector_specs)
        
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
        
        pytree_bij = create_bijector_tree(sample_tree)
        
        x = sample_tree
        y = pytree_bij.forward(x)
        
        # All should be unchanged (Identity transformations)
        assert tree.all(tree.map(jnp.allclose, x, y))


class TestCreateZScalingBijectorTree:
    """Test Z-scaling bijector tree creation."""
    
    def test_custom_bijectors_with_zscaling(self):
        """Test custom bijector specifications with Z-scaling."""
        # Create sample data with known structure
        sample_tree = {
            'observations': jnp.array([1.0, 2.0, 3.0]),
            'parameters': jnp.array([0.5, 0.8])
        }
        
        # Generate representative data with different statistics
        representative_data = {
            'observations': jnp.array([[10.0, 20.0, 30.0]] * 500),  # Mean ~20
            'parameters': jnp.array([[0.1, 0.9]] * 500)           # Mean ~0.5
        }
        
        bijector_specs = {
            'observations': tfb.Softplus(),  # For positive values
            'parameters': tfb.Sigmoid()     # For [0,1] bounded values
        }
        
        pytree_bij = create_zscaling_bijector_tree(
            sample_tree, representative_data, bijector_specs
        )
        
        # Test transformations
        x = {
            'observations': jnp.array([5.0, 10.0, 15.0]),
            'parameters': jnp.array([0.2, 0.8])
        }
        
        y = pytree_bij.forward(x)
        x_recovered = pytree_bij.inverse(y)
        
        assert jnp.allclose(x_recovered['observations'], x['observations'])
        assert jnp.allclose(x_recovered['parameters'], x['parameters'], atol=1e-6)
    
    def test_mixed_specified_unspecified(self):
        """Test mix of specified and unspecified bijectors."""
        sample_tree = {
            'specified': jnp.array([1.0, 2.0]),
            'unspecified': jnp.array([3.0, 4.0])
        }
        
        # Representative data with non-zero mean/non-unit std
        representative_data = {
            'specified': jnp.array([[5.0, 6.0]] * 300),    # Will be exp'd then Z-scaled
            'unspecified': jnp.array([[10.0, 12.0]] * 300)  # Identity then Z-scaled
        }
        
        bijector_specs = {
            'specified': tfb.Exp()
        }
        
        pytree_bij = create_zscaling_bijector_tree(
            sample_tree, representative_data, bijector_specs
        )
        
        x = {
            'specified': jnp.array([0.0, 1.0]),  # Will be exp'd to [1, e]
            'unspecified': jnp.array([7.0, 8.0])  # Identity transformation
        }
        
        y = pytree_bij.forward(x)
        x_recovered = pytree_bij.inverse(y)
        
        # Check round-trip consistency
        assert jnp.allclose(x_recovered['specified'], x['specified'])
        assert jnp.allclose(x_recovered['unspecified'], x['unspecified'])
        
        # Check that 'unspecified' was only Z-scaled (no other transformation)
        # After Z-scaling, should have different values than input
        assert not jnp.allclose(y['unspecified'], x['unspecified'])
    
    def test_pure_zscaling_no_specs(self):
        """Test pure Z-scaling with no custom bijector specifications."""
        sample_tree = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([3.0])
        }
        
        # Representative data with known statistics
        representative_data = {
            'a': jnp.array([[10.0, 20.0]] * 400),  # Mean [10, 20], std varies
            'b': jnp.array([[5.0]] * 400)          # Mean 5.0
        }
        
        # No bijector specs - should be pure Z-scaling
        pytree_bij = create_zscaling_bijector_tree(
            sample_tree, representative_data
        )
        
        x = sample_tree
        y = pytree_bij.forward(x)
        x_recovered = pytree_bij.inverse(y)
        
        # Should recover original through Identity + Z-scaling
        assert jnp.allclose(x_recovered['a'], x['a'])
        assert jnp.allclose(x_recovered['b'], x['b'])
    
    def test_zscaling_statistics_validation(self):
        """Test that the Z-scaling component produces correct statistics."""
        sample_tree = {
            'param1': jnp.array([0.0]),
            'param2': jnp.array([1.0])
        }
        
        # Generate representative data with proper batch shapes
        key = jr.PRNGKey(42)
        key1, key2 = jr.split(key)
        representative_data = {
            'param1': jr.normal(key1, (1000, 1)) + 5.0,  # Mean=5, std=1
            'param2': jr.uniform(key2, (1000, 1)) * 10    # Range [0, 10]
        }
        
        pytree_bij = create_zscaling_bijector_tree(
            sample_tree, representative_data
        )
        
        # Generate test samples with proper batch shapes
        key3, key4 = jr.split(jr.PRNGKey(999))
        test_samples = {
            'param1': jr.normal(key3, (2000, 1)) + 5.0,   # Same distribution
            'param2': jr.uniform(key4, (2000, 1)) * 10    # Same distribution
        }
        
        transformed = pytree_bij.forward(test_samples)
        
        # Check that Z-scaling worked (approximately normalized)
        param1_mean = jnp.mean(transformed['param1'])
        param1_std = jnp.std(transformed['param1'])
        param2_mean = jnp.mean(transformed['param2'])
        param2_std = jnp.std(transformed['param2'])
        
        assert jnp.abs(param1_mean) < 0.2
        assert jnp.abs(param1_std - 1.0) < 0.2
        assert jnp.abs(param2_mean) < 0.2
        assert jnp.abs(param2_std - 1.0) < 0.2


class TestPyTreeBijectorVariableEventSizes:
    """Test PyTreeBijector with variable event sizes at runtime."""
    
    def test_basic_variable_event_sizes(self):
        """Test bijector instantiated with one event size, used with different sizes."""
        bijector_tree = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid()
        }
        
        # Initialize with shape (batch=2, event_size=3, value_size=2)
        example_tree = {
            'a': jnp.array([[[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
                           [[3.0, 4.0], [3.5, 4.5], [4.0, 5.0]]]),
            'b': jnp.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                           [[0.15, 0.25], [0.35, 0.45], [0.55, 0.65]]])
        }
        
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        # Test with smaller event size (event_size=2) using slicing
        x_small = {
            'a': example_tree['a'][:, :2, :],  # (2, 2, 2)
            'b': example_tree['b'][:, :2, :]   # (2, 2, 2)
        }
        
        y_small = pytree_bij.forward(x_small)
        x_small_recovered = pytree_bij.inverse(y_small)
        
        assert jnp.allclose(y_small['a'], x_small['a'])  # Identity
        assert jnp.allclose(y_small['b'], tfb.Sigmoid().forward(x_small['b']))
        assert jnp.allclose(x_small_recovered['a'], x_small['a'])
        assert jnp.allclose(x_small_recovered['b'], x_small['b'], atol=1e-6)
        
        # Test with larger event size (event_size=5) using repeat
        x_large = {
            'a': jnp.repeat(example_tree['a'], 5, axis=1)[:, :5, :],  # (2, 5, 2)
            'b': jnp.repeat(example_tree['b'], 5, axis=1)[:, :5, :]   # (2, 5, 2)
        }
        
        y_large = pytree_bij.forward(x_large)
        x_large_recovered = pytree_bij.inverse(y_large)
        
        assert jnp.allclose(y_large['a'], x_large['a'])  # Identity
        assert jnp.allclose(y_large['b'], tfb.Sigmoid().forward(x_large['b']))
        assert jnp.allclose(x_large_recovered['a'], x_large['a'])
        assert jnp.allclose(x_large_recovered['b'], x_large['b'], atol=1e-6)
    
    def test_different_event_sizes_per_key(self):
        """Test with different event_size changes for each key."""
        bijector_tree = {
            'small': tfb.Exp(),
            'medium': tfb.Softplus(),
            'large': tfb.Sigmoid()
        }
        
        # Initialize with baseline event sizes
        example_tree = {
            'small': jnp.array([[[1.0]], [[2.0]]]),          # (2, 1, 1)
            'medium': jnp.array([[[0.5, 1.0]], [[1.5, 2.0]]]), # (2, 1, 2)  
            'large': jnp.array([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]) # (2, 1, 3)
        }
        
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        # Test with modified event sizes
        x_modified = {
            'small': jnp.repeat(example_tree['small'], 3, axis=1),    # (2, 3, 1)
            'medium': example_tree['medium'][:, :1, :1],              # (2, 1, 1) - smaller
            'large': jnp.repeat(example_tree['large'], 2, axis=1)[:, :2, :] # (2, 2, 3)
        }
        
        y = pytree_bij.forward(x_modified)
        x_recovered = pytree_bij.inverse(y)
        
        # Check transformations
        assert jnp.allclose(y['small'], jnp.exp(x_modified['small']))
        assert jnp.allclose(y['medium'], tfb.Softplus().forward(x_modified['medium']))
        assert jnp.allclose(y['large'], tfb.Sigmoid().forward(x_modified['large']))
        
        # Check recovery
        assert jnp.allclose(x_recovered['small'], x_modified['small'], atol=1e-5)
        assert jnp.allclose(x_recovered['medium'], x_modified['medium'], atol=1e-5)
        assert jnp.allclose(x_recovered['large'], x_modified['large'], atol=1e-5)
    
    def test_log_det_jacobian_variable_event_sizes(self):
        """Test log-det-jacobian with different event sizes."""
        bijector_tree = {
            'param1': tfb.Identity(),
            'param2': tfb.Sigmoid()
        }
        
        # Initialize with (1, 2, 2)
        example_tree = {
            'param1': jnp.array([[[1.0, 2.0], [3.0, 4.0]]]),
            'param2': jnp.array([[[0.2, 0.3], [0.4, 0.5]]])
        }
        
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        # Test with smaller size (1, 1, 2)
        x_small = {
            'param1': example_tree['param1'][:, :1, :],
            'param2': example_tree['param2'][:, :1, :]
        }
        
        event_ndims_small = {
            'param1': 0,
            'param2': 0
        }
        
        fldj_small = pytree_bij.forward_log_det_jacobian(x_small, event_ndims_small)
        assert jnp.isfinite(fldj_small)
        
        # Test with larger size (1, 4, 2)
        x_large = {
            'param1': jnp.repeat(example_tree['param1'], 2, axis=1),
            'param2': jnp.repeat(example_tree['param2'], 2, axis=1)
        }
        
        event_ndims_large = {
            'param1': 0,
            'param2': 0
        }
        
        fldj_large = pytree_bij.forward_log_det_jacobian(x_large, event_ndims_large)
        assert jnp.isfinite(fldj_large)
    
    def test_zscaling_variable_event_sizes(self):
        """Test Z-scaling bijectors with different event sizes at runtime."""
        # Sample tree with (1, 2, 1) shape  
        sample_tree = {
            'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1)
            'b': jnp.array([[[0.5], [0.6]]])   # (1, 2, 1)
        }
        
        # Generate representative data with (100, 2, 1) shape
        key = jr.PRNGKey(42)
        representative_data = {
            'a': jr.normal(key, (100, 2, 1)),
            'b': jnp.clip(jr.uniform(jr.PRNGKey(1), (100, 2, 1)), 0.01, 0.99)
        }
        
        bijector_specs = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid()
        }
        
        pytree_bij = create_zscaling_bijector_tree(
            sample_tree, representative_data, bijector_specs
        )
        
        # Test with larger sample but different event size to verify normalization
        # Create a prior with different event size for testing
        large_sample = tree.map(
            lambda leaf: leaf.repeat(2, axis=1),
            representative_data
        )
        
        transformed_large = pytree_bij.forward(large_sample)
        
        # Check that Z-scaling produces approximately normalized data
        for key_name in transformed_large.keys():
            mean_val = jnp.mean(transformed_large[key_name])
            std_val = jnp.std(transformed_large[key_name])
            assert jnp.abs(mean_val) < 0.2, f"Mean should be ~0, got {mean_val}"
            assert jnp.abs(std_val - 1.0) < 0.2, f"Std should be ~1, got {std_val}"
        
        # Test with smaller event size (batch, 1, 1)
        test_small = {
            'a': large_sample['a'][:10, :1, :],  # (10, 1, 1)
            'b': large_sample['b'][:10, :1, :]   # (10, 1, 1)
        }
        
        transformed_small = pytree_bij.forward(test_small)
        recovered_small = pytree_bij.inverse(transformed_small)
        
        assert jnp.allclose(recovered_small['a'], test_small['a'], atol=1e-4)
        assert jnp.allclose(recovered_small['b'], test_small['b'], atol=1e-4)
        
        # Test with larger event size (batch, 3, 1) - use subset of the large sample
        large_sample_subset = tree.map(lambda x: x[:10], large_sample)  # (10, 4, 1)
        transformed_large_events = pytree_bij.forward(large_sample_subset)
        recovered_large_events = pytree_bij.inverse(transformed_large_events)
        
        assert jnp.allclose(recovered_large_events['a'], large_sample_subset['a'], atol=1e-4)
        assert jnp.allclose(recovered_large_events['b'], large_sample_subset['b'], atol=1e-4)
    
    def test_manual_zscaling_variable_event_sizes(self):
        """Test manual Z-scaling with runtime event size changes."""
        # Initialize with (2, 2, 1) shape
        sample_tree = {
            'observations': jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
            'parameters': jnp.array([[[0.5], [0.6]], [[0.7], [0.8]]])
        }
        
        # Representative data with same initial shape
        representative_data = {
            'observations': jr.normal(jr.PRNGKey(0), (100, 2, 1)) + 10.0,
            'parameters': jnp.clip(jr.uniform(jr.PRNGKey(1), (100, 2, 1)), 0.01, 0.99)
        }
        
        bijector_specs = {
            'observations': tfb.Softplus(),
            'parameters': tfb.Sigmoid()
        }
        
        pytree_bij = create_zscaling_bijector_tree(
            sample_tree, representative_data, bijector_specs
        )
        
        # Test normalization with larger sample
        large_test_data = {
            'observations': jr.normal(jr.PRNGKey(2), (1000, 2, 1)) + 10.0,
            'parameters': jnp.clip(jr.uniform(jr.PRNGKey(3), (1000, 2, 1)), 0.01, 0.99)
        }
        
        transformed_large = pytree_bij.forward(large_test_data)
        
        # Check normalization
        for key_name in transformed_large.keys():
            mean_val = jnp.mean(transformed_large[key_name])
            std_val = jnp.std(transformed_large[key_name])
            assert jnp.abs(mean_val) < 0.2, f"Mean should be ~0, got {mean_val}"
            assert jnp.abs(std_val - 1.0) < 0.2, f"Std should be ~1, got {std_val}"
        
        # Test with smaller size (1, 1, 1)
        x_small = {
            'observations': jnp.array([[[5.0]]]),
            'parameters': jnp.array([[[0.2]]])
        }
        
        y_small = pytree_bij.forward(x_small)
        x_small_recovered = pytree_bij.inverse(y_small)
        
        assert jnp.allclose(x_small_recovered['observations'], x_small['observations'], atol=1e-5)
        assert jnp.allclose(x_small_recovered['parameters'], x_small['parameters'], atol=1e-5)
        
        # Test with larger size (2, 3, 1)
        x_large = {
            'observations': jnp.array([[[5.0], [7.0], [9.0]], [[6.0], [8.0], [10.0]]]),
            'parameters': jnp.array([[[0.2], [0.4], [0.6]], [[0.3], [0.5], [0.7]]])
        }
        
        y_large = pytree_bij.forward(x_large)
        x_large_recovered = pytree_bij.inverse(y_large)
        
        assert jnp.allclose(x_large_recovered['observations'], x_large['observations'], atol=1e-5)
        assert jnp.allclose(x_large_recovered['parameters'], x_large['parameters'], atol=1e-5)
    
    def test_complex_bijectors_variable_event_sizes(self):
        """Test complex bijectors with runtime event size changes."""
        bijector_tree = {
            'positive': tfb.Softplus(),
            'exponential': tfb.Exp(),
            'sigmoid': tfb.Sigmoid(),
            'identity': tfb.Identity()
        }
        
        # Initialize with specific shapes
        example_tree = {
            'positive': jnp.array([[[0.5, 1.0]]]),      # (1, 1, 2)
            'exponential': jnp.array([[[0.0]]]),        # (1, 1, 1)
            'sigmoid': jnp.array([[[0.0, -1.0]]]),     # (1, 1, 2)
            'identity': jnp.array([[[5.0, 10.0]]])     # (1, 1, 2)
        }
        
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        # Test with different event sizes
        x_modified = {
            'positive': jnp.array([[[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]]),  # (1, 3, 2)
            'exponential': jnp.array([[[0.0], [0.5]]]),                      # (1, 2, 1)
            'sigmoid': jnp.array([[[0.0, -1.0]]]),                           # (1, 1, 2) - same
            'identity': jnp.array([[[5.0, 10.0], [15.0, 20.0], [25.0, 30.0], [35.0, 40.0]]]) # (1, 4, 2)
        }
        
        y = pytree_bij.forward(x_modified)
        x_recovered = pytree_bij.inverse(y)
        
        # Check transformations work correctly
        assert jnp.allclose(y['positive'], tfb.Softplus().forward(x_modified['positive']))
        assert jnp.allclose(y['exponential'], jnp.exp(x_modified['exponential']))
        assert jnp.allclose(y['sigmoid'], tfb.Sigmoid().forward(x_modified['sigmoid']))
        assert jnp.allclose(y['identity'], x_modified['identity'])
        
        # Check recovery
        assert jnp.allclose(x_recovered['positive'], x_modified['positive'], atol=1e-5)
        assert jnp.allclose(x_recovered['exponential'], x_modified['exponential'], atol=1e-5)
        assert jnp.allclose(x_recovered['sigmoid'], x_modified['sigmoid'], atol=1e-5)
        assert jnp.allclose(x_recovered['identity'], x_modified['identity'])


class TestPyTreeBijectorMissingKeys:
    """Test PyTreeBijector when input pytrees have missing keys."""
    
    def test_missing_first_key(self):
        """Test with first key missing to catch flattening/unflattening issues."""
        # Example tree has keys ['a', 'b', 'c']
        example_tree = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([0.5, -0.5]),
            'c': jnp.array([2.0, 3.0])
        }
        
        bijector_tree = {
            'a': tfb.Identity(),
            'b': tfb.Sigmoid(),
            'c': tfb.Exp()
        }
        
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        # Input missing the first key 'a'
        x_missing_first = {
            'b': jnp.array([0.2, 0.8]),
            'c': jnp.array([1.0, 1.5])
        }
        
        y = pytree_bij.forward(x_missing_first)
        x_recovered = pytree_bij.inverse(y)
        
        # Check structure is preserved
        assert tree.structure(y) == tree.structure(x_missing_first)
        assert set(y.keys()) == set(x_missing_first.keys())
        
        # Check transformations applied to present keys
        assert jnp.allclose(y['b'], tfb.Sigmoid().forward(x_missing_first['b']))
        assert jnp.allclose(y['c'], jnp.exp(x_missing_first['c']))
        
        # Check recovery
        assert jnp.allclose(x_recovered['b'], x_missing_first['b'], atol=1e-6)
        assert jnp.allclose(x_recovered['c'], x_missing_first['c'], atol=1e-6)
    
    def test_missing_middle_key(self):
        """Test with middle key missing."""
        example_tree = {
            'param1': jnp.array([1.0, 2.0]),
            'param2': jnp.array([0.5, -0.5]),
            'param3': jnp.array([2.0, 3.0])
        }
        
        bijector_tree = {
            'param1': tfb.Softplus(),
            'param2': tfb.Sigmoid(),
            'param3': tfb.Identity()
        }
        
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        # Input missing the middle key 'param2'
        x_missing_middle = {
            'param1': jnp.array([0.5, 1.0]),
            'param3': jnp.array([5.0, 10.0])
        }
        
        y = pytree_bij.forward(x_missing_middle)
        x_recovered = pytree_bij.inverse(y)
        
        # Check transformations
        assert jnp.allclose(y['param1'], tfb.Softplus().forward(x_missing_middle['param1']))
        assert jnp.allclose(y['param3'], x_missing_middle['param3'])  # Identity
        
        # Check recovery
        assert jnp.allclose(x_recovered['param1'], x_missing_middle['param1'], atol=1e-5)
        assert jnp.allclose(x_recovered['param3'], x_missing_middle['param3'])
    
    def test_missing_multiple_keys(self):
        """Test with several keys missing."""
        example_tree = {
            'key1': jnp.array([1.0]),
            'key2': jnp.array([0.5]),
            'key3': jnp.array([2.0]),
            'key4': jnp.array([0.1]),
            'key5': jnp.array([3.0])
        }
        
        bijector_tree = {
            'key1': tfb.Identity(),
            'key2': tfb.Sigmoid(),
            'key3': tfb.Exp(),
            'key4': tfb.Softplus(),
            'key5': tfb.Identity()
        }
        
        pytree_bij = PyTreeBijector(bijector_tree, example_tree)
        
        # Input with only keys 2 and 4 present
        x_sparse = {
            'key2': jnp.array([0.3]),
            'key4': jnp.array([1.5])
        }
        
        y = pytree_bij.forward(x_sparse)
        x_recovered = pytree_bij.inverse(y)
        
        # Check correct bijectors are applied
        assert jnp.allclose(y['key2'], tfb.Sigmoid().forward(x_sparse['key2']))
        assert jnp.allclose(y['key4'], tfb.Softplus().forward(x_sparse['key4']))
        
        # Check recovery
        assert jnp.allclose(x_recovered['key2'], x_sparse['key2'], atol=1e-6)
        assert jnp.allclose(x_recovered['key4'], x_sparse['key4'], atol=1e-6)
    
    def test_missing_keys_zscaling_bijector(self):
        """Test Z-scaling bijectors handle missing keys correctly."""
        # Create sample tree with (batch=2, events=1, value=1) shape
        sample_tree = {
            'alpha': jnp.array([[[1.0]], [[2.0]]]),  # (2, 1, 1)
            'beta': jnp.array([[[0.3]], [[0.7]]]),   # (2, 1, 1)
            'gamma': jnp.array([[[1.5]], [[2.5]]])   # (2, 1, 1)
        }
        
        # Generate representative data
        key = jr.PRNGKey(42)
        representative_data = {
            'alpha': jr.normal(key, (100, 1, 1)),
            'beta': jnp.clip(jr.uniform(jr.PRNGKey(1), (100, 1, 1)), 0.01, 0.99),
            'gamma': jr.gamma(jr.PRNGKey(2), 2.0, (100, 1, 1))
        }
        
        bijector_specs = {
            'alpha': tfb.Identity(),
            'beta': tfb.Sigmoid(),
            'gamma': tfb.Softplus()
        }
        
        pytree_bij = create_zscaling_bijector_tree(
            sample_tree, representative_data, bijector_specs
        )
        
        # Test with missing 'alpha' (first key)
        test_missing_alpha = {
            'beta': jnp.array([0.3, 0.7]),
            'gamma': jnp.array([1.5, 2.5])
        }
        
        transformed_missing_alpha = pytree_bij.forward(test_missing_alpha)
        recovered_missing_alpha = pytree_bij.inverse(transformed_missing_alpha)
        
        assert jnp.allclose(recovered_missing_alpha['beta'], test_missing_alpha['beta'], atol=1e-4)
        assert jnp.allclose(recovered_missing_alpha['gamma'], test_missing_alpha['gamma'], atol=1e-4)
        
        # Test with missing 'beta' (middle key)
        test_missing_beta = {
            'alpha': jnp.array([1.0, -1.0]),
            'gamma': jnp.array([0.5, 3.0])
        }
        
        transformed_missing_beta = pytree_bij.forward(test_missing_beta)
        recovered_missing_beta = pytree_bij.inverse(transformed_missing_beta)
        
        assert jnp.allclose(recovered_missing_beta['alpha'], test_missing_beta['alpha'], atol=1e-4)
        assert jnp.allclose(recovered_missing_beta['gamma'], test_missing_beta['gamma'], atol=1e-4)
    
    def test_missing_keys_forward_inverse_consistency(self):
        """Test forward/inverse operations preserve structure with missing keys."""
        # Create a complex example tree with nested structure
        example_tree = {
            'observations': {
                'data1': jnp.array([1.0, 2.0]),
                'data2': jnp.array([0.5, 0.6])
            },
            'parameters': {
                'theta': jnp.array([0.3]),
                'sigma': jnp.array([1.5])
            },
            'auxiliary': jnp.array([10.0])
        }
        
        # Bijector tree with manual specifications
        bijector_specs = {
            'data1': tfb.Identity(),
            'data2': tfb.Sigmoid(),
            'theta': tfb.Sigmoid(),
            'sigma': tfb.Softplus(),
            'auxiliary': tfb.Identity()
        }
        
        pytree_bij = create_bijector_tree(example_tree, bijector_specs)
        
        # Input missing entire 'parameters' subtree
        x_missing_subtree = {
            'observations': {
                'data1': jnp.array([3.0, 4.0]),
                'data2': jnp.array([0.2, 0.8])
            },
            'auxiliary': jnp.array([20.0])
        }
        
        y = pytree_bij.forward(x_missing_subtree)
        x_recovered = pytree_bij.inverse(y)
        
        # Check structure matches input
        assert tree.structure(y) == tree.structure(x_missing_subtree)
        assert tree.structure(x_recovered) == tree.structure(x_missing_subtree)
        
        # Check transformations for present keys
        assert jnp.allclose(y['observations']['data1'], x_missing_subtree['observations']['data1'])
        assert jnp.allclose(y['observations']['data2'], tfb.Sigmoid().forward(x_missing_subtree['observations']['data2']))
        assert jnp.allclose(y['auxiliary'], x_missing_subtree['auxiliary'])
        
        # Check recovery
        assert jnp.allclose(x_recovered['observations']['data1'], x_missing_subtree['observations']['data1'])
        assert jnp.allclose(x_recovered['observations']['data2'], x_missing_subtree['observations']['data2'], atol=1e-6)
        assert jnp.allclose(x_recovered['auxiliary'], x_missing_subtree['auxiliary'])
        
        # Test with missing part of nested structure
        x_partial_missing = {
            'observations': {
                'data1': jnp.array([5.0, 6.0])
                # 'data2' missing
            },
            'parameters': {
                'theta': jnp.array([0.7])
                # 'sigma' missing
            }
            # 'auxiliary' missing
        }
        
        y_partial = pytree_bij.forward(x_partial_missing)
        x_partial_recovered = pytree_bij.inverse(y_partial)
        
        # Verify correct transformations are applied
        assert jnp.allclose(y_partial['observations']['data1'], x_partial_missing['observations']['data1'])
        assert jnp.allclose(y_partial['parameters']['theta'], tfb.Sigmoid().forward(x_partial_missing['parameters']['theta']))
        
        # Verify recovery
        assert jnp.allclose(x_partial_recovered['observations']['data1'], x_partial_missing['observations']['data1'])
        assert jnp.allclose(x_partial_recovered['parameters']['theta'], x_partial_missing['parameters']['theta'], atol=1e-6)
