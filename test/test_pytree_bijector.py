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
    create_manual_bijector_tree,
    create_zscaling_bijector_from_distribution,
    create_manual_zscaling_bijector_tree
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


class TestCreateZScalingBijectorFromDistribution:
    """Test creation of Z-scaling bijector from JointDistributionNamed."""
    
    def test_basic_functionality(self):
        """Test basic Z-scaling with simple joint distribution."""
        prior = tfd.JointDistributionNamed({
            'a': tfd.Normal(0.0, 1.0),
            'b': tfd.Uniform(0.0, 1.0)
        })
        
        # Generate representative data with known statistics
        key = jr.PRNGKey(42)
        representative_data = prior.sample(1000, seed=key)
        
        # Create Z-scaling bijector
        pytree_bij = create_zscaling_bijector_from_distribution(
            prior, representative_data
        )
        
        # Test with new samples
        test_samples = prior.sample(10, seed=jr.PRNGKey(123))
        transformed = pytree_bij.forward(test_samples)
        
        # Check structure is preserved
        assert tree.structure(transformed) == tree.structure(test_samples)
        
        # Forward-inverse consistency
        recovered = pytree_bij.inverse(transformed)
        assert jnp.allclose(recovered['a'], test_samples['a'], atol=1e-5)
        assert jnp.allclose(recovered['b'], test_samples['b'], atol=1e-5)
    
    def test_zscaling_statistics(self):
        """Test that Z-scaling produces normalized statistics."""
        prior = tfd.JointDistributionNamed({
            'normal_param': tfd.Normal(5.0, 2.0),  # Non-standard mean/std
            'bounded_param': tfd.Beta(2.0, 5.0)   # Skewed distribution
        })
        
        # Generate representative data
        key = jr.PRNGKey(111)
        representative_data = prior.sample(2000, seed=key)
        
        pytree_bij = create_zscaling_bijector_from_distribution(
            prior, representative_data
        )
        
        # Transform large sample to check statistics
        large_sample = prior.sample(5000, seed=jr.PRNGKey(222))
        transformed_sample = pytree_bij.forward(large_sample)
        
        # Check that transformed data has approximately zero mean, unit std
        for key_name in transformed_sample.keys():
            mean_val = jnp.mean(transformed_sample[key_name])
            std_val = jnp.std(transformed_sample[key_name])
            
            assert jnp.abs(mean_val) < 0.1, f"Mean should be ~0, got {mean_val}"
            assert jnp.abs(std_val - 1.0) < 0.1, f"Std should be ~1, got {std_val}"
    
    def test_complex_distribution(self):
        """Test with complex multi-parameter distribution."""
        prior = tfd.JointDistributionNamed({
            'positive': tfd.Gamma(3.0, 2.0),
            'bounded': tfd.Beta(1.5, 3.5),
            'normal': tfd.Normal(-2.0, 3.0),
            'uniform': tfd.Uniform(-5.0, 10.0)
        })
        
        key = jr.PRNGKey(333)
        representative_data = prior.sample(1500, seed=key)
        
        pytree_bij = create_zscaling_bijector_from_distribution(
            prior, representative_data
        )
        
        # Test forward-inverse consistency
        test_samples = prior.sample(20, seed=jr.PRNGKey(444))
        transformed = pytree_bij.forward(test_samples)
        recovered = pytree_bij.inverse(transformed)
        
        for key_name in test_samples.keys():
            assert jnp.allclose(
                recovered[key_name], 
                test_samples[key_name], 
                atol=1e-4
            )


class TestCreateManualZScalingBijectorTree:
    """Test manual Z-scaling bijector tree creation."""
    
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
        
        pytree_bij = create_manual_zscaling_bijector_tree(
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
        
        pytree_bij = create_manual_zscaling_bijector_tree(
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
        pytree_bij = create_manual_zscaling_bijector_tree(
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
        
        pytree_bij = create_manual_zscaling_bijector_tree(
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