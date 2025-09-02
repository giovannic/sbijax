import pytest
from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from sfmpe.metrics.mmd import mmd_test
from jaxtyping import Array

pytestmark = pytest.mark.slow

@pytest.mark.parametrize("dim", [1, 2, 16])
def test_mmd_accepts_null_normal(dim):
    """
    Test that MMD does not reject the null when empirical samples are from 
    the same Normal distribution as the reference distribution.
    """
    key = jr.PRNGKey(42)
    n_samples = 1500
    
    # Reference distribution: standard normal
    reference_dist = tfd.Normal(jnp.zeros(dim), jnp.ones(dim))
    
    # Generate empirical samples from the same distribution
    empirical_samples : Array = reference_dist.sample(n_samples, seed=key) # type: ignore
    
    # Run MMD test
    test_key = jr.PRNGKey(123)
    result = mmd_test(
        empirical_samples=empirical_samples,
        reference_distribution=reference_dist,
        rng_key=test_key,
        n_permutations=500,
        threshold=0.05
    )
    
    # Should not reject null hypothesis
    assert not result['reject_null'], (
        f"MMD incorrectly rejected null for dim={dim}. "
        f"p_value={result['p_value']:.4f}"
    )
    assert result['p_value'] > 0.05, (
        f"p-value {result['p_value']:.4f} should be > 0.05 for dim={dim}"
    )
    
    # Check return types
    assert isinstance(result['reject_null'], jnp.ndarray)
    assert isinstance(result['statistic'], jnp.ndarray) 
    assert isinstance(result['critical_value'], jnp.ndarray)
    assert isinstance(result['p_value'], jnp.ndarray)


@pytest.mark.parametrize("dim", [1, 2, 16])
def test_mmd_rejects_shifted_mean(dim):
    """
    Test that MMD rejects the null when empirical samples have a shifted mean
    compared to the reference distribution.
    """
    key = jr.PRNGKey(42)
    n_samples = 1500
    shift_magnitude = 2.0
    
    # Reference distribution: standard normal
    reference_dist = tfd.Normal(jnp.zeros(dim), jnp.ones(dim))
    
    # Generate empirical samples with shifted mean
    shifted_dist = tfd.Normal(jnp.ones(dim) * shift_magnitude, jnp.ones(dim))
    empirical_samples: Array = shifted_dist.sample(n_samples, seed=key) # type: ignore
    
    # Run MMD test
    test_key = jr.PRNGKey(123)
    result = mmd_test(
        empirical_samples=empirical_samples,
        reference_distribution=reference_dist,
        rng_key=test_key,
        n_permutations=500,
        threshold=0.05
    )
    
    # Should reject null hypothesis
    assert result['reject_null'], (
        f"MMD failed to reject null for shifted mean, dim={dim}. "
        f"p_value={result['p_value']:.4f}"
    )
    assert result['p_value'] < 0.05, (
        f"p-value {result['p_value']:.4f} should be < 0.05 for dim={dim}"
    )


@pytest.mark.parametrize("dim", [1, 2, 16])
def test_mmd_rejects_scaled_variance(dim):
    """
    Test that MMD rejects the null when empirical samples have different 
    variance compared to the reference distribution.
    """
    key = jr.PRNGKey(42)
    n_samples = 1500
    scale_factor = 3.0
    
    # Reference distribution: standard normal
    reference_dist = tfd.Normal(jnp.zeros(dim), jnp.ones(dim))
    
    # Generate empirical samples with scaled variance
    scaled_dist = tfd.Normal(jnp.zeros(dim), jnp.ones(dim) * scale_factor)
    empirical_samples: Array = scaled_dist.sample(n_samples, seed=key) # type: ignore
    
    # Run MMD test
    test_key = jr.PRNGKey(123)
    result = mmd_test(
        empirical_samples=empirical_samples,
        reference_distribution=reference_dist,
        rng_key=test_key,
        n_permutations=500,
        threshold=0.05
    )
    
    # Should reject null hypothesis
    assert result['reject_null'], (
        f"MMD failed to reject null for scaled variance, dim={dim}. "
        f"p_value={result['p_value']:.4f}"
    )
    assert result['p_value'] < 0.05, (
        f"p-value {result['p_value']:.4f} should be < 0.05 for dim={dim}"
    )
