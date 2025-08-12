from typing import Dict
from jax import numpy as jnp, random as jr
from jaxtyping import Array
from tensorflow_probability.substrates.jax import distributions as tfd


def _gaussian_kernel(x: Array, y: Array, bandwidth: float) -> Array:
    """
    Compute Gaussian (RBF) kernel between two sets of points.
    
    Parameters
    ----------
    x : Array
        First set of points, shape (n, d)
    y : Array  
        Second set of points, shape (m, d)
    bandwidth : float
        Kernel bandwidth parameter
        
    Returns
    -------
    Array
        Kernel matrix of shape (n, m)
    """
    # Compute pairwise squared distances
    x_sqnorms = jnp.sum(x**2, axis=1, keepdims=True)  # (n, 1)
    y_sqnorms = jnp.sum(y**2, axis=1, keepdims=True)  # (m, 1)
    
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    sq_distances = (x_sqnorms + y_sqnorms.T - 2 * jnp.dot(x, y.T))
    
    return jnp.exp(-sq_distances / (2 * bandwidth**2))


def _mmd_statistic(X: Array, Y: Array, bandwidth: float) -> Array:
    """
    Compute the biased MMD^2 statistic between two samples.
    
    Parameters
    ----------
    X : Array
        First sample, shape (n, d)
    Y : Array
        Second sample, shape (m, d)  
    bandwidth : float
        Kernel bandwidth parameter
        
    Returns
    -------
    Array
        MMD^2 statistic value
    """
    n, m = X.shape[0], Y.shape[0]
    
    # Compute kernel matrices
    Kxx = _gaussian_kernel(X, X, bandwidth)
    Kyy = _gaussian_kernel(Y, Y, bandwidth) 
    Kxy = _gaussian_kernel(X, Y, bandwidth)
    
    # MMD^2 = (1/n^2) * sum(Kxx) + (1/m^2) * sum(Kyy) - (2/nm) * sum(Kxy)
    # Subtract diagonal terms to get unbiased estimate
    mmd2 = (jnp.sum(Kxx) - jnp.trace(Kxx)) / (n * (n - 1))
    mmd2 += (jnp.sum(Kyy) - jnp.trace(Kyy)) / (m * (m - 1))
    mmd2 -= 2 * jnp.sum(Kxy) / (n * m)
    
    return mmd2


def _permutation_test(
    X: Array, 
    Y: Array, 
    n_permutations: int, 
    bandwidth: float,
    rng_key: Array
) -> Array:
    """
    Perform permutation test to compute null distribution of MMD statistic.
    
    Parameters
    ----------
    X : Array
        First sample, shape (n, d)
    Y : Array
        Second sample, shape (m, d)
    n_permutations : int
        Number of permutations to perform
    bandwidth : float
        Kernel bandwidth parameter  
    rng_key : Array
        JAX random key
        
    Returns
    -------
    Array
        Array of MMD statistics under null hypothesis, shape (n_permutations,)
    """
    n, m = X.shape[0], Y.shape[0]
    combined = jnp.concatenate([X, Y], axis=0)
    
    def single_permutation(key):
        # Permute combined samples
        perm_idx = jr.permutation(key, n + m)
        perm_combined = combined[perm_idx]
        
        # Split back into two groups
        X_perm = perm_combined[:n]
        Y_perm = perm_combined[n:]
        
        return _mmd_statistic(X_perm, Y_perm, bandwidth)
    
    keys = jr.split(rng_key, n_permutations)
    null_stats = jnp.array([single_permutation(key) for key in keys])
    
    return null_stats


def mmd_test(
    empirical_samples: Array,
    reference_distribution: tfd.Distribution,
    rng_key: Array,
    n_permutations: int = 1000,
    threshold: float = 0.05,
    bandwidth: float = 1.0
) -> Dict[str, Array]:
    """
    Maximum Mean Discrepancy two-sample test between empirical samples and a 
    known parametric distribution.
    
    Parameters
    ----------
    empirical_samples : Array
        Empirical samples to test, shape (n, d)
    reference_distribution : tfd.Distribution
        TensorFlow Probability distribution to compare against
    rng_key : Array
        JAX random key for sampling and permutations
    n_permutations : int, default=1000
        Number of permutations for computing p-value
    threshold : float, default=0.05
        Significance threshold for the test
    bandwidth : float, default=1.0
        Gaussian kernel bandwidth parameter
        
    Returns
    -------
    Dict[str, Array]
        Dictionary containing:
        - 'reject_null': Boolean indicating if null hypothesis is rejected
        - 'statistic': MMD test statistic value  
        - 'critical_value': Critical value at given threshold
        - 'p_value': Computed p-value from permutation test
    """
    n_reference_samples = empirical_samples.shape[0]

    # Sample from reference distribution
    sample_key, perm_key = jr.split(rng_key)
    reference_samples: Array = reference_distribution.sample( # type: ignore
        n_reference_samples,
        seed=sample_key
    )

    # Compute MMD statistic
    mmd_stat = _mmd_statistic(empirical_samples, reference_samples, bandwidth)

    # Perform permutation test
    null_stats = _permutation_test(
        empirical_samples,
        reference_samples,
        n_permutations,
        bandwidth,
        perm_key
    )
    
    # Compute p-value
    p_value = jnp.mean(null_stats >= mmd_stat)
    
    # Compute critical value
    critical_value = jnp.quantile(null_stats, 1 - threshold)
    
    # Make decision
    reject_null = mmd_stat > critical_value
    
    return {
        'reject_null': reject_null,
        'statistic': mmd_stat,
        'critical_value': critical_value,
        'p_value': p_value
    }
