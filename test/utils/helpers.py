from jax import numpy as jnp, scipy as js
from jaxtyping import Array

def ks_norm_test(data: Array, alpha: float = 0.05) -> dict:
    
    """
    Kolmogorov-Smirnov test for normality using cumulative statistics.
    Tests if data comes from a standard normal distribution.
    
    Args:
        data: 1D array of data points
        alpha: significance level (default 0.05)
    
    Returns:
        dict with test statistic, critical value, and p_value estimate
    """
    n = len(data)
    
    # Sort the data
    sorted_data = jnp.sort(data)
    
    # Compute empirical CDF
    empirical_cdf = jnp.arange(1, n + 1) / n
    
    # Compute theoretical CDF (standard normal)
    theoretical_cdf = js.stats.norm.cdf(sorted_data)
    
    # KS test statistic: maximum difference between CDFs
    D = jnp.max(jnp.abs(empirical_cdf - theoretical_cdf))
    
    # Critical value for KS test (asymptotic approximation)
    critical_value = jnp.sqrt(-0.5 * jnp.log(alpha / 2)) / jnp.sqrt(n)
    
    # Rough p-value estimate (Kolmogorov distribution approximation)
    p_value_approx = 2 * jnp.exp(-2 * n * D**2)
    
    return {
        'statistic': D,
        'critical_value': critical_value,
        'p_value_approx': p_value_approx,
        'reject_null': D > critical_value,
        'sample_size': n
    }


