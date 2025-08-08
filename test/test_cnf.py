from sfmpe.cnf import CNF
from sfmpe.nn.mlp import MLPVectorField, VectorFieldModel
from flax import nnx
from jax import numpy as jnp, random as jr, vmap
from utils.helpers import ks_norm_test
import pytest

@pytest.mark.parametrize(
    "n_dim,n_context",
    [
        (1, 1),
        (8, 2),
        (16, 5)
    ]
)
def test_cnf_can_be_initialised_at_correct_size(n_dim, n_context):
    # create a continuous flow with a linear transform
    rngs = nnx.Rngs(0, base_dist=0)
    n = 10

    class DummyTransform(VectorFieldModel):
        def __init__(self, n_context, n_dim, rngs):
            self.linear = nnx.Linear(n_context, n_dim, rngs=rngs)
        def __call__(self, theta, time, context):
            del theta
            del time
            return self.linear(context)

    transform = DummyTransform(n_context, n_dim, rngs)
    cnf = CNF(transform)
    assert cnf.sample(
        jnp.zeros((1, n_context)),
        (n_dim,),
        sample_size=n
    ).shape == (n, n_dim)
    assert cnf.vector_field(
        jnp.zeros((n, n_dim)), # theta
        .5, # time
        jnp.zeros((n, n_context)) # context
    ).shape == (n, n_dim)


def test_cnf_manages_rngs_correctly():
    """Test that CNF properly manages RNG state during sampling.
    """
    dim = 1
    n_obs = 10
    n_layers = 2
    sample_size = 5
    
    # Setup model
    key = jr.PRNGKey(42)
    key, nnx_key = jr.split(key)
    rngs = nnx.Rngs(nnx_key, dropout=1)
    
    nn = MLPVectorField(
        theta_dim=dim,
        context_dim=dim * n_obs,
        latent_dim=64,
        n_layers=n_layers,
        dropout=0.1,
        activation=nnx.relu,
        rngs=rngs
    )
    
    model = CNF(nn, rngs=rngs)
    nn.eval()  # Disable dropout during sampling
    context = jr.normal(key, (1, dim * n_obs))
    
    # Get RNG state before sampling
    rng_before = nnx.state(model, nnx.RngState)['rngs']
    dropout_count_before = int(rng_before['dropout']['count'].value) # type: ignore
    
    # Sample from the model
    model.sample(
        context=context,
        theta_shape=(dim,),
        sample_size=sample_size
    )
    
    # Get RNG state after sampling
    rng_after = nnx.state(model, nnx.RngState)['rngs']
    dropout_count_after = rng_after['dropout']['count'].value # type: ignore
    
    # Expected: dropout should be incremented by sample_size * n_layers
    expected_count = dropout_count_before
    
    # This assertion will fail due to tracer leak - dropout_count_after is a tracer
    assert dropout_count_after == expected_count, (
        f"Expected dropout count {expected_count}, got {dropout_count_after} "
        f"(type: {type(dropout_count_after)})"
    )

@pytest.mark.parametrize(
    "n_dim,n_context",
    [
        (1, 1),
        (4, 2)
    ]
)
def test_that_cnf_preserves_latent_space(n_dim, n_context):
    """
    Test that when sampling forward and then backward, the latent space is the same
    Check the CDF with Kolmogorov-Smirnov and the independence between dimensions
    """
    
    # Setup
    rngs = nnx.Rngs(0, base_dist=1)
    sample_size = 1000

    class DummyTransform(VectorFieldModel):
        def __init__(self, n_context, n_dim, rngs):
            self.linear = nnx.Linear(n_context, n_dim, rngs=rngs)
        def __call__(self, theta, time, context):
            del theta
            del time
            return self.linear(context)

    transform = DummyTransform(n_context, n_dim, rngs)
    cnf = CNF(transform)
    
    # Generate initial latent samples from standard normal
    initial_latent = jr.normal(
        jr.PRNGKey(42), 
        (sample_size, n_dim)
    )
    
    # Create dummy context - single context for all samples
    context = jnp.zeros((1, n_context))
    
    # Forward sampling: latent -> data space
    data_samples = cnf.sample(
        context=context,
        theta_shape=(n_dim,),
        sample_size=sample_size,
        theta_0=initial_latent,
        direction='forward'
    )
    
    # Backward sampling: data space -> recovered latent
    recovered_latent = cnf.sample(
        context=context,
        theta_shape=(n_dim,),
        sample_size=sample_size,
        theta_0=data_samples,
        direction='backward'
    )
    
    # Test 1: Check that each dimension follows standard normal distribution
    ks_results = vmap(
        lambda x: ks_norm_test(x, alpha=0.01),
        in_axes=(1,)
    )(recovered_latent)
    
    # All dimensions should pass normality test
    assert jnp.all(~ks_results['reject_null']), (
        f"Normality test failed for some dimensions. "
        f"Statistics: {ks_results['statistic']}, "
        f"Critical values: {ks_results['critical_value']}"
    )
    
    # Test 2: Check independence between dimensions (for multi-dimensional case)
    if n_dim > 1:
        # Compute correlation matrix
        corr_matrix = jnp.corrcoef(recovered_latent, rowvar=False)
        
        # Off-diagonal elements should be close to zero (independence)
        off_diagonal = corr_matrix[jnp.triu_indices(n_dim, k=1)]
        max_correlation = jnp.max(jnp.abs(off_diagonal))
        
        assert max_correlation < 0.1, (
            f"Dimensions are not independent. "
            f"Maximum correlation: {max_correlation}, "
            f"Correlation matrix: {corr_matrix}"
        )
    
    # Test 3: Check that mean and std are approximately correct
    mean = jnp.mean(recovered_latent, axis=0)
    std = jnp.std(recovered_latent, axis=0)
    
    assert jnp.allclose(mean, 0.0, atol=0.1), (
        f"Mean not close to 0: {mean}"
    )
    assert jnp.allclose(std, 1.0, atol=0.1), (
        f"Standard deviation not close to 1: {std}"
    )


@pytest.mark.parametrize(
    "n_dim,n_context",
    [
        (1, 1),
        (4, 2)
    ]
)
def test_that_cnf_preserves_latent_space_without_theta_0(n_dim, n_context):
    """
    Test that CNF preserves latent space without explicitly setting theta_0.
    Uses default sampling from base distribution for forward pass.
    """
    # Setup
    rngs = nnx.Rngs(0, base_dist=1)
    sample_size = 1000

    class DummyTransform(VectorFieldModel):
        def __init__(self, n_context, n_dim, rngs):
            self.linear = nnx.Linear(n_context, n_dim, rngs=rngs)
        def __call__(self, theta, time, context):
            del theta
            del time
            return self.linear(context)

    transform = DummyTransform(n_context, n_dim, rngs)
    cnf = CNF(transform)
    
    # Create dummy context - single context for all samples
    context = jnp.zeros((1, n_context))
    
    # Forward sampling: base dist -> data space (without theta_0)
    data_samples = cnf.sample(
        context=context,
        theta_shape=(n_dim,),
        sample_size=sample_size,
        direction='forward'
    )
    
    # Backward sampling: data space -> recovered latent
    recovered_latent = cnf.sample(
        context=context,
        theta_shape=(n_dim,),
        sample_size=sample_size,
        theta_0=data_samples,
        direction='backward'
    )
    
    # Test 1: Check that each dimension follows standard normal distribution
    ks_results = vmap(
        lambda x: ks_norm_test(x, alpha=0.01),
        in_axes=(1,)
    )(recovered_latent)
    
    # All dimensions should pass normality test
    assert jnp.all(~ks_results['reject_null']), (
        f"Normality test failed for some dimensions. "
        f"Statistics: {ks_results['statistic']}, "
        f"Critical values: {ks_results['critical_value']}"
    )
    
    # Test 2: Check independence between dimensions (for multi-dimensional case)
    if n_dim > 1:
        # Compute correlation matrix
        corr_matrix = jnp.corrcoef(recovered_latent, rowvar=False)
        print(f'correlation matrix: {corr_matrix}')
        
        # Off-diagonal elements should be close to zero (independence)
        off_diagonal = corr_matrix[jnp.triu_indices(n_dim, k=1)]
        max_correlation = jnp.max(jnp.abs(off_diagonal))
        
        assert max_correlation < 0.1, (
            f"Dimensions are not independent. "
            f"Maximum correlation: {max_correlation}, "
            f"Correlation matrix: {corr_matrix}"
        )
    
    # Test 3: Check that mean and std are approximately correct
    mean = jnp.mean(recovered_latent, axis=0)
    std = jnp.std(recovered_latent, axis=0)
    
    assert jnp.allclose(mean, 0.0, atol=0.1), (
        f"Mean not close to 0: {mean}"
    )
    assert jnp.allclose(std, 1.0, atol=0.1), (
        f"Standard deviation not close to 1: {std}"
    )
