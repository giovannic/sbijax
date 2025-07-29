from sfmpe.cnf import CNF
from sfmpe.nn.mlp import MLPVectorField
from flax import nnx
from jax import numpy as jnp, random as jr

#TODO: different sizes
def test_cnf_can_be_initialised_at_correct_size():
    # create a continuous flow with a linear transform
    rngs = nnx.Rngs(0, base_dist=0)
    n = 10
    n_context = 5
    n_dim = 2


    class DummyTransform(nnx.Module):
        def __init__(self, n_context, n_dim, rngs):
            self.linear = nnx.Linear(n_context, n_dim, rngs=rngs)
        def __call__(self, theta, time, context):
            return self.linear(context)

    transform = DummyTransform(n_context, n_dim, rngs)
    cnf = CNF(n_dim, transform)
    assert cnf.sample(rngs, jnp.zeros((n, n_context))).shape == (n, n_dim)
    assert cnf.vector_field(
            jnp.zeros((n, n_dim)), # theta
            .5, # time
            jnp.zeros((n, n_context)) # context
        ).shape == (n, n_dim)


def test_cnf_manages_rngs_correctly():
    """Test that CNF properly manages RNG state during sampling.
    
    This test demonstrates the RNG tracer leak issue where dropout RNG 
    counts should increment by sample_size * n_layers but become traced.
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
    dropout_count_before = int(rng_before['dropout']['count'].value)
    
    # Sample from the model
    theta_samples = model.sample(
        context=context,
        theta_shape=(dim,),
        sample_size=sample_size
    )
    
    # Get RNG state after sampling
    rng_after = nnx.state(model, nnx.RngState)['rngs']
    dropout_count_after = rng_after['dropout']['count'].value
    
    # Expected: dropout should be incremented by sample_size * n_layers
    expected_count = dropout_count_before
    
    # This assertion will fail due to tracer leak - dropout_count_after is a tracer
    assert dropout_count_after == expected_count, (
        f"Expected dropout count {expected_count}, got {dropout_count_after} "
        f"(type: {type(dropout_count_after)})"
    )

