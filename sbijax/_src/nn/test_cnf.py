from sbijax._src.nn.make_continuous_flow import CNF
from flax import nnx
from jax import numpy as jnp

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

