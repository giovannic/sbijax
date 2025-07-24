from flax import nnx
from .transformer.encoder import MLP
from jax import numpy as jnp

class VectorFieldModel(nnx.Module):

    def __call__(self, theta, time, context) -> jnp.ndarray:
        del theta
        del time
        del context
        return jnp.ndarray([])

class VanillaMLPVectorField(VectorFieldModel):
    mlp: MLP

    def __init__(
        self,
        theta_dim,
        context_dim,
        latent_dim,
        n_layers,
        dropout,
        activation,
        rngs=nnx.Rngs(0)
        ):

        self.in_linear = nnx.Linear(
            1 + theta_dim + context_dim,
            latent_dim, rngs=rngs
        )
        self.mlp = MLP(
            latent_dim,
            n_layers,
            dropout,
            activation,
            rngs
        )
        self.out_linear = nnx.Linear(latent_dim, theta_dim, rngs=rngs)

    def __call__(self, theta, time, context):
        x = jnp.concatenate([theta, time, context], axis=-1)
        x = self.in_linear(x)
        x = self.mlp(x)
        return self.out_linear(x)

