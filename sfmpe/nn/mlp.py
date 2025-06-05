from flax import nnx
from .transformer.encoder import MLP #TODO: should be in this dir
from .transformer.embedding import Embedding
import jax.numpy as jnp

class MLPVectorField(nnx.Module):
    def __init__(self, config, n_labels, in_dim, value_dim, out_dim, rngs=nnx.Rngs(0)):
        self.latent_dim = config['latent_dim']
        self.embedding = Embedding(
            value_dim,
            n_labels,
            config['label_dim'],
            0,
            config['index_out_dim'],
            config['latent_dim'],
            rngs=rngs
        )
        self.in_linear = nnx.LinearGeneral(
            (in_dim, config['latent_dim']),
            config['latent_dim'],
            rngs=rngs,
            axis=(-2, -1)
        )
        self.mlp = MLP(
            dim=config['latent_dim'],
            n_layers=config['n_ff'],
            dropout=config['dropout'],
            activation=config['activation'],
            rngs=rngs
        )
        self.out_linear= nnx.LinearGeneral(
            config['latent_dim'],
            (out_dim, 1),
            rngs=rngs,
            axis=-1
        )

    def __call__(
        self,
        context,
        context_label,
        context_index,
        context_mask,
        theta,
        theta_label,
        theta_index,
        theta_mask,
        cross_mask,
        time,
        ):
        theta = self.embedding(theta, theta_label, theta_index, time)
        x = jnp.concatenate([
            self.embedding(context, context_label, context_index, time, is_context=True),
            theta,
            # jnp.broadcast_to(
                # time,
                # theta.shape[:1] + (1, self.latent_dim)
            # )
        ], axis=1)
        # import jax
        # jax.debug.print('broken {}', x[:2])
        # jax.debug.print('working {}', jnp.concatenate([
            # self.embedding(context, context_label, context_index),
            # theta,
            # jnp.broadcast_to(
                # time,
                # theta.shape[:1] + (1, self.latent_dim)
            # )
        # ], axis=1)[:2])
        # jax.debug.print('time {}', time[:2])
        return self.out_linear(self.mlp(self.in_linear(x)))
