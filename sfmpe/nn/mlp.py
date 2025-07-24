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
        # self.in_linear = nnx.LinearGeneral(
            # (in_dim, config['latent_dim']),
            # config['latent_dim'],
            # rngs=rngs,
            # axis=(-2, -1)
        # )
        # self.in_linear = nnx.LinearGeneral(
            # (in_dim + 1, value_dim),
            # config['latent_dim'],
            # rngs=rngs,
            # axis=(-2, -1)
        # )
        self.in_linear = nnx.Linear(
            (in_dim * value_dim + 1),
            config['latent_dim'],
            rngs=rngs
        )
        self.mlp = MLP(
            dim=config['latent_dim'],
            n_layers=config['n_ff'],
            dropout=config['dropout'],
            activation=config['activation'],
            rngs=rngs
        )
        self.out_linear = nnx.LinearGeneral(
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
        # theta = self.embedding(theta, theta_label, theta_index, time)
        # context = self.embedding(
            # context,
            # context_label,
            # context_index,
            # time,
            # is_context=True
        # )
        # x = jnp.concatenate([context, theta, time], axis=1)
        x = jnp.concatenate([
            context.reshape(context.shape[0], -1),
            theta.reshape(theta.shape[0], -1),
            time.reshape(time.shape[0], -1),
        ], axis=-1)
        # x = jnp.concatenate([context, theta, time], axis=1)
        x = self.in_linear(x)
        x = self.mlp(x)
        return self.out_linear(x)
