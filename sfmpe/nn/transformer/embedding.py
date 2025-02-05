from jax import numpy as jnp, random
from flax import nnx
from ...util.dataloader import PAD_VALUE

class GaussianFourierEmbedding(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs):
        b_dim = out_dim // 2
        self.b = nnx.Param(
            random.normal(rngs.params(), (in_dim, b_dim))
        )

    def __call__(self, inputs):
        return jnp.concatenate([
            jnp.cos(2 * jnp.pi * jnp.dot(inputs, self.b.value)), #type: ignore
            jnp.sin(2 * jnp.pi * jnp.dot(inputs, self.b.value)) #type: ignore
        ], axis=-1)

class Embedding(nnx.Module):

    def __init__(
        self,
        value_dim,
        n_labels,
        label_dim,
        index_in_dim,
        index_out_dim,
        out_dim,
        rngs=nnx.Rngs(0),
        ):
        self.embedding = nnx.Embed(
            n_labels,
            features=label_dim,
            rngs=rngs
        )
        self.gf_embedding = GaussianFourierEmbedding(
            index_in_dim,
            index_out_dim,
            rngs
        )
        in_dim = value_dim + label_dim + index_out_dim + 1
        self.linear = nnx.Linear(
            in_dim, # value + label + index + time
            out_dim,
            rngs=rngs
        )

    def __call__(self, values, labels, indices, time=None):
        """
        Embed random variables for encoding

        Args:
            values: random variable values
            labels: the type of random variable
            indices: the index for the random variable in infinite space
        """
        # embed labels
        labels = self.embedding(labels)
        # reshape to samples x tokens x features
        labels = jnp.broadcast_to(
            labels,
            (values.shape[0],) + labels.shape[1:]
        )

        # concatenate into tokens
        if time is None:
            time = jnp.full(
                values.shape[:2] + (1,),
                PAD_VALUE
            )
        else:
            time = jnp.broadcast_to(
                time,
                values.shape[:2] + (1,)
            )

        # gaussian fourier embedding of indices
        if indices is not None:
            indices = self.gf_embedding(indices)

            x = jnp.concatenate([
                values,
                labels,
                indices,
                time
            ], axis=-1)
        else:
            x = jnp.concatenate([
                values,
                labels,
                time
            ], axis=-1)

        # apply linear transform to tokens
        return self.linear(x)
