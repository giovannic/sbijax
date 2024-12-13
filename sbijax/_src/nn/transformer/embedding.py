from jax import numpy as jnp, random
from flax import nnx

class GaussianFourierEmbedding(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs):
        b_dim = out_dim // 2
        self.b = nnx.Param(
            random.normal(rngs.params(), (in_dim, b_dim))
        )

    def __call__(self, inputs):
        return jnp.concatenate([
            jnp.cos(2 * jnp.pi * jnp.dot(inputs, self.b)), #type: ignore
            jnp.sin(2 * jnp.pi * jnp.dot(inputs, self.b)) #type: ignore
        ], axis=-1)

class Embedding(nnx.Module):

    def __init__(
        self,
        n_labels,
        label_dim,
        index_in_dim,
        index_out_dim,
        out_dim,
        rngs=nnx.Rngs(0)
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
        self.linear = nnx.Linear(
            2 + label_dim + index_out_dim, # value + label + index + time
            out_dim,
            rngs=rngs
        )

    def __call__(self, values, labels, indices, time):
        """
        Embed random variables for encoding

        Args:
            values: random variable values
            labels: the type of random variable
            indices: the index for the random variable in infinite space
        """
        batch_size = values.shape[0]
        n_tokens = values.shape[1]

        # reshape to batch x tokens x features
        values = values[..., jnp.newaxis]

        # embed labels
        labels = self.embedding(labels)
        # reshape to batch x tokens x features
        labels = jnp.tile(labels, (batch_size, 1, 1))

        # gaussian fourier embedding of indices
        indices = self.gf_embedding(indices)

        # fill scalar time to match batch x tokens x features
        time = jnp.full((batch_size, n_tokens, 1), time)

        # reshape into tokens
        x = jnp.concatenate([values, labels, indices, time], axis=-1)

        # apply linear transform to tokens
        return self.linear(x)
