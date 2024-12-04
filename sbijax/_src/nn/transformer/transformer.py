import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Optional
from jaxtyping import Array
from .encoder import Encoder

class Transformer(nn.Module):
    """Transformer with encoder-decoder architecture
    which encodes observations and decodes a vector field
    for flow matching.

    Attributes:
    num_layers: number of layers.
    latent_dim: latent dimension.
    num_heads: number of heads.
    dim_feedforward: feedforward dimension.
    dropout_prob: dropout rate.
    padding_value: padding value.
    sentinel_value: sentinel value.
    """

    num_layers : int
    latent_dim : int
    output_dim: int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float
    padding_value: float = -1.0
    rng_collection: str = 'transformer'

    def setup(self):
        self.embedding = Embedding(self.latent_dim, self.padding_value)
        self.encoder = Encoder(
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout_prob=self.dropout_prob
        )
        self.decoder = Encoder(
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout_prob=self.dropout_prob
        )
        self.y_proj = nn.Dense(self.output_dim)

    def _encode(self, obs: Array, train: bool=True) -> Array:
        """Encode inputs into the latent space

        Args:
        inputs: inputs to encode into latent space
        train: (optional) whether we are training or not
        """
        # Embed targets into transformers latent space
        x = self.embedding(
            obs,
            obs[..., :1] != self.padding_value
        )

        mask = nn.make_attention_mask(
            to_1d(x) != self.padding_value,
            to_1d(x) != self.padding_value
        )

        return self.encoder(x, train=train, encoder_mask=mask)

    def _decode(self, pos: Array, obs: Array, train: bool=True) -> Array:
        """Decode flow positions into vector field given encoded observations

        Args:
        pos: Array, encoded positions of variables in flow
        obs: Array, encoded observations
        train: bool (optional) whether we are training or not
        """
        # Embed targets into transformers latent space
        targets = self.embedding(
            pos,
            pos[..., :1] != self.padding_value
        )

        # Create mask for decoder
        mask = nn.make_attention_mask(
            to_1d(targets) != self.padding_value,
            to_1d(targets) != self.padding_value
        )

        # Decode targets
        y = self.decoder(targets, train=train, encoder_mask=mask)
        return self.y_proj(y)

    def __call__( # type: ignore
        self,
        obs: Array,
        pos: Array,
        train: bool=True
        ):
        obs = self._encode(obs, train=train)
        v_field= self._decode(pos, obs, train=train)
        return v_field

class Embedding(nn.Module):
    """Embedding layer which allows for masking.

    Attributes:
    output_dim: output dimension.
    value: value to use in place of masked values.
    """
    output_dim : int
    value : float

    @nn.compact
    def __call__(self, x: Array, mask: Array) -> Array: # type: ignore
        y = nn.Dense(self.output_dim)(x)
        y = jnp.where(mask, y, jnp.ones_like(y) * self.value)
        return y #type: ignore

def to_1d(x):
    return x[..., 0]
