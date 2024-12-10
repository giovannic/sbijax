from typing import Callable
from jaxtyping import Array
from flax import nnx

class FFLayer(nnx.Module):

    activation: Callable

    def __init__(self, dim, dropout, activation, rngs=nnx.Rngs(0)):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.activation = activation

    def __call__(self, x: Array):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x, None

class MLP(nnx.Module):
    """Transformer MLP block."""

    n_layers: int

    def __init__(
        self,
        dim,
        n_layers,
        dropout,
        activation,
        rngs=nnx.Rngs(0)
        ):

        self.layers= nnx.Scan.constructor(FFLayer, length=n_layers)(
            dim,
            dropout,
            activation,
            rngs=rngs
        )

    def __call__(self, x) -> Array:
        y, _ = self.layers(x)
        return y

class EncoderBlock(nnx.Module):
    """Transformer Encoder Block."""

    def __init__(
            self,
            n_heads,
            dim,
            n_ff,
            dropout,
            activation,
            rngs=nnx.Rngs(0)
        ):
        
        self.attn = nnx.MultiHeadAttention(
            in_features=dim,
            num_heads=n_heads,
            qkv_features=dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs
        )
        self.attn_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.ff = MLP(
            dim=dim,
            n_layers=n_ff,
            dropout=dropout,
            activation=activation,
            rngs=rngs
        )
        self.ff_norm= nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x, mask=None):
        x_att = self.attn(x, x, x, mask=mask)
        x_att = self.attn_norm(x_att)
        x = x + x_att
        x_ff  = self.ff(x)
        x_ff = self.ff_norm(x_ff)
        return x_ff + x

class EncoderDecoderBlock(nnx.Module):
    """Transformer Encoder Decoder Block."""

    def __init__(
            self,
            n_heads,
            dim,
            n_ff,
            dropout,
            activation,
            rngs=nnx.Rngs(0)
        ):
        self.dec_attn = nnx.MultiHeadAttention(
            in_features=dim,
            num_heads=n_heads,
            qkv_features=dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs
        )
        self.dec_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.enc_attn = nnx.MultiHeadAttention(
            in_features=dim,
            num_heads=n_heads,
            qkv_features=dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs
        )
        self.enc_norm= nnx.LayerNorm(dim, rngs=rngs)
        self.ff = MLP(
            dim=dim,
            n_layers=n_ff,
            dropout=dropout,
            activation=activation,
            rngs=rngs
        )
        self.ff_norm = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x, encoded, mask=None):
        x_att = self.dec_attn(x, x, x, mask=mask)
        x_att = self.dec_norm(x)
        x = x + x_att
        x_att = self.enc_attn(x, encoded, encoded, mask=mask)
        x_att = self.enc_norm(x_att)
        x = x + x_att
        y = self.ff(x)
        y = self.ff_norm(y)
        return x + y
