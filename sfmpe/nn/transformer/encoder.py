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
        return x

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

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs):
            return FFLayer(dim, dropout, activation, rngs=rngs)

        self.layers = create_layer(rngs)
        self.n_layers = n_layers

    def __call__(self, x) -> Array:
        @nnx.split_rngs(splits=self.n_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x)
            return x

        return forward(x, self.layers)

class EncoderBlock(nnx.Module):
    """Transformer Encoder Block."""

    def __init__(
            self,
            dim,
            n_ff,
            dropout,
            activation,
            n_heads,
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
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        
        self.ff = MLP(
            dim=dim,
            n_layers=n_ff,
            dropout=dropout,
            activation=activation,
            rngs=rngs
        )
        self.ff_norm = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x, mask=None):
        x_att = self.attn(x, x, x, mask=mask)
        x_att = self.norm(x)
        x = x + x_att

        x_ff  = self.ff(x)
        x_ff = self.ff_norm(x_ff)
        return x_ff + x

class DecoderBlock(nnx.Module):
    """Transformer Decoder Block."""

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
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.ff = MLP(
            dim=dim,
            n_layers=n_ff,
            dropout=dropout,
            activation=activation,
            rngs=rngs
        )
        self.ff_norm = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x, context, mask=None):
        x_att = self.attn(x, context, context, mask=mask)
        x_att = self.norm(x_att)
        x = x + x_att
        y = self.ff(x)
        y = self.ff_norm(y)
        return x + y
