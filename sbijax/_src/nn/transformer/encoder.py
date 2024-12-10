from typing import Callable
from jaxtyping import Array
from flax import nnx

class FFLayer(nnx.Module):

    activation: Callable

    def __init__(self, dim, dropout, activation, rngs):
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
        rngs
        ):

        @nnx.split_rngs(splits=n_layers) #type: ignore
        @nnx.vmap(in_axes=(0,), out_axes=0) #type: ignore
        def create_layers(rngs: nnx.Rngs):
            return FFlayer(dim, dropout, activation, rngs=rngs)

        self.layers= create_layers(rngs)
        self.n_layers = n_layers

    def __call__(self, x) -> Array:
        @nnx.split_rngs(splits=self.num_layers) #type: ignore
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)  #type: ignore
        def forward(x, layer):
              return layer(x)

        return forward(x, self.layers)

class EncoderBlock(nnx.Module):
    """Transformer Encoder Block."""

    def __init__(
            self,
            num_heads,
            dim,
            n_ff,
            dropout,
            activation,
            rngs
        ):
        
        self.attn = nnx.MultiHeadAttention(
            in_features=dim,
            num_heads=num_heads,
            qkv_features=dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=dropout,
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
            num_heads,
            dim,
            n_ff,
            dropout,
            activation,
            rngs
        ):
        self.dec_attn = nnx.MultiHeadAttention(
            in_features=dim,
            num_heads=num_heads,
            qkv_features=dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=dropout,
            rngs=rngs
        )
        self.dec_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.enc_attn = nnx.MultiHeadAttention(
            in_features=dim,
            num_heads=num_heads,
            qkv_features=dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=dropout,
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
