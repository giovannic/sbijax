import pytest
import jax.numpy as jnp
from flax import nnx
from .encoder import EncoderBlock, EncoderDecoderBlock

@pytest.mark.parametrize('batch_dim', [2])
@pytest.mark.parametrize('token_dim', [3])
@pytest.mark.parametrize('embed_dim', [4])
@pytest.mark.parametrize('n_heads', [2])
def test_encoder_block_forward_pass_dims(
    batch_dim,
    token_dim,
    embed_dim,
    n_heads
    ):
    x= jnp.zeros((batch_dim, token_dim, embed_dim))

    encoder_block = EncoderBlock(
        n_heads=n_heads,
        dim=embed_dim,
        n_ff=2,
        dropout=.5,
        activation=nnx.relu,
    )

    encoder_block.eval()
    out = encoder_block(x)

    assert out.shape == (batch_dim, token_dim, embed_dim)

@pytest.mark.parametrize('batch_dim', [2])
@pytest.mark.parametrize('token_dim', [3])
@pytest.mark.parametrize('enc_token_dim', [5])
@pytest.mark.parametrize('embed_dim', [4])
@pytest.mark.parametrize('n_heads', [2])
def test_encoder_decoder_block_forward_pass_dims(
    batch_dim,
    token_dim,
    enc_token_dim,
    embed_dim,
    n_heads
    ):
    x= jnp.zeros((batch_dim, token_dim, embed_dim))
    encoded = jnp.zeros((batch_dim, enc_token_dim, embed_dim))

    encoder_block = EncoderDecoderBlock(
        n_heads=n_heads,
        dim=embed_dim,
        n_ff=2,
        dropout=.5,
        activation=nnx.relu,
    )

    encoder_block.eval()
    out = encoder_block(x, encoded)

    assert out.shape == (batch_dim, token_dim, embed_dim)
