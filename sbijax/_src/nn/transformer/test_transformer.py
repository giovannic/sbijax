import pytest
from jax import numpy as jnp
from flax import nnx
from .transformer import Transformer

@pytest.mark.parametrize('batch_dim', [4])
@pytest.mark.parametrize('context_token_dim', [10])
@pytest.mark.parametrize('latent_dim', [12])
@pytest.mark.parametrize('n_context_labels', [3])
@pytest.mark.parametrize('context_index_dim', [2])
@pytest.mark.parametrize('context_z_stats', [(0, 1)])
@pytest.mark.parametrize('n_theta_labels', [3])
@pytest.mark.parametrize('theta_token_dim', [10])
@pytest.mark.parametrize('theta_index_dim', [2])
def test_forward(
  batch_dim,
  context_token_dim,
  latent_dim,
  n_context_labels,
  context_index_dim,
  context_z_stats,
  n_theta_labels,
  theta_index_dim,
  theta_token_dim
):

    config = {
        'latent_dim': latent_dim,
        'label_dim': 2,
        'index_out_dim': 2,
        'n_encoder': 2,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .5,
        'activation': nnx.relu,
    }

    transformer = Transformer(
        config,
        n_context_labels,
        context_index_dim,
        context_z_stats,
        n_theta_labels,
        theta_index_dim,
        rngs=nnx.Rngs(params=0)
    )

    transformer.eval()

    context = jnp.zeros((batch_dim, context_token_dim))
    context_label = jnp.zeros((context_token_dim,), dtype=jnp.int32)
    context_index = jnp.zeros((
        batch_dim,
        context_token_dim,
        context_index_dim
    ))
    pos= jnp.zeros((batch_dim, theta_token_dim))
    pos_label = jnp.zeros((theta_token_dim,), dtype=jnp.int32)
    pos_index = jnp.zeros((
        batch_dim,
        theta_token_dim,
        theta_index_dim
    ))

    vector = transformer(
        context,
        context_label,
        context_index,
        pos,
        pos_label,
        pos_index
    )

    assert vector.shape == (batch_dim, theta_token_dim, 1)
