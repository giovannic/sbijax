from sfmpe.nn.transformer.embedding import Embedding
from jax import numpy as jnp
import pytest

@pytest.mark.parametrize('value_dim', [1, 2])
@pytest.mark.parametrize('theta_dim', [2])
@pytest.mark.parametrize('index_dim', [2, 3])
@pytest.mark.parametrize('latent_dim', [4])
@pytest.mark.parametrize('out_dim', [24])
@pytest.mark.parametrize('batch_size', [10])
def test_embedding_outputs_correct_dimensions_basic(
    value_dim,
    theta_dim,
    index_dim,
    latent_dim,
    out_dim,
    batch_size
    ):
    values = jnp.zeros((batch_size, theta_dim, value_dim))
    labels = jnp.arange(theta_dim)
    index = jnp.zeros((batch_size, theta_dim, index_dim))

    embedding = Embedding(
        value_dim=value_dim,
        n_labels=theta_dim,
        label_dim=latent_dim,
        index_in_dim=index_dim,
        index_out_dim=latent_dim,
        out_dim=out_dim
    )

    t = 0.5

    out = embedding(values, labels, index, t)

    assert out.shape == (batch_size, theta_dim, out_dim)

@pytest.mark.parametrize('value_dim', [1, 2])
@pytest.mark.parametrize('theta_dim', [2])
@pytest.mark.parametrize('index_dim', [2, 3])
@pytest.mark.parametrize('latent_dim', [4])
@pytest.mark.parametrize('out_dim', [24])
@pytest.mark.parametrize('batch_size', [10])
def test_embedding_outputs_correct_dimensions_with_set_rvs(
    value_dim,
    theta_dim,
    index_dim,
    latent_dim,
    out_dim,
    batch_size
    ):
    values = jnp.zeros((batch_size, theta_dim, value_dim))
    labels = jnp.full((theta_dim,), 0)
    index = jnp.zeros((batch_size, theta_dim, index_dim))

    embedding = Embedding(
        value_dim=value_dim,
        n_labels=1,
        label_dim=latent_dim,
        index_in_dim=index_dim,
        index_out_dim=latent_dim,
        out_dim=out_dim
    )

    t = 0.5

    out = embedding(values, labels, index, t)

    assert out.shape == (batch_size, theta_dim, out_dim)
