import pytest
from functools import reduce
from jax import numpy as jnp, random as jr, vmap
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.sfmpe import SFMPE
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.util.dataloader import structured_as_batch_iterators

from sbijax import FMPE
from sbijax.nn import make_cnf

from flax import nnx

def prod(x):
    return reduce(lambda x, y: x * y, x)

def test_structured_loader_flattens_theta_and_y():
    sample_size = 10
    event_size = 3
    batch_shape= (2, 4)
    index_size = 6

    data = {
        'theta': {
            'x': jnp.zeros((sample_size, event_size) + batch_shape),
        },
        'theta_index': {
            'x': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
        'y': {
            'obs': jnp.zeros((sample_size, event_size) + batch_shape),
        },
        'y_index': {
            'obs': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
    }

    train_iter, _, labels = structured_as_batch_iterators(
        jr.PRNGKey(0),
        data,
        3,
        .5,
        True,
        data_batch_ndims={
            'theta': {'x': 2},
            'y': {'obs': 2}
        }
    )

    for batch in train_iter:
        train_n = batch['theta'].shape[0]
        # check that theta and y have been flattened
        target_shape = (train_n, event_size, prod(batch_shape))
        assert batch['theta'].shape == target_shape
        assert batch['y'].shape == target_shape

        # check that indices are broadcasted
        target_shape = (train_n, event_size, index_size)
        assert batch['theta_index'].shape == target_shape
        assert batch['y_index'].shape == target_shape

    # check that labels are correct shape
    assert labels['theta'].shape == (1, event_size, 1)
    assert labels['y'].shape == (1, event_size, 1)

def test_structured_loader_labels_multiple_data():
    sample_size = 10
    event_size = 3
    batch_shape= (2, 4)
    index_size = 6

    data = {
        'theta': {
            'x': jnp.zeros((sample_size, event_size) + batch_shape),
            'y': jnp.zeros((sample_size, 1) + batch_shape),
        },
        'theta_index': {
            'x': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
        'y': {
            'obs_1': jnp.zeros((sample_size, event_size) + batch_shape),
            'obs_2': jnp.zeros((sample_size, 1) + batch_shape),
        },
        'y_index': {
            'obs_1': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
    }

    _, _, labels = structured_as_batch_iterators(
        jr.PRNGKey(0),
        data,
        3,
        .5,
        True,
        data_batch_ndims={
            'theta': {'x': 2, 'y': 2},
            'y': {'obs_1': 2, 'obs_2': 2}
        },
    )

    # check that labels are correct shape
    assert labels['theta'].shape == (1, event_size + 1, 1)
    assert labels['y'].shape == (1, event_size + 1, 1)
