from functools import reduce
from typing import Optional
import tensorflow as tf
from jax import numpy as jnp, Array, random as jr
from jax import tree
import jax

from .types import PyTree
from itertools import accumulate

PAD_VALUE = -1e8

# cumulative sum in pure python
def cumsum(x):
    return list(accumulate(x))

# pylint: disable=missing-class-docstring,too-few-public-methods
class DataLoader:
    # noqa: D101
    def __init__(self, itr, num_samples):  # noqa: D107
        self._itr = itr
        self.num_samples = num_samples

    def __iter__(self):
        """Iterate over the data set."""
        yield from self._itr.as_numpy_iterator()

def prod(x):
    return reduce(lambda x, y: x * y, x)

def structured_as_batch_iterators(
    rng_key: Array,
    data: PyTree,
    batch_size,
    split,
    shuffle,
    data_sample_ndims=1,
    data_batch_ndims: Optional[PyTree]=None,
    index_map=None,
    pad_value=PAD_VALUE
):
    batch_ndims_tree = _get_batch_ndims(data, data_batch_ndims)

    # flatten event dims for tokenisation
    data = _flatten_event_dims(data, data_sample_ndims, batch_ndims_tree)

    theta_labels = _get_flat_labels(
        data['theta'],
        data_sample_ndims
    )
    context_labels = _get_flat_labels(
        data['y'],
        data_sample_ndims
    )

    data = {
        'theta': _flatten_data(
            data['theta'],
            data_sample_ndims,
            batch_ndims_tree['theta'],
            pad_value
        ),
        'theta_index': _flatten_index(
            data['theta_index'],
            pad_value,
            index_map
        ),
        'y': _flatten_data(
            data['y'],
            data_sample_ndims,
            batch_ndims_tree['y'],
            pad_value
        ),
        'y_index': _flatten_index(
            data['y_index'],
            pad_value,
            index_map
        )
    }

    train_iter, val_iter = as_batch_iterators(
        rng_key,
        data,
        batch_size,
        split,
        shuffle
    )
    labels = {
        'theta': theta_labels,
        'y': context_labels
    }
    return train_iter, val_iter, labels

def encode_context(
    data,
    data_sample_ndims=1,
    data_batch_ndims=None,
    pad_value=PAD_VALUE,
    index_map=None
    ):
    batch_ndims_tree = _get_batch_ndims(data, data_batch_ndims)
    
    # flatten event dims for tokenisation
    data = _flatten_event_dims(
        data,
        data_sample_ndims,
        batch_ndims_tree
    )

    labels = _get_flat_labels(
        data['y'],
        data_sample_ndims
    )

    data = {
        'y': _flatten_data(
            data['y'],
            data_sample_ndims,
            batch_ndims_tree['y'],
            pad_value
        ),
        'y_index': _flatten_index(
            data['y_index'],
            pad_value,
            index_map
        )
    }

    return data, labels

def encode_unknown_theta(
    data,
    data_sample_ndims=1,
    data_batch_ndims=None,
    pad_value=PAD_VALUE,
    index_map=None
    ):
    batch_ndims_tree = _get_batch_ndims(data, data_batch_ndims)
    
    # flatten event dims for tokenisation
    data = _flatten_event_dims(
        data,
        data_sample_ndims,
        batch_ndims_tree
    )

    labels = _get_flat_labels(
        data['theta_index'],
        data_sample_ndims
    )

    index= _flatten_index(
        data['theta_index'],
        pad_value,
        index_map
    )

    return index, labels

def decode_theta(
    theta,
    labels,
    sample_shape,
    event_shapes,
    batch_shapes=None
    ):
    if batch_shapes is None:
        batch_shapes = tree.map(lambda _: (1,), event_shapes)

    # split theta into rvs
    sorted_event_shapes = [
        event_shapes[k]
        for k in batch_shapes.keys()
    ]
    event_sizes= [
        prod(s)
        for s in sorted_event_shapes
    ]

    # find batch sizes
    batch_sizes = [
        prod(shape)
        for shape in batch_shapes.values()
    ]

    # decode theta
    return {
        k: v[...,:s].reshape(sample_shape + e_s + b_s)
        for k, v, s, e_s, b_s
        in zip(
            labels,
            jnp.split(theta, event_sizes, axis=len(sample_shape)),
            batch_sizes,
            sorted_event_shapes,
            batch_shapes.values()
        )
    }

def _get_batch_ndims(data, data_batch_ndims):
    if data_batch_ndims is None:
        return tree.map(lambda _: 1, data)
    else:
        batch_ndims_tree = data_batch_ndims
        # set index batch ndims to 1
        if 'theta_index' in data:
            batch_ndims_tree['theta_index'] = tree.map(lambda _: 1, data['theta_index'])
        if 'y_index' in data:
            batch_ndims_tree['y_index'] = tree.map(lambda _: 1, data['y_index'])
        return batch_ndims_tree

def _flatten_event_dims(data, sample_ndims, batch_ndims_tree):
    return tree.map(
        lambda x, batch_ndims: jnp.reshape(
            x,
            x.shape[:sample_ndims] + (-1,) + x.shape[-batch_ndims:]
        ),
        data,
        batch_ndims_tree
    )

def _get_flat_labels(data, sample_ndims=1):
    # get label data by mapping dict keys to integers,
    # broadcasting them to the leaf shape,
    # and then flattening
    labels = list(data.keys())
    label_map = dict(zip(
        labels,
        range(len(labels))
    ))

    return jnp.concatenate([
        jnp.full((leaf.shape[sample_ndims],), label_map[k])
        for k, leaf in data.items()
    ])[jnp.newaxis, :]

def _flatten_data(data, sample_ndims, batch_ndims, pad_value):
    # pre: data is a pytree and all leaves have a single event dimension
    # find max batch size
    batch_sizes = tree.leaves(
        tree.map(
            lambda leaf, ndims: prod(leaf.shape[-ndims:]),
            data,
            batch_ndims
        )
    )
    max_batch_size = max(batch_sizes)

    def _pad_args(n):
        return ((0, 0),) * (sample_ndims + 1) + ((0, n),)

    # flatten and pad to max batch size
    padded = [
        jnp.pad(
            jnp.reshape(
                leaf,
                leaf.shape[:sample_ndims] + # sample shape
                (-1, batch_size)# event and batch shape
            ),
            _pad_args(max_batch_size - batch_size),
            mode='constant',
            constant_values=pad_value
        )
        for leaf, batch_size
        in zip(tree.leaves(data), batch_sizes)
    ]

    # and then concatenate all leaves in the event dimension
    return jnp.concatenate(tree.leaves(padded), axis=-2)

def _flatten_index(index, pad_value, index_map=None):
    # pre: index is a pytree with a single batch and event dimension
    # extend batch shape to the sum of all batch shapes
    batch_sizes= [
        leaf.shape[-1]
        for leaf in tree.leaves(index)
    ]
    batch_start = [0] + cumsum(batch_sizes)
    batch_size = batch_start[-1]

    # one-hot encode batch dimension
    if index_map is None:
        one_hot = [
            jnp.full(
                leaf.shape[:-1] + (batch_size,),
                pad_value
            ).at[..., batch_start[i]:batch_start[i+1]].set(leaf)
            for i, leaf in enumerate(tree.leaves(index))
        ]
    else:
        # one-hot encode batch dimension
        one_hot = tree.leaves(
            tree.map(
                lambda leaf, dim_map: jnp.full(
                    leaf.shape[:-1] + (batch_size,),
                    pad_value
                ).at[..., dim_map].set(leaf),
                index,
                index_map
            )
        )

    # concatenate leaves in the event dimension
    return jnp.concatenate(one_hot, axis=-2)

def _pad_args(sample_ndims, event_n, batch_n):
    return ((0, 0),) * sample_ndims + ((0, event_n), (0, batch_n))



# pylint: disable=missing-function-docstring
def as_batch_iterators(
    rng_key: Array, data: PyTree, batch_size, split, shuffle
):
    """Create two data batch iterators from a data set.

    Args:
        rng_key: a jax random key
        data: a named tuple with elements 'y' and 'theta' all data
        batch_size: size of each batch
        split: fraction of data to use for training data set. Rest is used
            for validation data set.
        shuffle: shuffle the data set or no

    Returns:
        returns two iterators
    """
    n = data["y"].shape[0]
    n_train = int(n * split)

    if shuffle:
        idxs = jr.permutation(rng_key, jnp.arange(n))
        data = jax.tree_util.tree_map(lambda x: x[idxs], data)

    y_train = jax.tree_util.tree_map(lambda x: x[:n_train], data)
    y_val = jax.tree_util.tree_map(lambda x: x[n_train:], data)

    train_rng_key, val_rng_key = jr.split(rng_key)

    train_itr = as_batch_iterator(train_rng_key, y_train, batch_size, shuffle)
    val_itr = as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

    return train_itr, val_itr


def as_batched_numpy_iterator_from_tf(
    rng_key: Array, data: tf.data.Dataset, iter_size, batch_size, shuffle
):
    """Create a data batch iterator from a tensorflow data set.

    Args:
        rng_key: a jax random key
        data: a named tuple with elements 'y' and 'theta' all data
        iter_size: total number of elements in the data set
        batch_size: size of each batch
        shuffle: shuffle the data set or no

    Returns:
        a tensorflow iterator
    """
    # hack, cause the tf stuff doesn't support jax keys :)
    max_int32 = jnp.iinfo(jnp.int32).max
    seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)

    data = (
        data.shuffle(
            10 * batch_size,
            seed=int(seed),
            reshuffle_each_iteration=shuffle,
        )
        .batch(batch_size)
        .prefetch(buffer_size=batch_size)
    )
    return DataLoader(data, iter_size)


# pylint: disable=missing-function-docstring
def as_batch_iterator(rng_key: Array, data: PyTree, batch_size, shuffle):
    """Create a data batch iterator from a data set.

    Args:
        rng_key: a jax random key
        data: a named tuple with elements 'y' and 'theta' all data
        batch_size: size of each batch
        shuffle: shuffle the data set or no

    Returns:
        a tensorflow iterator
    """
    itr = tf.data.Dataset.from_tensor_slices(data)
    return as_batched_numpy_iterator_from_tf(
        rng_key, itr, data["y"].shape[0], batch_size, shuffle
    )

def as_numpy_iterator_from_slices(data: PyTree, batch_size):
    itr = tf.data.Dataset.from_tensor_slices(data)
    itr = itr.batch(batch_size).prefetch(buffer_size=batch_size)
    itr = itr.as_numpy_iterator()
    return itr
