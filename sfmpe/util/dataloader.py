from typing import Tuple, Dict
from functools import reduce
from typing import Optional
import tensorflow as tf
from jax import numpy as jnp, Array, random as jr
from jax import tree
import jax
import numpy as np
from operator import mul

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

def prod(xs):
    return reduce(mul, xs, 1)

def flatten_index_tuple(idx_tuple, shape):
    """
    Convert multi-dim index (row-major) to a flat index.
    """
    if len(idx_tuple) == 0:
        return 0
    linear = idx_tuple[0]
    for i, dim_size in zip(idx_tuple[1:], shape[1:]):
        linear = linear * dim_size + i
    return linear

def pad_multidim_event(
    arr: jnp.ndarray,
    event_shape: Tuple[int, ...],
    event_start: int,
    max_shape: Tuple[int, ...],
    pad_value: float
) -> jnp.ndarray:
    """
    Pad `arr` on the event dimensions so that arr.event_shape[i]
    becomes max_shape[i], for i in [0..len(event_shape)-1].
    We'll pad only 'on the right' side (i.e., (0, needed_pad)).
    """
    if len(event_shape) != len(max_shape):
        raise ValueError(
            f"Incompatible event shapes: got {event_shape}, expected {max_shape}"
        )

    # Build the pad config for *just* the event dims:
    # event dims are in the middle of arr's shape:
    #  arr.shape = sample_dims + event_shape + batch_dims
    # We figure out how many leading sample dims, trailing batch dims, etc.
    nd = arr.ndim
    # We'll build a pad_config of length nd, each entry is (pad_left, pad_right).
    pad_config = [(0, 0)] * nd

    for i in range(len(event_shape)):
        dim_idx = event_start + i
        actual_size = arr.shape[dim_idx]
        needed_size = max_shape[i]
        if actual_size > needed_size:
            raise ValueError(
                f"Cannot pad dimension {i} from {actual_size} to {needed_size}; "
                f"the array is already larger than the max."
            )
        pad_amount = needed_size - actual_size
        pad_config[dim_idx] = (0, pad_amount) #type: ignore

    arr_padded = jnp.pad(arr, pad_config, constant_values=pad_value)
    return arr_padded

def _flatten_leaf(
    leaf: jnp.ndarray,
    sample_ndims: int,
    batch_ndims: int,
    pad_value: float,
    max_batch_size: int,
) -> jnp.ndarray:
    """
    Steps:
    (1) Find the event shape
    (3) Flatten the event dims -> single dimension
    (4) Flatten batch dims -> single dimension
    (5) Pad batch dims -> max_batch_size
    (6) Return the resulting array of shape [ *sample_dims, flattened_event, batch ]
    """
    # Assume leaf.shape = (sample_dims..., event_shape..., batch_dims...)
    # We'll extract them:
    event_shape = leaf.shape[sample_ndims : -batch_ndims]

    # Flatten event dims
    flatten_evt_size = prod(event_shape)
    sample_shape = leaf.shape[:sample_ndims]
    batch_size= prod(leaf.shape[-batch_ndims:])

    # Reshape
    leaf_reshaped = leaf.reshape(
        sample_shape + (flatten_evt_size,) + (batch_size,)
    )

    # Pad batch dim to max_batch_size
    leaf_reshaped = jnp.pad(
        leaf_reshaped,
        [(0, 0)] * (sample_ndims + 1) + [(0, max_batch_size - batch_size)],
        constant_values=pad_value
    )
    return leaf_reshaped

def flatten_blocks(
    data,
    sample_ndims: int,
    batch_ndims: dict,
    pad_value: float,
    max_batch_size: int,
):
    """
    A conceptual function that:
      (1) For each block in `data`, does optional event-dim padding, then flattens.
      (2) Concatenates all blocks in the *event* dimension.

    Returns a tuple of a flattened ndarra and a dictionary of metadata for building masks
    """
    # We'll store the flattened leaves plus block metadata:
    flattened_leaves = []
    slices= {}  # (outer_key,inner_key) -> { offset, size, original_event_shape, ... }
    current_offset = 0

    for key, leaf in data.items():
        # Possibly get the corresponding batch_ndims
        if batch_ndims is not None:
            batch_ndim= batch_ndims.get(key, 1)
        else:
            batch_ndim= 1

        # pad+flatten
        leaf_flat = _flatten_leaf(
            leaf,
            sample_ndims,
            batch_ndim,
            pad_value,
            max_batch_size
        )

        # leaf_flat now has shape = sample_dims + (flatten_evt_size,) + (max_batch_size,)
        block_size = leaf_flat.shape[-2]  # the "event" dimension after flatten

        event_shape = leaf.shape[sample_ndims : -batch_ndim]
        slices[key] = {
            "offset": current_offset,
            "size": block_size,
            "shape": event_shape
        }
        current_offset += block_size
        flattened_leaves.append(leaf_flat)

    # Concatenate along the event dimension = -2
    flat_data = jnp.concatenate(flattened_leaves, axis=-2)
    return flat_data, slices

def build_self_attention_mask(
    block_slices: dict,
    independence: dict
):
    """
    Builds masks for local and cross-local independence
    """
    total_size = sum(s['size'] for s in block_slices)

    # Build default mask NxN = 1
    mask_np = np.ones((total_size, total_size), dtype=np.int8)

    # local independence => zero out self-block
    for key in independence.get("local", []):
        if key in block_slices:
            off = block_slices[key]["offset"]
            sz  = block_slices[key]["size"]
            mask_np[off:off+sz, off:off+sz] = 0

    # cross_local => zero out entire sub-block, re-enable diagonal or user-specified pairs
    for (blockA, blockB, idx_map) in independence.get("cross_local", []):
        if blockA not in block_slices or blockB not in block_slices:
            continue
        offA, sizeA, shapeA = (
            block_slices[blockA]["offset"],
            block_slices[blockA]["size"],
            block_slices[blockA]["shape"]
        )
        offB, sizeB, shapeB = (
            block_slices[blockB]["offset"],
            block_slices[blockB]["size"],
            block_slices[blockB]["shape"]
        )

        # zero out sub-block
        mask_np[offA:offA+sizeA, offB:offB+sizeB] = 0
        mask_np[offB:offB+sizeB, offA:offA+sizeA] = 0

        if idx_map is None:
            # diagonal only => must have sizeA == sizeB
            if sizeA != sizeB:
                raise ValueError("Cannot do diagonal cross_local if block sizes differ")
            for i in range(sizeA):
                mask_np[offA+i, offB+i] = 1
                mask_np[offB+i, offA+i] = 1
        else:
            # idx_map is list of pairs of multi-dim coords => flatten them
            if shapeA is None or shapeB is None:
                raise ValueError("We cannot interpret multi-dim coords if shape is None")
            for (coordsA, coordsB) in idx_map:
                iA = flatten_index_tuple(coordsA, shapeA)
                iB = flatten_index_tuple(coordsB, shapeB)
                if iA >= sizeA or iB >= sizeB:
                    raise IndexError(f"Index out of range for cross_local map: {coordsA}/{coordsB}")
                mask_np[offA + iA, offB + iB] = 1
                mask_np[offB + iB, offA + iA] = 1

    mask = jnp.array(mask_np, dtype=jnp.float32)
    return mask

def build_cross_attention_mask(
    query_slices,
    key_slices,
    independence
    ):
    """
    Returns a (Q, K) mask of 1=allowed, 0=blocked
    Only 'cross_local' rules that connect a block in query_slices
    to a block in key_slices are applied.
    """
    Q = sum(b["size"] for b in query_slices.values())
    K = sum(b["size"] for b in key_slices.values())
    mask_np = np.ones((Q, K), dtype=np.int8)

    # We'll ignore "local" because that's about within-block self attention,
    # not cross. We only apply cross_local.
    cross_specs = independence.get("cross_local", [])
    for (blockA, blockB, idx_map) in cross_specs:
        # Check if blockA is in query and blockB is in key
        if blockA in query_slices and blockB in key_slices:
            q_off, q_size, q_shape = (query_slices[blockA]["offset"],
                                      query_slices[blockA]["size"],
                                      query_slices[blockA].get("shape"))
            k_off, k_size, k_shape = (key_slices[blockB]["offset"],
                                      key_slices[blockB]["size"],
                                      key_slices[blockB].get("shape"))
            # Zero out sub-block
            mask_np[q_off:q_off+q_size, k_off:k_off+k_size] = 0

            if idx_map is None:
                # diagonal only => must match size
                if q_size != k_size:
                    raise ValueError("Cannot do cross_local diagonal if sizes differ.")
                for i in range(q_size):
                    mask_np[q_off + i, k_off + i] = 1
            else:
                # user-specified pairs => flatten multi-dim indices if needed
                for (coordsA, coordsB) in idx_map:
                    iA = flatten_index_tuple(coordsA, q_shape)
                    iB = flatten_index_tuple(coordsB, k_shape)
                    mask_np[q_off + iA, k_off + iB] = 1

    return jnp.array(mask_np, dtype=jnp.float32)

def _get_max_batch_size(data, batch_ndims_tree):
    return max(
        tree.leaves(
            tree.map(
                lambda leaf, ndims: prod(leaf.shape[-ndims:]),
                data,
                batch_ndims_tree
            )
        )
    )


def build_padding_mask(
    event_shapes: Dict[str, jnp.ndarray],
    block_slices: Dict[str, dict],
) -> jnp.ndarray:
    """
    Build a 0/1 mask indicating which tokens are valid
    Pre: assumes block_slices are sorted by block["offset"]

    Args:
      event_shapes: dict of actual event shapes,
		ndarrays are of shape [sample_shape + (n_event_dims,)].

      block_slices:
        { (key): {
            "offset": <int>,
            "size":   <int>,  # should == prod(shape)
            "shape":  tuple,  # the padded shape
          }, ... }

    Returns:
      mask: jnp.ndarray of shape sample_shape + (T,),
        where T = sum of block["size"] for all blocks in block_slices.
        mask[..., t] = 1 if the token t in that sample is valid, else 0.
    """

    def _build_block_mask(key, info):
        block_size = info["size"]
        actual_event_shape = event_shapes[key]

        # Build coordinate grid
        ranges = [jnp.arange(r) for r in info['shape']]
        coords = jnp.meshgrid(*ranges, indexing="ij")

        # filter coordinates and flatten
        n_event_dims = len(info['shape'])
        valid_in_dimension = [
           coord < jnp.expand_dims(actual_event_shape[..., i], range(-n_event_dims, 0))
           for i, coord in enumerate(coords)
        ]
        is_valid = jnp.all(
            jnp.stack(valid_in_dimension),
            axis=0
        )
        sample_shape = actual_event_shape.shape[:-1]
        is_valid_flat = is_valid.reshape(*sample_shape, block_size)

        return is_valid_flat.astype(jnp.float32)

    return jnp.concatenate(
        [
            _build_block_mask(key, info)
            for key, info in block_slices.items()
        ],
        axis=-1
    )

def structured_as_batch_iterators(
    rng_key: Array,
    data: PyTree,
    batch_size,
    split,
    shuffle,
    data_sample_ndims=1,
    data_batch_ndims: Optional[PyTree]=None,
    pad_value=PAD_VALUE,
    event_shapes: Optional[PyTree]=None,
):
    batch_ndims_tree = _get_batch_ndims(data, data_batch_ndims)

    max_batch_size = _get_max_batch_size(data, batch_ndims_tree)

    flat_theta, theta_slices = flatten_blocks(
        data['theta'],
        data_sample_ndims,
        batch_ndims_tree['theta'],
        pad_value,
        max_batch_size
    )

    flat_y, y_slices = flatten_blocks(
        data['y'],
        data_sample_ndims,
        batch_ndims_tree['y'],
        pad_value,
        max_batch_size
    )

    flat_theta_index = _flatten_index(
        data['theta_index'],
        pad_value
    )

    flat_y_index = _flatten_index(
        data['y_index'],
        pad_value
    )

    theta_attention = build_self_attention_mask(
        theta_slices,
        data['independence']
    )

    y_attention = build_self_attention_mask(
        y_slices,
        data['independence']
    )


    theta_padding_mask = y_padding_mask = None
    if event_shapes is not None:
        sample_shape = jnp.shape(flat_y)[:data_sample_ndims]
        theta_padding_mask = build_padding_mask(
            theta_slices,
            event_shapes['theta']
        )
        theta_padding_mask = jnp.broadcast_to(
            theta_padding_mask,
            sample_shape + theta_padding_mask.shape[data_sample_ndims:]
        )

        y_padding_mask = build_padding_mask(
            y_slices,
            event_shapes['y']
        )
        y_padding_mask = jnp.broadcast_to(
            y_padding_mask,
            sample_shape + y_padding_mask.shape[data_sample_ndims:]
        )

    cross_attention = build_cross_attention_mask(
        y_slices,
        theta_slices,
        data['independence']
    )

    labels = data['theta'].keys() + data['y'].keys()
    label_map = {
        label: i for (i, label) in enumerate(labels)
    }

    theta_labels = _get_flat_labels(
        theta_slices,
        label_map,
        data_sample_ndims
    )
    context_labels = _get_flat_labels(
        y_slices,
        label_map,
        data_sample_ndims
    )

    data = {
        'theta': flat_theta,
        'theta_index': flat_theta_index,
        'theta_padding_mask': theta_padding_mask,
        'y': flat_y,
        'y_index': flat_y_index,
        'y_padding_mask': y_padding_mask,
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
    masks = {
        'theta': theta_attention,
        'y': y_attention,
        'cross': cross_attention
    }
    return train_iter, val_iter, labels, masks

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

def _get_flat_labels(block_slices, label_map, sample_ndims=1):
    event_size = sum(s['size'] for s in block_slices.values())
    labels = np.zeros((1,) * sample_ndims + (event_size,), dtype=np.int8)
    for k, s in block_slices.items():
        off = s['offset']
        sz = s['size']
        labels[..., off:off+sz] = label_map[k]
    return jnp.array(labels)

def _flatten_index(index, pad_value):
    batch_sizes= [
        leaf.shape[-1]
        for leaf in tree.leaves(index)
    ]
    batch_start = [0] + cumsum(batch_sizes)
    batch_size = batch_start[-1]

    def _one_hot(i, leaf, pad_value):
        return jnp.full(
            leaf.shape[:-1] + (batch_size,),
            pad_value
        ).at[..., batch_start[i]:batch_start[i+1]].set(leaf)

    index = [
        _flatten_leaf(
            _one_hot(i, leaf, pad_value),
            sample_ndims=1,
            batch_ndims=1,
            pad_value=pad_value,
            max_batch_size=batch_size
        )
        for i, leaf in enumerate(tree.leaves(index))
    ]

    # concatenate leaves in the event dimension
    return jnp.concatenate(index, axis=-2)

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
