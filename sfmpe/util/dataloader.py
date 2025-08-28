from typing import Tuple, Dict, List
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

def event_indices(shape: Tuple[int, ...], dim: int) -> List[np.ndarray]:
    """
    Generate flat (row-major) indices that reproduce “:`” slicing at a given axis.

    For a C-contiguous array `x` of shape `shape`, let:

        indices = event_indices(shape, dim)

    Then for each i in range(shape[dim]):

        x[..., i, ...].ravel() == x.flat[indices[i]]

    In other words, applying `:` to every dimension before and after `dim` and
    fixing the `dim`-th index to `i` corresponds exactly to selecting the 1D
    slice of `x.flat` at positions `indices[i]`.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The full shape of the array.
    dim : int
        The axis along which you want to “slice” with `:` semantics.

    Returns
    -------
    List[np.ndarray]
        A list of length `shape[dim]`, where each entry is a 1D array of flat
        indices into `x.flat` corresponding to `x[..., i, ...]`.
    """
    # total number of elements in the array
    total_elems = int(np.prod(shape))
    # create a flat-index array and reshape into the target shape
    flat_idx = np.arange(total_elems, dtype=int).reshape(shape)

    # for each position i along `dim`, take the slice and flatten it
    return [flat_idx.take(i, axis=dim).ravel() for i in range(shape[dim])]

def pad_multidim_event(
    arr: jnp.ndarray,
    event_start: int,
    max_shape: Tuple[int, ...],
    pad_value: float = PAD_VALUE
) -> jnp.ndarray:
    """
    Pad `arr` on the event dimensions so that arr.event_shape[i]
    becomes max_shape[i], for i in [0..len(event_shape)-1].
    We'll pad only 'on the right' side (i.e., (0, needed_pad)).
    """
    # Build the pad config for *just* the event dims:
    # event dims are in the middle of arr's shape:
    #  arr.shape = sample_dims + event_shape + batch_dims
    # We figure out how many leading sample dims, trailing batch dims, etc.
    nd = arr.ndim
    # We'll build a pad_config of length nd, each entry is (pad_left, pad_right).
    pad_config = [(0, 0)] * nd

    for i in range(len(max_shape)):
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

def pad_1d(a, new_size, pad_value=PAD_VALUE):
    return pad_multidim_event(
        a,
        event_start=1,
        max_shape=(new_size,),
        pad_value=pad_value
    )

def pad_2d(a, new_size, pad_value=PAD_VALUE):
    return pad_multidim_event(
        a,
        event_start=1,
        max_shape=(new_size, new_size),
        pad_value=pad_value
    )

def pad_2d_cross(a, new_size_x, new_size_y, pad_value=PAD_VALUE):
    return pad_multidim_event(
        a,
        event_start=1,
        max_shape=(new_size_x, new_size_y),
        pad_value=pad_value
    )

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
    slices = {}  # key -> { offset, size, original_event_shape, ... }
    current_offset = 0

    for key, leaf in data.items():
        # Possibly get the corresponding batch_ndims
        batch_ndim = batch_ndims.get(key, 1)

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
            "event_shape": event_shape,
            "batch_shape": leaf.shape[-batch_ndim:]
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
    Builds masks for cross, local and cross-local independence
    """
    total_size = sum(prod(s['event_shape']) for s in block_slices.values())

    # Build default mask NxN = 1
    mask_np = np.ones((total_size, total_size), dtype=np.int8)

    # cross independence => zero out entire sub-block
    for (blockA, blockB) in independence.get("cross", []):
        if blockA not in block_slices or blockB not in block_slices:
            continue
        offA, sizeA, shapeA = (
            block_slices[blockA]["offset"],
            prod(block_slices[blockA]["event_shape"]),
            block_slices[blockA]["event_shape"]
        )
        offB, sizeB, shapeB = (
            block_slices[blockB]["offset"],
            prod(block_slices[blockB]["event_shape"]),
            block_slices[blockB]["event_shape"]
        )

        # zero out sub-block
        mask_np[offA:offA+sizeA, offB:offB+sizeB] = 0

    # local independence => zero out self-block
    for key in independence.get("local", []):
        if key in block_slices:
            off = block_slices[key]["offset"]
            sz  = prod(block_slices[key]["event_shape"])
            mask_np[off:off+sz, off:off+sz] = 0

    # cross_local => zero out entire sub-block, re-enable diagonal or user-specified pairs
    for (blockA, blockB, idx_map) in independence.get("cross_local", []):
        if blockA not in block_slices or blockB not in block_slices:
            continue
        offA, sizeA, shapeA = (
            block_slices[blockA]["offset"],
            prod(block_slices[blockA]["event_shape"]),
            block_slices[blockA]["event_shape"]
        )
        offB, sizeB, shapeB = (
            block_slices[blockB]["offset"],
            prod(block_slices[blockB]["event_shape"]),
            block_slices[blockB]["event_shape"]
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
            # idx_map is a tuple of matching event dimensions, flatten them to find matching event blocks
            dim_a, dim_b = idx_map
            if dim_a >= len(shapeA) or dim_b >= len(shapeB):
                raise ValueError("Index map has invalid event shape dimensions")
            if shapeA[dim_a] != shapeB[dim_b]:
                raise ValueError("Cannot do cross_local if event shapes do not match")
            a_idx = event_indices(shapeA, dim_a)
            b_idx = event_indices(shapeB, dim_b)
            for (a, b) in zip(a_idx, b_idx):
                mask_np[offA+a, offB+b] = 1
                mask_np[offB+b, offA+a] = 1

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
    Q = sum(prod(b["event_shape"]) for b in query_slices.values())
    K = sum(prod(b["event_shape"]) for b in key_slices.values())
    mask_np = np.ones((Q, K), dtype=np.int8)

    # We'll ignore "local" because that's about within-block self attention,
    # not cross. We only apply cross and cross_local.

    # cross
    for (blockA, blockB) in independence.get("cross", []):
        # Check if blockA is in query and blockB is in key
        if blockA in query_slices and blockB in key_slices:
            q_off, q_size, q_shape = (query_slices[blockA]["offset"],
                                      prod(query_slices[blockA]["event_shape"]),
                                      query_slices[blockA]["event_shape"])
            k_off, k_size, k_shape = (key_slices[blockB]["offset"],
                                      prod(key_slices[blockB]["event_shape"]),
                                      key_slices[blockB]["event_shape"])
            # Zero out sub-block
            mask_np[q_off:q_off+q_size, k_off:k_off+k_size] = 0
    
    # cross_local => zero out entire sub-block
    cross_specs = independence.get("cross_local", [])
    for (blockA, blockB, idx_map) in cross_specs:
        # Check if blockA is in query and blockB is in key
        if blockA in query_slices and blockB in key_slices:
            q_off, q_size, q_shape = (query_slices[blockA]["offset"],
                                      prod(query_slices[blockA]["event_shape"]),
                                      query_slices[blockA]["event_shape"])
            k_off, k_size, k_shape = (key_slices[blockB]["offset"],
                                      prod(key_slices[blockB]["event_shape"]),
                                      key_slices[blockB]["event_shape"])
            # Zero out sub-block
            mask_np[q_off:q_off+q_size, k_off:k_off+k_size] = 0

            if idx_map is None:
                # diagonal only => must match size
                if q_size != k_size:
                    raise ValueError("Cannot do cross_local diagonal if sizes differ.")
                for i in range(q_size):
                    mask_np[q_off + i, k_off + i] = 1
            else:
                dim_a, dim_b = idx_map
                if dim_a >= len(q_shape) or dim_b >= len(k_shape):
                    raise ValueError("Index map has invalid event shape dimensions")
                if q_shape[dim_a] != k_shape[dim_b]:
                    raise ValueError("Cannot do cross_local if event shapes do not match")
                a_idx = event_indices(q_shape, dim_a)
                b_idx = event_indices(k_shape, dim_b)
                for (a, b) in zip(a_idx, b_idx):
                    mask_np[q_off+a, k_off+b] = 1

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
    block_slices: Dict[str, dict],
    event_shapes: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Build a 0/1 mask indicating which tokens are valid
    Pre: assumes block_slices are sorted by block["offset"]

    Args:
      block_slices:
        { (key): {
            "offset": <int>,
            "event_shape":  tuple,  # the padded shape
          }, ... }

      event_shapes: dict of actual event shapes,
		ndarrays are of shape [sample_shape + (n_event_dims,)].

    Returns:
      mask: jnp.ndarray of shape sample_shape + (T,),
        where T = sum of block["size"] for all blocks in block_slices.
        mask[..., t] = 1 if the token t in that sample is valid, else 0.
    """

    def _build_block_mask(key, info):
        block_size = prod(info["event_shape"])
        actual_event_shape = event_shapes[key]

        # Build coordinate grid
        ranges = [jnp.arange(r) for r in info['event_shape']]
        coords = jnp.meshgrid(*ranges, indexing="ij")

        # filter coordinates and flatten
        n_event_dims = len(info['event_shape'])
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

def flatten_structured(
    data: PyTree,
    data_sample_ndims=1,
    data_batch_ndims: Optional[PyTree]=None,
    index: Optional[PyTree]=None,
    pad_value=PAD_VALUE,
    event_shapes: Optional[PyTree]=None,
    independence: Dict={}
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
    labels = sorted(list(data['theta'].keys()) + list(data['y'].keys()))
    label_map = {
        label: i for i, label in enumerate(labels)
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

    flat_data = {
        'theta': flat_theta,
        'y': flat_y,
    }

    labels = {
        'theta': theta_labels,
        'y': context_labels
    }
    
    slices = {
        'theta': theta_slices,
        'y': y_slices
    }

    ret = {'data': flat_data, 'labels': labels}

    if event_shapes is not None or independence:
        masks = {}
        if independence:
            theta_attention = build_self_attention_mask(
                theta_slices,
                independence
            )

            y_attention = build_self_attention_mask(
                y_slices,
                independence
            )

            cross_attention = build_cross_attention_mask(
                theta_slices,
                y_slices,
                independence
            )
            masks['attention'] = {
                'y': jnp.expand_dims(y_attention, 0),
                'theta': jnp.expand_dims(theta_attention, 0),
                'cross': jnp.expand_dims(cross_attention, 0)
            }
        if event_shapes is not None:
            theta_padding_mask = build_padding_mask(
                theta_slices,
                event_shapes['theta']
            )
            y_padding_mask = build_padding_mask(
                y_slices,
                event_shapes['y']
            )
            masks['padding'] = {
                'theta': theta_padding_mask,
                'y': y_padding_mask
            }
        ret['masks'] = masks

    if index is not None:
        ret['index'] = {
            'theta': _flatten_index(
                { k: index[k] for k in index.keys() if k in data['theta'] },
                pad_value,
                data_sample_ndims
            ),
            'y': _flatten_index(
                { k: index[k] for k in index.keys() if k in data['y'] },
                pad_value,
                data_sample_ndims
            )
        }

    return ret, slices

def encode_unknown_theta(
    theta_slices,
    index=None,
    data_sample_ndims=1,
    pad_value=PAD_VALUE,
    ):

    labels = _get_flat_labels(
        theta_slices,
        data_sample_ndims
    )

    if index is not None:
        index = _flatten_index(
            { k: index[k] for k in theta_slices.keys() },
            pad_value
        )

    return labels, index

def decode_theta(
    theta: jnp.ndarray,
    theta_slices: dict[str, dict],
    sample_shape
) -> dict[str, jnp.ndarray]:
    """
    'Unflatten' the transformer's output back into a dictionary of blocks
    with known event and batch shapes.

    Args:
      theta:
        jnp.ndarray of shape [sample_shape..., tokens, max_batch_size].
      theta_slices:
        { block_name: {
            "offset": int,
            "event_shape": tuple[int,...],
            "batch_shape": tuple[int,...],
          },
          ...
        }

    Returns:
      A dict { block_name: jnp.ndarray }, where each array has shape
        [sample_shape..., *event_shape, *batch_shape].
    """

    def _decode_slice(info):
        offset = info["offset"]
        event_shape = info["event_shape"]
        batch_shape = info["batch_shape"]

        block_size = prod(event_shape)  # number of tokens for this block
        batch_size = prod(batch_shape) # actual batch size for tokens in this block

        new_shape = sample_shape + event_shape + batch_shape
        block_slice = theta[..., offset : offset + block_size, :]

        return jnp.reshape(block_slice[...,:batch_size], new_shape)

    return { k: _decode_slice(v) for k, v in theta_slices.items() }

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
    event_size = sum(prod(s['event_shape']) for s in block_slices.values())
    labels = np.zeros((1,) * sample_ndims + (event_size,), dtype=np.int8)
    for k, s in block_slices.items():
        off = s['offset']
        sz = prod(s['event_shape'])
        labels[..., off:off+sz] = label_map[k]
    return jnp.array(labels)

def _flatten_index(index, pad_value, sample_ndims=1):
    flattened = tree.leaves(
        tree.map(
            lambda leaf: _flatten_leaf(
                leaf,
                sample_ndims,
                batch_ndims=1,
                pad_value=pad_value,
                max_batch_size=leaf.shape[-1]
            ),
            index
        )
    )

    # concatenate leaves in the event dimension
    return jnp.concatenate(flattened, axis=-2)

def combine_data(x: Dict, y: Dict, pad_value=PAD_VALUE) -> Dict:
    out = {}
    out["data"] = {}

    xss, yss = x["data"]["y"].shape[0], y["data"]["y"].shape[0]

    def _combine_1d(x_leaf, y_leaf):
        tx, ty = x_leaf.shape[1], y_leaf.shape[1]
        max_t = max(tx, ty)
        x_leaf = pad_1d(x_leaf, max_t, pad_value=pad_value)
        y_leaf = pad_1d(y_leaf, max_t, pad_value=pad_value)
        return jnp.concatenate([x_leaf, y_leaf], axis=0)

    def _combine_broadcast(x_leaf, y_leaf, padder, pv=pad_value):
        tx, ty = x_leaf.shape[1], y_leaf.shape[1]
        max_t = max(tx, ty)
        x_leaf = padder(x_leaf, max_t, pad_value=pv)
        y_leaf = padder(y_leaf, max_t, pad_value=pv)
        # broadcast to sample shape
        x_leaf = jnp.broadcast_to(x_leaf, (xss,) + x_leaf.shape[1:])
        y_leaf= jnp.broadcast_to(y_leaf, (yss,) + y_leaf.shape[1:])
        return jnp.concatenate([x_leaf, y_leaf], axis=0)

    def _combine_cross_broadcast(x_leaf, y_leaf):
        tx_x, ty_x = x_leaf.shape[1], x_leaf.shape[2]
        tx_y, ty_y = y_leaf.shape[1], y_leaf.shape[2]
        max_tx = max(tx_x, tx_y)
        max_ty = max(ty_x, ty_y)
        x_leaf = pad_2d_cross(x_leaf, max_tx, max_ty, pad_value=0)
        y_leaf = pad_2d_cross(y_leaf, max_tx, max_ty, pad_value=0)
        # broadcast to sample shape
        x_leaf = jnp.broadcast_to(x_leaf, (xss,) + x_leaf.shape[1:])
        y_leaf= jnp.broadcast_to(y_leaf, (yss,) + y_leaf.shape[1:])
        return jnp.concatenate([x_leaf, y_leaf], axis=0)

    for block_key in ["theta", "y"]:
        if block_key not in x["data"] or block_key not in y["data"]:
            continue
        out["data"][block_key] = _combine_1d(x["data"][block_key], y["data"][block_key])

    if "labels" in x and "labels" in y:
        out["labels"] = {}
        for block_key in ["theta", "y"]:
            if block_key in x["labels"] and block_key in y["labels"]:
                out["labels"][block_key] = _combine_broadcast(
                    x['labels'][block_key],
                    y['labels'][block_key],
                    pad_1d,
                    pv=0
                )

    # Always create a "padding" field in the output
    out["masks"] = {}
    out["masks"]["padding"] = {}
    for block_key in ["theta", "y"]:
        # Try to get the padding mask; if missing, use a default (a 0 array of shape (1, T) based on the corresponding data)
        x_padding = x.get("masks", {}).get("padding", {}).get(block_key, None)
        y_padding = y.get("masks", {}).get("padding", {}).get(block_key, None)
        if x_padding is None:
            x_padding = jnp.ones((1, x["data"][block_key].shape[1]), dtype=x["data"][block_key].dtype)
        if y_padding is None:
            y_padding = jnp.ones((1, y["data"][block_key].shape[1]), dtype=y["data"][block_key].dtype)
        out["masks"]["padding"][block_key] = _combine_broadcast(
            x_padding,
            y_padding,
            pad_1d,
            pv=0
        )

    if "masks" in x and "masks" in y:
        if "attention" in x["masks"] and "attention" in y["masks"]:
            out["masks"]["attention"] = {}

            for block_key in ["theta", "y"]:
                if block_key in x["masks"]["attention"] and block_key in y["masks"]["attention"]:
                    out["masks"]["attention"][block_key] = _combine_broadcast(
                        x['masks']['attention'][block_key],
                        y['masks']['attention'][block_key],
                        pad_2d,
                        pv=0
                    )

            if "cross" in x["masks"]["attention"] and "cross" in y["masks"]["attention"]:

                out["masks"]["attention"]['cross'] = _combine_cross_broadcast(
                    x['masks']['attention']['cross'],
                    y['masks']['attention']['cross']
                )

    if "index" in x and "index" in y:
        out["index"] = {}
        for block_key in ["theta", "y"]:
            if block_key in x["index"] and block_key in y["index"]:
                out["index"][block_key] = _combine_broadcast(x['index'][block_key], y['index'][block_key], pad_1d)

    return out

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
    n = data["data"]["y"].shape[0]
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
        rng_key, itr, data['data']['y'].shape[0], batch_size, shuffle
    )

def as_numpy_iterator_from_slices(data: PyTree, batch_size):
    itr = tf.data.Dataset.from_tensor_slices(data)
    itr = itr.batch(batch_size).prefetch(buffer_size=batch_size)
    itr = itr.as_numpy_iterator()
    return itr
