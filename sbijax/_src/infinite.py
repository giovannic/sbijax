import jax.numpy as jnp
from itertools import accumulate

def index_data(data, indices, index_map):
    # create a matrix of shape data
    # with an extra dimension to hold any index
    index_shapes = [v.shape[-1] for v in indices.values()]
    index_starts = list(accumulate([0] + index_shapes))
    n_index = sum(index_shapes)
    index = jnp.full(data.shape + (n_index,), jnp.nan)

    # copy over indices
    for i, (k, v) in enumerate(indices.items()):
        start = index_starts[i]
        end = start + index_shapes[i]
        index = index.at[
            ..., # batch shape
            index_map[k], # parameter space
            start:end # index space
        ].set(v)

    return index
