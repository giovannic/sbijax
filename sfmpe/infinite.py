from typing import Dict, Iterable
from jaxtyping import Array
import jax.numpy as jnp
from itertools import accumulate
from jax.tree_util import tree_map_with_path

Index = Dict[str, Array]
IndexMap = Dict[str, Iterable[str]]

def index_theta(
    theta,
    indices: Index,
    index_map: IndexMap
    ):
    # index a PyTree of data
    referenced_indices = {
        k: indices[k]
        for ref in index_map.values()
        for k in ref
    }
    index_shapes = [
        v.shape[-1]
        for v in referenced_indices.values()
    ]
    index_starts = list(accumulate([0] + index_shapes))
    n_index = sum(index_shapes)

    # index a leaf
    def _index_leaf(path, leaf):
        param = path[0].key
        index = jnp.full(leaf.shape[:-1] + (n_index,), jnp.nan)
        # copy over indices
        if param in index_map:
            for i, (k, v) in enumerate(indices.items()):
                if k not in index_map[param]:
                    continue
                start = index_starts[i]
                end = start + index_shapes[i]
                index = index.at[
                    ..., # batch shape
                    start:end # index space
                ].set(v)

        return index

    return tree_map_with_path(_index_leaf, theta)
