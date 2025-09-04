"""
PyTreeBijector: A TFP Bijector that applies transformations to PyTree structures.
"""
from typing import Optional, Mapping
import jax.numpy as jnp
from jax import tree
import jax.tree_util as tree_util
from jaxtyping import PyTree, Array
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd


def _find_bijector_for_path(path, bijector_tree):
    """Find a bijector for a given path, defaulting to Identity."""
    try:
        # Navigate through the bijector tree following the path
        current = bijector_tree
        for key in path:
            if hasattr(key, 'key'):
                # Dictionary key
                current = current[key.key]
            else:
                # List/tuple index
                current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        # Path not found, use Identity
        return tfb.Identity()


def _create_path_bijector_mapping(bijector_tree: PyTree, example_tree: PyTree) -> dict:
    """Create a mapping from tree paths to bijectors."""
    example_with_paths, _ = tree_util.tree_flatten_with_path(example_tree)
    
    path_to_bijector = {}
    for path, _ in example_with_paths:
        # Try to find a bijector for this path
        bijector = _find_bijector_for_path(path, bijector_tree)
        path_to_bijector[path] = bijector
    
    return path_to_bijector


class PyTreeBijector(tfb.Bijector):
    """A bijector that applies a PyTree of bijectors to a PyTree structure.
    
    This bijector takes a PyTree of bijectors and an example PyTree to determine
    the structure, then applies bijectors element-wise to corresponding elements 
    in input PyTrees, preserving the tree structure.
    
    Args:
        bijector_tree: A PyTree of TFP Bijector objects. Can be sparse - missing
                      elements will be filled with Identity bijectors.
        example_tree: An example PyTree that defines the expected structure.
        validate_args: Whether to validate arguments (default: False).
        name: Name of the bijector (default: "pytree_bijector").
    """
    
    def __init__(
        self,
        path_to_bijector: dict,
        example_tree: PyTree,
        validate_args: bool = False,
        name: str = "pytree_bijector"
    ):
        # Store the example tree structure
        self._example_leaves, self._treedef = tree.flatten(example_tree)
        
        # Store path-to-bijector mapping
        self._path_to_bijector = path_to_bijector
        
        # Get the minimum event dimensionality from all bijectors
        # Handle both scalar and dict-valued event ndims
        forward_event_ndims = []
        inverse_event_ndims = []
        
        for b in self._path_to_bijector.values():
            f_ndims = b.forward_min_event_ndims
            i_ndims = b.inverse_min_event_ndims
            
            
            # If event_ndims is a dict, take the max value
            if isinstance(f_ndims, dict):
                f_ndims = max(f_ndims.values()) if f_ndims else 0
            if isinstance(i_ndims, dict):
                i_ndims = max(i_ndims.values()) if i_ndims else 0
                
            forward_event_ndims.append(f_ndims)
            inverse_event_ndims.append(i_ndims)
        
        forward_min_event_ndims = max(forward_event_ndims) if forward_event_ndims else 0
        inverse_min_event_ndims = max(inverse_event_ndims) if inverse_event_ndims else 0
        
        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            inverse_min_event_ndims=inverse_min_event_ndims,
            name=name
        )
    
    
    def forward(self, x: PyTree, name: str = 'forward', **kwargs) -> PyTree:
        """Apply forward transformation to PyTree, bypassing tensor conversion."""
        return self._forward(x)
    
    def inverse(self, y: PyTree, name: str = 'inverse', **kwargs) -> PyTree:
        """Apply inverse transformation to PyTree, bypassing tensor conversion."""
        return self._inverse(y)
    
    def forward_log_det_jacobian(
        self, 
        x: PyTree, 
        event_ndims: Optional[PyTree] = None, 
        name: str = 'forward_log_det_jacobian',
        **kwargs
    ) -> Array:
        """Compute forward log det jacobian, bypassing tensor conversion."""
        if event_ndims is None:
            # Use default event_ndims of 0 for all leaves
            x_leaves, x_treedef = tree.flatten(x)
            event_ndims = tree.unflatten(x_treedef, [0] * len(x_leaves))
        return self._forward_log_det_jacobian(x, event_ndims)
    
    def inverse_log_det_jacobian(
        self, 
        y: PyTree, 
        event_ndims: Optional[PyTree] = None, 
        name: str = 'inverse_log_det_jacobian',
        **kwargs
    ) -> Array:
        """Compute inverse log det jacobian, bypassing tensor conversion."""
        if event_ndims is None:
            # Use default event_ndims of 0 for all leaves  
            y_leaves, y_treedef = tree.flatten(y)
            event_ndims = tree.unflatten(y_treedef, [0] * len(y_leaves))
        return self._inverse_log_det_jacobian(y, event_ndims)
    
    
    def _forward(self, x: PyTree) -> PyTree:
        """Apply forward transformation to PyTree."""
        # Get input tree with paths
        x_with_paths, x_treedef = tree_util.tree_flatten_with_path(x)
        
        # Apply bijectors to corresponding leaves
        transformed_leaves = [
            self._path_to_bijector[path].forward(value) 
            for path, value in x_with_paths
        ]
        
        # Reconstruct the tree structure using input tree's structure
        return tree.unflatten(x_treedef, transformed_leaves)
    
    def _inverse(self, y: PyTree) -> PyTree:
        """Apply inverse transformation to PyTree."""
        # Get input tree with paths
        y_with_paths, y_treedef = tree_util.tree_flatten_with_path(y)
        
        # Apply inverse bijectors to corresponding leaves
        transformed_leaves = [
            self._path_to_bijector[path].inverse(value)
            for path, value in y_with_paths
        ]
        
        # Reconstruct the tree structure using input tree's structure
        return tree.unflatten(y_treedef, transformed_leaves)
    
    def _forward_log_det_jacobian(
        self, 
        x: PyTree, 
        event_ndims: PyTree
    ) -> Array:
        """Compute forward log determinant Jacobian."""
        # Get input trees with paths
        x_with_paths, _ = tree_util.tree_flatten_with_path(x)
        event_ndims_with_paths, _ = tree_util.tree_flatten_with_path(event_ndims)
        
        # Apply forward_log_det_jacobian to each element
        log_dets = [
            jnp.sum(self._path_to_bijector[x_path].forward_log_det_jacobian(x_val, ndims_val))
            for (x_path, x_val), (_, ndims_val) in zip(x_with_paths, event_ndims_with_paths)
        ]
        
        # Sum all log determinants
        return jnp.sum(jnp.array(log_dets)) if log_dets else jnp.array(0.0)
    
    def _inverse_log_det_jacobian(
        self, 
        y: PyTree, 
        event_ndims: PyTree
    ) -> Array:
        """Compute inverse log determinant Jacobian."""
        # Get input trees with paths
        y_with_paths, _ = tree_util.tree_flatten_with_path(y)
        event_ndims_with_paths, _ = tree_util.tree_flatten_with_path(event_ndims)
        
        # Apply inverse_log_det_jacobian to each element
        log_dets = [
            jnp.sum(self._path_to_bijector[y_path].inverse_log_det_jacobian(y_val, ndims_val))
            for (y_path, y_val), (_, ndims_val) in zip(y_with_paths, event_ndims_with_paths)
        ]
        
        # Sum all log determinants
        return jnp.sum(jnp.array(log_dets)) if log_dets else jnp.array(0.0)
    
    def forward_dtype(self, dtype=None, name='forward_dtype', **kwargs) -> PyTree:
        """Returns the dtype returned by forward for the provided input."""
        if dtype is None:
            # Use the bijector's default dtype structure based on example tree
            # Create a tree with bijector dtypes using path-to-bijector mapping
            example_with_paths, _ = tree_util.tree_flatten_with_path(tree.unflatten(self._treedef, self._example_leaves))
            dtype_leaves = [
                self._path_to_bijector[path].forward_dtype() 
                for path, _ in example_with_paths
            ]
            return tree.unflatten(self._treedef, dtype_leaves)
        else:
            # Handle PyTree dtype input
            dtype_with_paths, dtype_treedef = tree_util.tree_flatten_with_path(dtype)
            output_dtype_leaves = [
                self._path_to_bijector[path].forward_dtype(dt) 
                for (path, dt) in dtype_with_paths
            ]
            return tree.unflatten(dtype_treedef, output_dtype_leaves)


def create_bijector_map(
    sample_tree: PyTree,
    bijector_specs: Optional[Mapping[str, tfb.Bijector]] = None
) -> dict:
    """Create a path-to-bijector mapping with manually specified bijectors.
    
    Args:
        sample_tree: A sample PyTree to infer structure from.
        bijector_specs: Dictionary mapping tree keys to bijectors.
                       Missing keys will use Identity bijector.
        
    Returns:
        Dictionary mapping tree paths to bijectors.
    """
    if bijector_specs is None:
        bijector_specs = {}
    
    # Flatten the sample tree to get structure and paths
    sample_leaves_with_path, _ = tree_util.tree_flatten_with_path(sample_tree)
    
    # Create path-to-bijector mapping
    path_to_bijector = {}
    for path, value in sample_leaves_with_path:
        # Extract key from path - handle both dict keys and list indices
        if path:
            # For dict keys, use the key directly
            # For list indices or other path elements, convert to string
            key = path[-1].key if hasattr(path[-1], 'key') else str(path[-1])
        else:
            # Empty path means root element
            key = ''
            
        if key in bijector_specs:
            path_to_bijector[path] = bijector_specs[key]
        else:
            # Default to Identity bijector
            path_to_bijector[path] = tfb.Identity()
    
    return path_to_bijector


def _chain_with_zscale(base_bijector_map: dict, data: PyTree) -> dict:
    """Chain base bijectors with Z-scaling based on data statistics.
    
    Args:
        base_bijector_map: Dictionary mapping paths to bijectors.
        data: Representative data to compute Z-scaling statistics from.
        
    Returns:
        Dictionary mapping paths to chained bijectors with Z-scaling.
    """
    # Flatten data with paths
    data_with_paths, _ = tree_util.tree_flatten_with_path(data)
    
    # Compute statistics for each path
    stats = {
        path: (
            jnp.mean(
                base_bijector_map[path].forward(leaf),
                axis=tuple(range(leaf.ndim - 1))
            ),
            jnp.std(
                base_bijector_map[path].forward(leaf),
                axis=tuple(range(leaf.ndim - 1))
            )
        )
        for path, leaf in data_with_paths
    }
    
    # Chain Z-scaling with base bijectors
    return {
        path: tfb.Chain([
            tfb.Chain([
                tfb.Scale(1.0 / jnp.maximum(stats[path][1], 1e-8)),
                tfb.Shift(-stats[path][0])
            ]),
            bijector
        ])
        for path, bijector in base_bijector_map.items()
    }


def create_zscaling_bijector_tree(
    sample_tree: PyTree,
    representative_data: PyTree,
    bijector_specs: Optional[Mapping[str, tfb.Bijector]] = None
) -> dict:
    """Create a path-to-bijector mapping with Z-scaling and manually specified bijectors.
    
    Args:
        sample_tree: A sample PyTree to infer structure from.
        representative_data: Representative data samples used to compute 
                            Z-scaling statistics in unconstrained space.
        bijector_specs: Dictionary mapping tree keys to bijectors.
                       Missing keys will use Identity bijector.
        
    Returns:
        Dictionary mapping paths to bijectors with Z-scaling applied after specified transformations.
    """
    base_bijector_map = create_bijector_map(sample_tree, bijector_specs)
    return _chain_with_zscale(base_bijector_map, representative_data)
