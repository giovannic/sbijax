"""
PyTreeBijector: A TFP Bijector that applies transformations to PyTree structures.
"""
from typing import Dict, Any, Optional, Mapping
import jax.numpy as jnp
from jax import tree
import jax.tree_util as tree_util
from jaxtyping import PyTree, Array
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd


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
        bijector_tree: PyTree,
        example_tree: PyTree,
        validate_args: bool = False,
        name: str = "pytree_bijector"
    ):
        # Store the example tree structure
        self._example_leaves, self._treedef = tree.flatten(example_tree)
        
        # Create a complete bijector list matching the example structure
        self._bijector_list = self._create_bijector_list(bijector_tree, example_tree)
        
        # Get the minimum event dimensionality from all bijectors
        # Handle both scalar and dict-valued event ndims
        forward_event_ndims = []
        inverse_event_ndims = []
        
        for b in self._bijector_list:
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
    
    def _create_bijector_list(self, bijector_tree: PyTree, example_tree: PyTree) -> list:
        """Create a flat list of bijectors matching the example tree structure."""
        example_with_paths, _ = tree_util.tree_flatten_with_path(example_tree)
        
        bijector_list = []
        for path, _ in example_with_paths:
            # Try to find a bijector for this path
            bijector = self._find_bijector_for_path(path, bijector_tree)
            bijector_list.append(bijector)
        
        return bijector_list
    
    def _find_bijector_for_path(self, path, bijector_tree):
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
        # Flatten the input tree
        x_leaves, x_treedef = tree.flatten(x)
        
        # Apply bijectors to corresponding leaves
        transformed_leaves = [
            bij.forward(val) for bij, val in zip(self._bijector_list, x_leaves)
        ]
        
        # Reconstruct the tree structure
        return tree.unflatten(x_treedef, transformed_leaves)
    
    def _inverse(self, y: PyTree) -> PyTree:
        """Apply inverse transformation to PyTree."""
        # Flatten the input tree
        y_leaves, y_treedef = tree.flatten(y)
        
        # Apply inverse bijectors to corresponding leaves
        transformed_leaves = [
            bij.inverse(val) for bij, val in zip(self._bijector_list, y_leaves)
        ]
        
        # Reconstruct the tree structure
        return tree.unflatten(y_treedef, transformed_leaves)
    
    def _forward_log_det_jacobian(
        self, 
        x: PyTree, 
        event_ndims: PyTree
    ) -> Array:
        """Compute forward log determinant Jacobian."""
        # Flatten all trees to get corresponding leaves
        x_leaves, _ = tree.flatten(x)
        event_ndims_leaves, _ = tree.flatten(event_ndims)
        
        # Apply forward_log_det_jacobian to each element
        log_dets = [
            jnp.sum(bij.forward_log_det_jacobian(val, ndims))
            for bij, val, ndims in zip(self._bijector_list, x_leaves, event_ndims_leaves)
        ]
        
        # Sum all log determinants
        return jnp.sum(jnp.array(log_dets)) if log_dets else jnp.array(0.0)
    
    def _inverse_log_det_jacobian(
        self, 
        y: PyTree, 
        event_ndims: PyTree
    ) -> Array:
        """Compute inverse log determinant Jacobian."""
        # Flatten all trees to get corresponding leaves
        y_leaves, _ = tree.flatten(y)
        event_ndims_leaves, _ = tree.flatten(event_ndims)
        
        # Apply inverse_log_det_jacobian to each element
        log_dets = [
            jnp.sum(bij.inverse_log_det_jacobian(val, ndims))
            for bij, val, ndims in zip(self._bijector_list, y_leaves, event_ndims_leaves)
        ]
        
        # Sum all log determinants
        return jnp.sum(jnp.array(log_dets)) if log_dets else jnp.array(0.0)
    
    def forward_dtype(self, dtype=None, name='forward_dtype', **kwargs) -> PyTree:
        """Returns the dtype returned by forward for the provided input."""
        if dtype is None:
            # Use the bijector's default dtype structure based on example tree
            dtype_leaves = [bij.forward_dtype() for bij in self._bijector_list]
            return tree.unflatten(self._treedef, dtype_leaves)
        else:
            # Handle PyTree dtype input
            dtype_leaves, dtype_treedef = tree.flatten(dtype)
            output_dtype_leaves = [
                bij.forward_dtype(dt) for bij, dt in zip(self._bijector_list, dtype_leaves)
            ]
            return tree.unflatten(dtype_treedef, output_dtype_leaves)


def create_bijector_from_distribution(
    distribution: tfd.JointDistributionNamed
) -> PyTreeBijector:
    """Create a PyTreeBijector from a JointDistributionNamed using default bijectors.
    
    Args:
        distribution: A JointDistributionNamed distribution.
        
    Returns:
        A PyTreeBijector that transforms from constrained to unconstrained space.
    """
    # Sample from the distribution to get the structure
    import jax.random as jr
    key = jr.PRNGKey(0)
    example_sample = distribution.sample(seed=key)
    
    # Get default bijectors for the distribution
    default_bijector = distribution.experimental_default_event_space_bijector()
    
    # Create bijector tree by mapping model keys to individual bijectors
    if hasattr(default_bijector, 'bijectors'):
        # Map model keys to individual bijectors
        model_keys = list(distribution.model.keys())
        bijector_tree = dict(zip(model_keys, default_bijector.bijectors))
    else:
        # Fallback: create identity bijectors for all keys
        bijector_tree = {key: tfb.Identity() for key in example_sample.keys()}
    
    return PyTreeBijector(bijector_tree, example_sample)


def create_manual_bijector_tree(
    sample_tree: PyTree,
    bijector_specs: Optional[Mapping[str, tfb.Bijector]] = None
) -> PyTreeBijector:
    """Create a PyTreeBijector with manually specified bijectors.
    
    Args:
        sample_tree: A sample PyTree to infer structure from.
        bijector_specs: Dictionary mapping tree keys to bijectors.
                       Missing keys will use Identity bijector.
        
    Returns:
        A PyTreeBijector with specified transformations.
    """
    if bijector_specs is None:
        bijector_specs = {}
    
    # Flatten the sample tree to get structure and paths
    sample_leaves_with_path, treedef = tree_util.tree_flatten_with_path(sample_tree)
    
    # Create bijector leaves based on path information
    bijector_leaves = []
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
            bijector_leaves.append(bijector_specs[key])
        else:
            # Default to Identity bijector
            bijector_leaves.append(tfb.Identity())
    
    # Create bijector tree with same structure as sample
    bijector_tree = tree.unflatten(treedef, bijector_leaves)
    
    return PyTreeBijector(bijector_tree, sample_tree)


def _chain_with_zscale(base_bijector: PyTreeBijector, data: PyTree) -> PyTreeBijector:
    """Chain base bijector with Z-scaling based on data statistics.
    
    Args:
        base_bijector: Base PyTreeBijector to chain with Z-scaling.
        data: Representative data to compute Z-scaling statistics from.
        
    Returns:
        PyTreeBijector with Z-scaling chained after base transformations.
    """
    # Transform data to unconstrained space
    unconstrained_data = base_bijector.forward(data)
    
    # Compute mean and std in unconstrained space
    mean_leaves = [jnp.mean(leaf, axis=0) for leaf in tree.flatten(unconstrained_data)[0]]
    std_leaves = [jnp.std(leaf, axis=0) for leaf in tree.flatten(unconstrained_data)[0]]
    
    # Create Z-scaling bijectors: Chain([Scale(1/std), Shift(-mean)]) = (x - mean) / std
    zscaling_bijectors = [
        tfb.Chain([
            tfb.Scale(1.0 / jnp.maximum(std, 1e-8)),
            tfb.Shift(-mean)
        ])
        for mean, std in zip(mean_leaves, std_leaves)
    ]
    
    # Chain Z-scaling with base bijectors
    chained_bijectors = [
        tfb.Chain([zscale_bij, base_bij])
        for zscale_bij, base_bij in zip(zscaling_bijectors, base_bijector._bijector_list)
    ]
    
    # Reconstruct bijector tree structure
    bijector_tree = tree.unflatten(base_bijector._treedef, chained_bijectors)
    
    # Reconstruct example tree from base bijector
    example_tree = tree.unflatten(base_bijector._treedef, base_bijector._example_leaves)
    
    return PyTreeBijector(bijector_tree, example_tree)


def create_zscaling_bijector_from_distribution(
    distribution: tfd.JointDistributionNamed,
    representative_data: PyTree
) -> PyTreeBijector:
    """Create a PyTreeBijector with Z-scaling from a JointDistributionNamed.
    
    Args:
        distribution: A JointDistributionNamed distribution.
        representative_data: Representative data samples from the distribution
                            used to compute Z-scaling statistics.
        
    Returns:
        A PyTreeBijector that transforms from constrained to Z-scaled unconstrained space.
    """
    base_bijector = create_bijector_from_distribution(distribution)
    return _chain_with_zscale(base_bijector, representative_data)


def create_manual_zscaling_bijector_tree(
    sample_tree: PyTree,
    representative_data: PyTree,
    bijector_specs: Optional[Mapping[str, tfb.Bijector]] = None
) -> PyTreeBijector:
    """Create a PyTreeBijector with Z-scaling and manually specified bijectors.
    
    Args:
        sample_tree: A sample PyTree to infer structure from.
        representative_data: Representative data samples used to compute 
                            Z-scaling statistics in unconstrained space.
        bijector_specs: Dictionary mapping tree keys to bijectors.
                       Missing keys will use Identity bijector.
        
    Returns:
        A PyTreeBijector with Z-scaling applied after specified transformations.
    """
    base_bijector = create_manual_bijector_tree(sample_tree, bijector_specs)
    return _chain_with_zscale(base_bijector, representative_data)