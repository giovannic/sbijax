"""Abstract base class for transforms used in CNF."""

from abc import ABC, abstractmethod
from flax import nnx
from jax import numpy as jnp

class Transform(nnx.Module, ABC):
    """Abstract base class for transforms used in CNF."""
    
    @abstractmethod
    def __call__(
        self,
        context,
        context_label,
        context_index,
        context_mask,
        theta,
        theta_label,
        theta_index,
        theta_mask,
        cross_mask,
        time,
    ) -> jnp.ndarray:
        """Transform call signature matching Transformer.__call__."""
        jnp.array([])
