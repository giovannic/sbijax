from flax import nnx
from jax import numpy as jnp
from abc import abstractmethod, ABC

Array = jnp.ndarray

class StructuredVectorFieldModel(nnx.Module, ABC):

    @abstractmethod
    def __call__(
        self,
        context: Array,
        context_label: Array,
        context_index: Array,
        context_mask: Array,
        theta: Array,
        theta_label: Array,
        theta_index: Array,
        theta_mask: Array,
        cross_mask: Array,
        time: Array,
        ) -> Array:
        return jnp.zeros_like(theta)
