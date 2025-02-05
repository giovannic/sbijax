import abc
from abc import ABC

from typing import Tuple
from jaxtyping import Array
import arviz as az

from ._sbi_base import SBI

class NE(SBI, ABC):
    """Sequential neural estimation base class."""

    def __init__(
        self,
        network
        ):
        """Construct an SNE object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            network: a neural network
            indices: a dictionary of indices for infinite dimensional
                    priors or observations
        """
        self.model = network
        self.n_total_simulations = 0

    @abc.abstractmethod
    def sample_posterior(
        self,
        rng_key,
        params,
        observable,
        observed_index=None,
        *args,
        **kwargs
        ) -> Tuple[az.InferenceData, Array]:
        """Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: a data point
            *args: argument list
            **kwargs: keyword arguments
        """
        pass
