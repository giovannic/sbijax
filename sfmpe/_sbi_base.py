import abc

from .util.dataloader import as_batch_iterators

class SBI(abc.ABC):
    """SBI base class."""

    def __init__(self):
        """Construct an SBI object.

        Args:
            model_fns: tuple
        """

    @staticmethod
    def as_iterators(
        rng_key, data, batch_size, percentage_data_as_validation_set
    ):
        """Convert the data set to an iterable for training.

        Args:
            rng_key: a jax random key
            data: a tuple with 'y' and 'theta' elements
            batch_size: the size of each batch
            percentage_data_as_validation_set: fraction

        Returns:
            two batch iterators
        """
        return as_batch_iterators(
            rng_key,
            data,
            batch_size,
            1.0 - percentage_data_as_validation_set,
            True,
        )
