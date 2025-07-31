"""Utility functions for sfmpe package."""

from typing import Optional
from jax import tree, random as jr, numpy as jnp
from jaxtyping import PyTree, Array


def split_data(data: PyTree, size: int, split: float = 0.8, shuffle_rng: Optional[Array] = None) -> tuple[PyTree, PyTree]:
    """Split PyTree data into training and validation sets.
    
    Parameters
    ----------
    data : PyTree
        The data to split, where each leaf should have the same leading dimension.
    size : int
        Total size of the dataset.
    split : float, optional
        Fraction of data to use for training, by default 0.8.
    shuffle_rng : Optional[Array], optional
        JAX random key for shuffling data indices. If None, no shuffling is performed.
        
    Returns
    -------
    tuple[PyTree, PyTree]
        Training and validation data splits.
    """
    n_train = int(size * split)
    
    if shuffle_rng is not None:
        indices = jr.permutation(shuffle_rng, size)
        shuffled_data = tree.map(lambda x: x[indices], data)
        train = tree.map(lambda x: x[:n_train], shuffled_data)
        val = tree.map(lambda x: x[n_train:], shuffled_data)
    else:
        train = tree.map(lambda x: x[:n_train], data)
        val = tree.map(lambda x: x[n_train:], data)

    return train, val