"""Round-based training for FMPE models."""

from typing import Callable
from jax import numpy as jnp, random as jr, tree
import optax
from .fmpe import FMPE
from .utils import split_data


def train_fmpe_rounds(
    key: jnp.ndarray,
    estim: FMPE,
    prior_fn: Callable,
    simulator_fn: Callable, 
    y_observed: jnp.ndarray,
    theta_shape: tuple,
    n_rounds: int,
    n_simulations: int,
    n_epochs: int,
    optimizer: optax.GradientTransformation = optax.adam(3e-4),
    batch_size: int = 100
) -> FMPE:
    """Train FMPE model using round-based approach.
    
    Parameters
    ----------
    key : jnp.ndarray
        JAX random key
    estim : FMPE
        FMPE estimator to train
    prior_fn : Callable
        Function that takes (key, n_samples) and returns theta samples
    simulator_fn : Callable
        Function that takes (key, theta) and returns y samples
    y_observed : jnp.ndarray
        Observed data
    theta_shape : tuple
        Shape of theta parameter
    n_rounds : int
        Number of training rounds
    n_simulations : int
        Number of simulations per round
    n_epochs : int
        Number of training epochs per round
    optimizer : optax.GradientTransformation
        Optimizer for training
    batch_size : int
        Batch size for training
        
    Returns
    -------
    FMPE
        Trained FMPE estimator
    """
    
    all_data = None
    
    for round_idx in range(n_rounds):
        print(f"Round {round_idx + 1}/{n_rounds}")
        
        # Generate theta samples for this round
        if round_idx == 0:
            # First round: sample from prior
            prior_key, key = jr.split(key)
            theta_samples = prior_fn(prior_key, n_simulations)
        else:
            # Subsequent rounds: sample from posterior given observed data
            post_key, key = jr.split(key)
            theta_samples = estim.sample_posterior(
                y_observed[None, ...],
                theta_shape=theta_shape,
                n_samples=n_simulations
            )
        
        # Generate observations using simulator
        sim_key, key = jr.split(key)
        y_samples = simulator_fn(sim_key, theta_samples)
        
        # Create data structure for this round
        round_data = {
            'data': {
                'theta': theta_samples,
                'y': y_samples
            }
        }
        
        # Accumulate data across rounds using tree.map
        if all_data is None:
            all_data = round_data
        else:
            all_data = tree.map(
                lambda existing, new: jnp.concatenate([existing, new], axis=0),
                all_data,
                round_data
            )
        
        # Split data for training
        split_key, key = jr.split(key)
        total_samples = all_data['data']['theta'].shape[0]
        train_data, val_data = split_data(
            all_data, 
            total_samples, 
            split=0.8, 
            shuffle_rng=split_key
        )
        
        # Train model on accumulated data
        print(f"Training on {total_samples} samples")
        estim.fit(
            train_data,
            val_data,
            n_iter=n_epochs,
            optimizer=optimizer,
            batch_size=batch_size
        )
    
    return estim