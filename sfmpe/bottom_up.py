import time
import optax
import logging
from jax import numpy as jnp, random as jr, jit, vmap, tree
from typing import Callable, Optional, Iterable
from jaxtyping import Array, PyTree

from sfmpe.util.dataloader import (
    flatten_structured,
    pad_multidim_event,
    combine_data
)
from sfmpe.sfmpe import SFMPE
from sfmpe.utils import split_data

F_IN = Optional[PyTree | Callable]
F_IN_ARGS = Optional[Iterable]

def train_bottom_up(
    key: Array,
    estim: SFMPE,
    prior_fn: Callable,
    local_fn: Callable,
    simulator_fn: Callable,
    global_names: list[str],
    local_names: list[str],
    n_local: int,
    n_rounds: int,
    n_simulations: int,
    n_epochs: int,
    y_observed: PyTree,
    independence: dict,
    optimiser: optax.GradientTransformation=optax.adam(0.0003),
    batch_size: int=100,
    f_in: F_IN = None,
    f_in_args: F_IN_ARGS = None,
    f_in_args_global: F_IN_ARGS = None,
    f_in_target: Optional[PyTree] = None
    ):
    logger = logging.getLogger(__name__)
    
    data = {}
    flat_data, data_slices = ({'labels': None, 'masks': None}, {})
    train_data = None
    for i in range(n_rounds):
        logger.info(f"Starting bottom-up training round {i+1}/{n_rounds}")
        # Fit p(z|theta, y)
        theta_key, obs_key, key = jr.split(key, 3)
        if i > 0:
            if f_in is not None:
                if f_in_target is None:
                    raise ValueError(
                        "f_in_target must be provided for \
                        multi-round inference on \
                        function valued variables"
                    )

            flat_y_obs, data_slices = flatten_structured(
                {
                    'theta': data['theta'],
                    'y': y_observed
                },
                independence=independence,
                index=f_in_target
            )
            theta_samples_full = estim.sample_posterior(
                flat_y_obs['data']['y'],
                flat_data['labels'], #type: ignore
                data_slices['theta'], #type: ignore
                masks=flat_data['masks'], #type: ignore
                index=flat_y_obs.get('index'),
                n_samples=n_simulations
            )
            theta_samples = {}
            for g in global_names:
                theta_samples[g] = theta_samples_full[g]
            for l in local_names:
                choice_key, theta_key = jr.split(theta_key)
                theta_samples[l] = jr.choice(
                    choice_key,
                    theta_samples_full[l],
                    shape=(1,),
                    axis=1
                )
        else:
            theta_samples = prior_fn(1).sample(
                seed=theta_key,
                sample_shape=(n_simulations,)
            )
        if f_in is not None:
            # get a sample of f_in
            if callable(f_in):
                if f_in_args is None:
                    f_in_sample = f_in()
                else:
                    f_in_key, key = jr.split(key)
                    f_in_sample = f_in(*f_in_args).sample(
                        (n_simulations,),
                        seed=f_in_key
                    )
            else:
                f_in_sample = f_in

            # sample y
            y_samples = simulator_fn(obs_key, theta_samples, f_in_sample)
        else:
            y_samples = simulator_fn(obs_key, theta_samples)

        y_data = {
            'theta': y_samples,
            'y': theta_samples
        }

        y_flat, _ = flatten_structured(
            y_data,
            independence=independence,
            index=f_in_sample if f_in is not None else None
        )

        if train_data is None:
            train_data = y_flat
        else:
            train_data = combine_data(train_data, y_flat)


        # Split data for training
        split_key, key = jr.split(key)
        total_samples = train_data['data']['theta'].shape[0]
        train_split, val_split = split_data(
            train_data, 
            total_samples, 
            split=0.8, 
            shuffle_rng=split_key
        )

        logger.info(f"Training p(y|theta,z) for round {i+1} with {total_samples} samples")
        start_time = time.time()
        losses = estim.fit(
            train_split,
            val_split,
            optimizer=optimiser,
            n_iter=n_epochs,
            batch_size=batch_size
        )
        final_train_loss = losses[0][-1]  # Last training loss
        final_val_loss = losses[1][-1]    # Last validation loss
        logger.info(f"Round {i+1} p(y|theta,z) training completed in {time.time() - start_time:.2f}s - Final train loss: {final_train_loss:.4f}, val loss: {final_val_loss:.4f}")

        # simulate from p(y_vec|theta, z_vec)
        logger.info(f"Sampling from p(y_vec|theta,z_vec) for round {i+1}")
        start_time = time.time()

        sim_key, key = jr.split(key)

        global_samples = {
            k: v
            for k, v in theta_samples.items()
            if k in global_names
        }

        local_samples = local_fn(global_samples, n_local).sample(
            seed=sim_key,
        )

        # Resample f_in to match resampled observations if f_in is present
        if f_in is not None:
            # get a sample of f_in
            if callable(f_in):
                if f_in_args_global is None:
                    f_in_sample = f_in()
                else:
                    f_in_key, key = jr.split(key)
                    f_in_sample = f_in(*f_in_args_global).sample(
                        (n_simulations,),
                        seed=f_in_key
                    )
            else:
                f_in_sample = f_in

        # Construct y_sim from scratch to avoid reference issues
        y_sim = {
            'theta': {
                'obs': pad_multidim_event(
                    y_data['theta']['obs'],
                    1,
                    (n_local,)
                )
            },
            'y': global_samples | local_samples
        }

        (
            flat_y_sim,
            y_sim_slices
        ) = flatten_structured(
            y_sim,
            independence=independence,
            index=f_in_sample if f_in is not None else None
        )

        if f_in is not None:
            @jit
            @vmap
            def sample_for_context(context, index):
                post = estim.sample_posterior(
                    jnp.expand_dims(context, 0),
                    flat_y_sim['labels'], #type: ignore
                    y_sim_slices['theta'],
                    masks=flat_y_sim['masks'],
                    index=tree.map(lambda leaf: jnp.expand_dims(leaf, 0), index),
                    n_samples=1
                )
                return tree.map(lambda leaf: leaf[0], post)

            y_vec = sample_for_context(flat_y_sim['data']['y'], flat_y_sim['index'])
        else:
            @jit
            @vmap
            def sample_for_context(context):
                post = estim.sample_posterior(
                    jnp.expand_dims(context, 0),
                    flat_y_sim['labels'],
                    y_sim_slices['theta'],
                    masks=flat_y_sim['masks'],
                    n_samples=1
                )
                return tree.map(lambda leaf: leaf[0], post)

            y_vec = sample_for_context(flat_y_sim['data']['y'])


        logger.info(f"Round {i+1} posterior sampling completed in {time.time() - start_time:.2f} seconds")

        # fit p(theta,z_vec|y_vec)
        logger.info(f"Preparing data for p(theta,z_vec|y_vec) training in round {i+1}")
        data = {
            'theta': y_sim['y'],
            'y': y_vec
        }

        flat_data, data_slices = flatten_structured(
            data,
            independence=independence,
            index=f_in_sample if f_in is not None else None
        )

        train_data = tree.map(
            lambda leaf: leaf[:n_simulations],
            combine_data(train_data, flat_data)
        )

        train_data = combine_data(train_data, flat_data)
        
        # Split data for training
        split_key, key = jr.split(key)
        total_samples = train_data['data']['theta'].shape[0]
        train_split, val_split = split_data(
            train_data, 
            total_samples, 
            split=0.8, 
            shuffle_rng=split_key
        )

        logger.info(f"Training p(theta,z_vec|y_vec) for round {i+1} with {total_samples} samples")
        start_time = time.time()
        losses = estim.fit(
            train_split,
            val_split,
            optimizer=optimiser,
            n_iter=n_epochs,
            batch_size=batch_size
        )
        final_train_loss = losses[0][-1]  # Last training loss
        final_val_loss = losses[1][-1]    # Last validation loss
        logger.info(f"Round {i+1} p(theta,z_vec|y_vec) training completed in {time.time() - start_time:.2f}s - Final train loss: {final_train_loss:.4f}, val loss: {final_val_loss:.4f}")

    # TODO: refactor into structuring code or estimator
    return (
        flat_data['labels'],
        data_slices['theta'],
        flat_data['masks']
    )
