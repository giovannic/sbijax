from sfmpe.util.dataloader import (
    flatten_structured,
    pad_multidim_event,
    combine_data
)
from sfmpe.utils import split_data
import optax
import logging
import time

from jax import numpy as jnp, random as jr, jit, vmap, tree

def train_bottom_up(
    key,
    estim,
    prior_fn,
    simulator_fn,
    global_names,
    local_names,
    n_local,
    n_rounds,
    n_simulations,
    n_epochs,
    y_observed,
    independence,
    optimiser=optax.adam(0.0003),
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
            flat_y_obs, data_slices = flatten_structured(
                {
                    'theta': data['theta'],
                    'y': y_observed
                },
                independence=independence
            )
            theta_samples_full = estim.sample_posterior(
                flat_y_obs['data']['y'],
                flat_data['labels'], #type: ignore
                data_slices['theta'], #type: ignore
                masks=flat_data['masks'], #type: ignore
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

        y_samples = simulator_fn(obs_key, theta_samples)

        z_data = {
            'theta': {
                l: theta_samples[l] #type: ignore
                for l in local_names
            },
            'y': {
                'obs': y_samples['obs']
            } | {
                g: theta_samples[g] #type: ignore
                for g in global_names
            }
        }

        z_flat, _ = flatten_structured(
            z_data,
            independence=independence
        )

        if train_data is None:
            train_data = z_flat
        else:
            train_data = combine_data(train_data, z_flat)

        # Split data for training
        split_key, key = jr.split(key)
        total_samples = train_data['data']['theta'].shape[0]
        train_split, val_split = split_data(
            train_data, 
            total_samples, 
            split=0.8, 
            shuffle_rng=split_key
        )

        logger.info(f"Training p(z|theta,y) for round {i+1} with {total_samples} samples")
        start_time = time.time()
        losses = estim.fit(
            train_split,
            val_split,
            optimizer=optimiser,
            n_iter=n_epochs,
            batch_size=100
        )
        final_train_loss = losses[0][-1]  # Last training loss
        final_val_loss = losses[1][-1]    # Last validation loss
        logger.info(f"Round {i+1} p(z|theta,y) training completed in {time.time() - start_time:.2f}s - Final train loss: {final_train_loss:.4f}, val loss: {final_val_loss:.4f}")

        # simulate from p(z_vec|theta, y_vec)
        logger.info(f"Sampling from p(z_vec|theta, y_vec) for round {i+1}")
        start_time = time.time()
        z_sim = z_data.copy()
        sim_key, key = jr.split(key)

        z_sim['y']['obs'] = jr.choice(
            sim_key,
            z_data['y']['obs'],
            shape=(n_local,),
            axis=1
        )

        # pad z to n_z
        for l in local_names:
            z_sim['theta'][l] = pad_multidim_event(
                z_sim['theta'][l],
                1,
                (n_local,)
            )

        (
            flat_z_sim,
            z_sim_slices
        ) = flatten_structured(
            z_sim,
            independence=independence
        )

        @jit
        @vmap
        def sample_for_context(context):
            post = estim.sample_posterior(
                jnp.expand_dims(context, 0),
                flat_z_sim['labels'],
                z_sim_slices['theta'],
                masks=flat_z_sim['masks'],
                n_samples=1
            )
            return tree.map(lambda leaf: leaf[0], post)

        z_vec = sample_for_context(flat_z_sim['data']['y'])
        logger.info(f"Round {i+1} posterior sampling completed in {time.time() - start_time:.2f} seconds")

        # fit p(theta,z_vec|y_vec)
        logger.info(f"Preparing data for p(theta,z_vec|y_vec) training in round {i+1}")
        data = {
            'theta': {
                l: z_vec[l] #type: ignore
                for l in local_names
            } | {
                g: z_sim['y'][g] #type: ignore
                for g in global_names
            },
            'y': {
                'obs': z_sim['y']['obs']
            }
        }

        flat_data, data_slices = flatten_structured(
            data,
            independence=independence
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
            batch_size=100
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
