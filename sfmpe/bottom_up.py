from sfmpe.util.dataloader import (
    flatten_structured,
    flat_as_batch_iterators,
    pad_multidim_event,
    combine_data
)

from jax import numpy as jnp, random as jr, vmap, tree

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
    independence
    ):

    data = {}
    flat_data, data_slices = (None, {})
    train_data = None
    for _ in range(n_rounds):
        # Fit p(z|theta, y)
        theta_key, obs_key, key = jr.split(key, 3)
        if flat_data is not None and data_slices:
            flat_y_obs, data_slices = flatten_structured(
                {
                    'theta': data['theta'],
                    'y': y_observed
                },
                independence=independence
            )
            theta_samples_full = estim.sample_structured_posterior(
                theta_key,
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

        y_samples= simulator_fn(obs_key, theta_samples)

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

        print(tree.map(lambda x: x.shape, z_data))

        z_flat, _ = flatten_structured(
            z_data,
            independence=independence
        )

        if train_data is None:
            train_data = z_flat
        else:
            train_data = combine_data(train_data, z_flat)

        #TODO: where is the padding mask??
        print(tree.map(lambda x: x.shape, train_data))

        itr_key, key = jr.split(key)
        train_iter, val_iter = flat_as_batch_iterators(
            itr_key,
            train_data
        )

        fit_key, key = jr.split(key)

        estim.fit(
            fit_key,
            train_iter,
            val_iter,
            n_iter=n_epochs
        )

        # simulate from p(z_vec|theta, y_vec)
        z_sim = z_data.copy()
        sim_key, key = jr.split(key)
        z_sim['y']['obs'] = vmap(
            lambda k: jr.choice(k, z_data['y']['obs'], shape=(n_local,)) # type: ignore
        )(jr.split(sim_key, n_simulations))

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

        sample_key, key = jr.split(key)

        z_vec = vmap(
            lambda key, obs: tree.map(
                lambda leaf: leaf[0], #TODO: clean up
                estim.sample_structured_posterior(
                    key,
                    jnp.expand_dims(obs, 0),
                    flat_z_sim['labels'],
                    z_sim_slices['theta'],
                    masks=flat_z_sim['masks'],
                    n_samples=1
                )
            )
        )(jr.split(sample_key, n_simulations), flat_z_sim['data']['y'])

        # fit p(theta,z_vec|y_vec)
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

        print(tree.map(lambda x: x.shape, data))

        flat_data, data_slices = flatten_structured(
            data,
            independence=independence
        )

        train_data = tree.map(
            lambda leaf: leaf[:n_simulations],
            combine_data(train_data, flat_data)
        )

        train_data = combine_data(train_data, flat_data)

        train_key, itr_key, key = jr.split(key, 3)
        train_iter, val_iter = flat_as_batch_iterators(
            itr_key,
            train_data
        )

        estim.fit(
            train_key,
            train_iter,
            val_iter,
            n_iter=n_epochs
        )

        # TODO: refactor into structuring code or estimator
        return (
            flat_data['labels'],
            data_slices['theta'],
            flat_data['masks']
        )


def get_posterior(
    key,
    estim,
    labels,
    slices,
    masks,
    n_samples,
    theta_truth,
    y_observed,
    independence
    ):
    obs_flat, _ = flatten_structured(
        { 'theta': theta_truth, 'y': y_observed },
        independence=independence
    )

    posterior = estim.sample_structured_posterior( 
        key,
        obs_flat['data']['y'],
        labels,
        slices,
        masks=masks,
        n_samples=n_samples
    )

    return posterior
