import pytest

from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.fmpe import SFMPE
from sfmpe.bottom_up import train_bottom_up, get_posterior
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer

from sbijax import FMPE
from sbijax.nn import make_cnf

from flax import nnx

def run():
    tol = 1e-3
    n_rounds = 10
    n_simulations = 1_000
    n_epochs = 1_000
    n_post_samples = 10
    var_theta = 1.
    var_mu = 1.
    var_obs = 1.
    n_obs = 10
    n_theta = 5

    independence = {
        'local': ['theta', 'obs'],
        'cross_local': [('theta', 'obs', (0, 0))],
    }

    # make prior distribution
    def prior_fn(n):
        prior = tfd.JointDistributionNamed(
            dict(
                mu = tfd.Normal(jnp.zeros((1, 1)), jnp.full((1, 1), var_mu)), #type:ignore
                theta = lambda mu: tfd.Independent(
                    tfd.Normal(
                        jnp.repeat(mu, n, axis=-2),
                        var_theta
                    ),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=1,
        )
        return prior

    # make simulator function
    def simulator_fn(seed, theta):
        obs = tfd.Independent(
            tfd.Normal(theta['theta'], var_obs),
            reinterpreted_batch_ndims=1
        ).sample((n_obs,), seed=seed)
        obs = jnp.transpose(obs, (1, 2, 0, 3)) #type:ignore
        return {
            'obs': obs
        }

    key = jr.PRNGKey(0)

    theta_key, y_key, key = jr.split(key, 3)
    theta_truth = prior_fn(n_theta).sample((1,), seed=theta_key)
    y_observed = simulator_fn(y_key, theta_truth)

    rngs = nnx.Rngs(0)
    config = {
        'latent_dim': 64,
        'label_dim': 8,
        'index_out_dim': 0,
        'n_encoder': 1,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }

    nn = Transformer(
        config,
        value_dim=1,
        n_labels=3,
        index_dim=0,
        rngs=rngs
    )

    model = CNF(transform=nn)

    estim = SFMPE(model)

    train_key, key = jr.split(key)
    labels, slices, masks = train_bottom_up(
        train_key,
        estim,
        prior_fn,
        simulator_fn,
        ['mu'],
        ['theta'],
        n_theta,
        n_rounds,
        n_simulations,
        n_epochs,
        y_observed,
        independence
    )

    post_key, key = jr.split(key)
    posterior = get_posterior(
        post_key,
        estim,
        labels,
        slices,
        masks,
        n_post_samples,
        theta_truth,
        y_observed,
        independence
    )

    print('sbijax')

    # flatten y_observed
    sbijax_y_observed = y_observed['obs'].reshape((1, -1)) # type: ignore
    sbijax_nn = make_cnf(1 + n_theta)
    def sbijax_prior_fn():
        prior = tfd.JointDistributionNamed(
            dict(
                mu = tfd.Normal(0., var_mu),
                theta = lambda mu: tfd.Independent(
                    tfd.Normal(
                        jnp.full((n_theta,), mu),
                        var_theta
                    ),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=1,
            use_vectorized_map=True
        )

        return tfd.JointDistributionNamed(
            dict(
                theta=tfd.Blockwise(prior)
            ),
            batch_ndims=1,
        )

    def sbijax_simulator_fn(seed, theta):
        arg = {
            'mu': theta['theta'][...,:1, None],
            'theta': theta['theta'][...,1:, None]
        }
        n_samples = arg['mu'].shape[0]
        obs = simulator_fn(seed, arg)['obs']
        obs = obs.reshape((n_samples, -1))
        return obs # type:ignore

    sbijax_estim = FMPE(
        (sbijax_prior_fn, sbijax_simulator_fn),
        sbijax_nn,
    )

    data, params = None, {}
    for _ in range(n_rounds):
        sample_key, key = jr.split(key)
        data, params = sbijax_estim.simulate_data_and_possibly_append(
            sample_key,
            params=params,
            observable=sbijax_y_observed,
            data=data,
            n_simulations=n_simulations // n_theta,
        )
        train_key, key = jr.split(key)
        params, _ = sbijax_estim.fit(
            train_key,
            data=data,
            n_iter=n_epochs
        )

    post_key, key = jr.split(key)
    sbijax_posterior, _ = sbijax_estim.sample_posterior(
        rng_key=post_key,
        params=params,
        observable=sbijax_y_observed,
        n_samples=n_post_samples,
    )
    sbijax_theta_hat_array = jnp.array(sbijax_posterior.posterior.theta).reshape( # type: ignore
        (n_post_samples, (1 + n_theta))
    )
    sbijax_theta_hat = jnp.mean(
            sbijax_theta_hat_array[..., :1],
        keepdims=True,
        axis=0
    )
    sbijax_z_hat = jnp.mean(
        sbijax_theta_hat_array[..., 1:],
        keepdims=True,
        axis=0
    )
    theta_hat = jnp.mean(posterior['mu'], keepdims=True, axis=0)
    z_hat = jnp.mean(posterior['theta'], keepdims=True, axis=0)
    metrics = {
        'lc2stnf': {
            'sfmpe': lc2stnf(estim, prior_fn, simulator_fn),
            'sbijax': lc2stnf(sbijax_estim, sbijax_prior_fn, sbijax_simulator_fn)
        }
    }
    print(theta_truth)
    print(theta_hat)
    print(z_hat)
    print(sbijax_theta_hat)
    print(sbijax_z_hat)
    assert sbijax_theta_hat[None,...] == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert theta_hat == pytest.approx(theta_truth['theta'], tol) # type: ignore
    assert sbijax_z_hat[None,...] == pytest.approx(theta_truth['z'], tol) # type: ignore
    assert z_hat == pytest.approx(theta_truth['z'], tol) # type: ignore

if __name__ == "__main__":
    run()
