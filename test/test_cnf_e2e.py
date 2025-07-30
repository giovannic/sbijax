import pytest
from jax import (
    numpy as jnp,
    random as jr,
    vmap,
    scipy as js,
    tree
)

import flax.nnx as nnx
from sfmpe.util.dataloader import flatten_structured
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.cnf import CNF
from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.nn.mlp import MLPVectorField
import optax

n_epochs_train = 100

def _split_data(data, size, split=.8):
    n_train = int(size * split)

    train = tree.map(lambda x: x[:n_train], data)
    val = tree.map(lambda x: x[n_train:], data)

    return train, val

def ks_norm_test(data, alpha=0.05):
    """
    Kolmogorov-Smirnov test for normality using cumulative statistics.
    Tests if data comes from a standard normal distribution.
    
    Args:
        data: 1D array of data points
        alpha: significance level (default 0.05)
    
    Returns:
        dict with test statistic, critical value, and p_value estimate
    """
    n = len(data)
    
    # Sort the data
    sorted_data = jnp.sort(data)
    
    # Compute empirical CDF
    empirical_cdf = jnp.arange(1, n + 1) / n
    
    # Compute theoretical CDF (standard normal)
    theoretical_cdf = js.stats.norm.cdf(sorted_data)
    
    # KS test statistic: maximum difference between CDFs
    D = jnp.max(jnp.abs(empirical_cdf - theoretical_cdf))
    
    # Critical value for KS test (asymptotic approximation)
    critical_value = jnp.sqrt(-0.5 * jnp.log(alpha / 2)) / jnp.sqrt(n)
    
    # Rough p-value estimate (Kolmogorov distribution approximation)
    p_value_approx = 2 * jnp.exp(-2 * n * D**2)
    
    return {
        'statistic': D,
        'critical_value': critical_value,
        'p_value_approx': p_value_approx,
        'reject_null': D > critical_value,
        'sample_size': n
    }

def _create_transformer(rngs, config, dim):
    estimator = Transformer(
        config,
        value_dim=dim,
        n_labels=2,
        index_dim=0,
        rngs=rngs
    )
    return estimator

@pytest.mark.parametrize(
    "dim,train_size,builder",
    [
        (1, 10_000, _create_transformer),
        (10, 10_000, _create_transformer),
    ],
)
def test_scnf_can_recover_base_distribution_from_training_set(
    dim,
    train_size,
    builder
    ):
    n_obs = 10
    key = jr.PRNGKey(0)
    nnx_key, key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    config = {
        'latent_dim': 64,
        'label_dim': 1,
        'index_out_dim': 0,
        'n_encoder': 1,
        'n_decoder': 1,
        'n_heads': 1,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }

    nn = builder(rngs, config, dim)

    model = StructuredCNF(nn)

    estim = SFMPE(model)
    optimiser = optax.adam(3e-4)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.normal(sample_key, (train_size, dim))

    sigma = 1e-1
    noise = jr.normal(sample_key, (train_size, dim * n_obs)) * sigma
    y = jnp.tile(theta, (1, n_obs)) + noise
    data = {
        'theta': { 'theta': theta[..., None] },
        'y': { 'y': y[..., None] },
    }
    data, slices = flatten_structured(data)
    train, val = _split_data(data['data'], train_size, .8)
    train = { 'data': train, 'labels': data['labels'] }
    val = { 'data': val, 'labels': data['labels'] }
    labels = data['labels']
    masks = None

    train_key, key = jr.split(key)
    estim.fit(
        train_key,
        train,
        val,
        optimizer=optimiser,
        n_iter=n_epochs_train
    )

    inv_key, key = jr.split(key)
    z = estim.sample_base_dist(
        inv_key,
        train['data']['theta'],
        train['data']['y'],
        labels,
        slices['theta'],
        masks=masks
    )
    z = z['theta'].reshape((train['data']['theta'].shape[0], -1))

    # check each dimension is normal
    print(f'mean: {jnp.mean(z)}, std: {jnp.std(z)}')
    ks_response = vmap(
        lambda x: ks_norm_test(x, alpha=0.01),
        in_axes=(1,)
    )(z)
    print(ks_response)
    assert jnp.all(~ks_response['reject_null'])

@pytest.mark.parametrize(
    "dim,train_size,builder",
    [
        (1, 10_000, _create_transformer),
        (10, 10_000, _create_transformer),
    ],
)
def test_scnf_can_recover_base_distribution_from_posterior_estimate(
    dim,
    train_size,
    builder
    ):
    key = jr.PRNGKey(0)
    key, nnx_key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_obs = 10
    config = {
        'latent_dim': 64,
        'label_dim': 2,
        'index_out_dim': 0,
        'n_encoder': 1,
        'n_decoder': 1,
        'n_heads': 1,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }

    nn = builder(rngs, config, dim)

    optimiser = optax.adam(3e-4)

    model = StructuredCNF(nn)

    estim = SFMPE(model)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.normal(sample_key, (train_size, dim))

    sigma = 1e-1
    noise = jr.normal(sample_key, (train_size, dim * n_obs)) * sigma
    y = jnp.tile(theta, (1, n_obs)) + noise
    data = {
        'theta': { 'theta': theta[..., None] },
        'y': { 'y': y[..., None] },
    }
    data, slices = flatten_structured(data)
    train, val = _split_data(data['data'], train_size, .8)
    train = { 'data': train, 'labels': data['labels'] }
    val = { 'data': val, 'labels': data['labels'] }
    masks = None

    train_key, key = jr.split(key)

    estim.fit(
        train_key,
        train,
        val,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        batch_size = train_size // 10
    )

    # check that posterior estimations for some contexts are close
    test_y = train['data']['y'][0]
    n_post = 1_000
    theta_hat = estim.sample_posterior(
        key,
        test_y[None, ...],
        data['labels'],
        slices['theta'],
        n_samples=n_post
    )
    theta_truth = theta[0]
    print(f'theta_truth: {theta_truth}')
    print(f'mean theta_hat: {jnp.mean(theta_hat["theta"])}')
    # assert jnp.allclose(
        # jnp.mean(theta_hat['theta']),
        # theta_truth,
        # atol=1e-3
    # )
    # TODO: compute true posterior https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/gaussian_linear/task.py

    # check that reverse flow recovers the base distribution
    inv_key, key = jr.split(key)
    z = estim.sample_base_dist(
        inv_key,
        theta_hat['theta'],
        jnp.broadcast_to(
            test_y[None, ...],
            (n_post, n_obs * dim, 1)
        ),
        data['labels'],
        slices['theta'],
        masks=masks
    )
    z = z['theta'].reshape((n_post, -1))

    # check each dimension is normal
    print(f'mean: {jnp.mean(z)}, std: {jnp.std(z)}')
    ks_response = vmap(lambda x: ks_norm_test(x, alpha=0.01), in_axes=(1,))(z)
    print(ks_response)
    assert jnp.all(~ks_response['reject_null'])

@pytest.mark.parametrize(
    "dim,train_size",
    [
        (1, 10_000),
        (10, 10_000),
    ],
)
def test_cnf_can_recover_base_distribution_from_training_set(dim, train_size):
    key = jr.PRNGKey(0)
    n_obs = 10

    nn = MLPVectorField(
        theta_dim=dim,
        context_dim=dim * n_obs,
        latent_dim=64,
        n_layers=2,
        dropout=.1,
        activation=nnx.relu
    )
    optimiser = optax.adam(3e-4)

    model = CNF(nn)

    estim = FMPE(model)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.normal(sample_key, (train_size, dim))

    sigma = 1e-1
    noise = jr.normal(sample_key, (train_size, dim * n_obs)) * sigma
    y = jnp.tile(theta, (1, n_obs)) + noise
    data = {
        'data': {
            'theta': theta,
            'y': y
        }
    }

    train, val = _split_data(data, train_size, .8)

    estim.fit(
        train,
        val,
        optimizer=optimiser,
        n_iter=n_epochs_train
    )

    z = estim.sample_base_dist(
        train['data']['theta'],
        train['data']['y'],
        (dim,)
    )

    # check each dimension is normal
    print(f'mean: {jnp.mean(z)}, std: {jnp.std(z)}')
    ks_response = vmap(
        lambda x: ks_norm_test(x, alpha=0.01),
        in_axes=(1,)
    )(z)
    print(ks_response)
    assert jnp.all(~ks_response['reject_null'])

@pytest.mark.parametrize(
    "dim,train_size",
    [
        (1, 10_000),
        (10, 10_000),
    ],
)
def test_cnf_can_recover_base_distribution_from_posterior_estimate(
    dim,
    train_size,
    ):
    key = jr.PRNGKey(0)
    key, nnx_key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_obs = 10

    nn = MLPVectorField(
        theta_dim=dim,
        context_dim=dim * n_obs,
        latent_dim=64,
        n_layers=2,
        dropout=.1,
        activation=nnx.relu,
        rngs=rngs
    )
    optimiser = optax.adam(3e-4)

    model = CNF(nn)

    estim = FMPE(model)

    optimiser = optax.adam(3e-4)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.normal(sample_key, (train_size, dim))

    sigma = 1e-1
    noise = jr.normal(sample_key, (train_size, dim * n_obs)) * sigma
    y = jnp.tile(theta, (1, n_obs)) + noise
    data = {
        'data': {
            'theta': theta,
            'y': y
        }
    }

    train, val = _split_data(data, train_size, .8)

    estim.fit(
        train,
        val,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        batch_size = train_size // 10
    )

    # check that posterior estimations for some contexts are close
    test_y = y[0]
    theta_hat = estim.sample_posterior(
        test_y[None, ...],
        theta_shape = (dim,),
        n_samples=1_000,
    )
    theta_truth = theta[0]
    print(f'theta_truth: {theta_truth}')
    print(f'mean: {jnp.mean(theta_hat)}, std: {jnp.std(theta_hat)}')
    # assert jnp.allclose(jnp.mean(theta_hat), theta_truth, atol=1e-3)
    # TODO: compute true posterior https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/gaussian_linear/task.py

    # check that reverse flow recovers the base distribution
    z = estim.sample_base_dist(
        theta_hat,
        jnp.broadcast_to(
            test_y[None, ...],
            (theta_hat.shape[0], n_obs * dim)
        ),
        (dim,)
    )

    # check each dimension is normal
    print(f'mean: {jnp.mean(z)}, std: {jnp.std(z)}')
    ks_response = vmap(lambda x: ks_norm_test(x, alpha=0.01), in_axes=(1,))(z)
    print(ks_response)
    assert jnp.all(~ks_response['reject_null'])
