import pytest
from jax import numpy as jnp, random as jr, vmap

from sfmpe.util.dataloader import flatten_structured
from sfmpe.utils import split_data
import flax.nnx as nnx
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.cnf import CNF
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.metrics.lc2st import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    train_lc2st_classifiers,
    evaluate_lc2st
)
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.mlp import MLPVectorField
import optax

n_epochs_train = 1_000


def create_transformer(rngs, config, dim):
    estimator = Transformer(
        config,
        value_dim=dim,
        n_labels=2,
        index_dim=0,
        rngs=rngs
    )
    optimiser = optax.adam(3e-4)

    return estimator, optimiser

@pytest.mark.parametrize(
    "dim,train_size,num_classifiers,builder",
    [
        (1, 1_000, 10, create_transformer),
    ]
)
def test_lc2st_on_learned_distribution_sfmpe(dim, train_size, num_classifiers, builder):
    key = jr.PRNGKey(0)
    nnx_key, key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_epochs = 100
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

    nn, optimiser = builder(rngs, config, dim)

    model = StructuredCNF(nn, rngs=rngs)

    estim = SFMPE(model, rngs=rngs)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.normal(sample_key, (train_size, dim))
    n_obs = 10

    sigma = 1e-1
    noise = jr.normal(sample_key, (train_size, dim * n_obs)) * sigma
    y = jnp.tile(theta, (1, n_obs)) + noise
    data = {
        'theta': { 'theta': theta[..., None] },
        'y': { 'y': y[..., None] },
    }

    data, slices = flatten_structured(data)
    split_key, key = jr.split(key)
    train, val = split_data(data['data'], train_size, split=.8, shuffle_rng=split_key)
    train = { 'data': train, 'labels': data['labels'] }
    val = { 'data': val, 'labels': data['labels'] }
    labels = data['labels']
    masks = None

    train_key, key = jr.split(key)

    losses = estim.fit(
        train,
        val,
        optimizer=optimiser,
        n_iter=n_epochs_train
    )
    print(f'Training losses: {losses[0][-1]}')
    print(f'Validation losses: {losses[1][-1]}')

    # Create calibration data (x, theta, theta_q)
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (train_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + jr.normal(key_x, (train_size, dim * n_obs)) * sigma
    
    # Generate posterior samples theta_q using the trained estimator (batched with vmap)
    def sample_single_posterior(x):
        return estim.sample_posterior(
            x[None, ..., None],
            labels,
            slices['theta'],
            n_samples=1  
        )['theta'].reshape((dim,))
    
    theta_q = vmap(sample_single_posterior)(x_cal)

    # Create calibration dataset (x, theta, theta_q)
    d_cal = (x_cal, theta_cal, theta_q)

    # Initialize classifiers for (x, theta) concatenated input
    n_layers = 1
    latent_dim = 16
    main = BinaryMLPClassifier(
        dim=dim * n_obs + dim,  # x_dim + theta_dim
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=nnx.relu,
        rngs=rngs,
    )

    null_classifier = MultiBinaryMLPClassifier(
        dim=dim * n_obs + dim,  # x_dim + theta_dim
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=nnx.relu,
        n=num_classifiers,
        rngs=rngs,
    )

    # Train both classifiers together
    train_key, key = jr.split(key)
    train_lc2st_classifiers(
        train_key,
        d_cal,
        main,
        null_classifier,
        n_epochs
    )

    # Evaluate Local-Classifier 2 Sample Test using posterior samples
    ev_key, key = jr.split(key)
    theta_truth = jr.normal(ev_key, (dim,))
    observation = jnp.tile(theta_truth, (n_obs,)) + jr.normal(ev_key, (dim * n_obs,)) * sigma
    
    # Sample from posterior for a specific observation
    posterior_samples = estim.sample_posterior(
        observation[None, ..., None],
        labels,
        slices['theta'],
        n_samples=100  
    )['theta'].reshape((100, dim))
    
    null_stats, main_stat, p_value = evaluate_lc2st(
        observation,
        posterior_samples,
        main,
        null_classifier,
    )
    print(f'null_stats: {null_stats}')
    print(f'main_stat: {main_stat}')
    print(f'p_value: {p_value}')
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05

@pytest.mark.parametrize(
    "dim,train_size,cal_size,num_classifiers",
    [
        (1, 1_000, 1_000, 10),
        (10, 1_000, 1_000, 10),
    ],
)
def test_lc2st_on_learned_distribution_fmpe(dim, train_size, cal_size, num_classifiers):
    key = jr.PRNGKey(42)
    nnx_key, key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_epochs = 100
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

    batch_size = train_size // 10
    train, val = split_data(
        data,
        train_size,
        .8,
        shuffle_rng=key
    )

    losses = estim.fit(
        train,
        val,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        batch_size=batch_size
    )

    print(f'Training losses: {losses[0][-1]}')
    print(f'Validation losses: {losses[1][-1]}')

    # Create calibration data (x, theta, theta_q)
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (cal_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + \
            jr.normal(key_x, (cal_size, dim * n_obs)) * sigma
    
    # Generate posterior samples theta_q using the trained estimator (batched with vmap)
    def sample_single_posterior(x):
        return estim.sample_posterior(
            x[None, ...],
            theta_shape = (dim,),
            n_samples=1
        ).reshape((dim,))
    
    theta_q = vmap(sample_single_posterior)(x_cal)

    # Create calibration dataset (x, theta, theta_q)
    d_cal = (x_cal, theta_cal, theta_q)

    # Initialize classifiers for (x, theta) concatenated input
    n_layers = 1
    latent_dim = 16
    main = BinaryMLPClassifier(
        dim=dim * n_obs + dim,  # x_dim + theta_dim
        latent_dim = latent_dim,
        n_layers=n_layers,
        activation=nnx.relu,
        rngs=rngs,
    )

    null_classifier = MultiBinaryMLPClassifier(
        dim=dim * n_obs + dim,  # x_dim + theta_dim
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=nnx.relu,
        n=num_classifiers,
        rngs=rngs,
    )

    # Train both classifiers together
    train_key, key = jr.split(key)
    train_lc2st_classifiers(
        train_key,
        d_cal,
        main,
        null_classifier,
        n_epochs
    )

    # Evaluate Local-Classifier 2 Sample Test using posterior samples
    ev_key, ev_noise_key, key = jr.split(key, 3)
    theta_truth = jr.normal(ev_key, (dim,))
    obs_noise = jr.normal(ev_noise_key, (dim * n_obs,)) * sigma
    observation = jnp.tile(theta_truth, (n_obs,)) + obs_noise
    
    # Sample from posterior  
    n_post = 1_000
    posterior_samples = estim.sample_posterior(
        observation[None,...],
        theta_shape=(dim,),
        n_samples=n_post
    ).reshape((n_post, dim))
    print(f'truth: {theta_truth}')
    print(f'observation: {observation}')
    print(f'posterior mean: {jnp.mean(posterior_samples, axis=0)}')
    print(f'posterior std: {jnp.std(posterior_samples, axis=0)}')
    
    null_stats, main_stat, p_value = evaluate_lc2st(
        observation,
        posterior_samples,
        main,
        null_classifier,
    )
    print(f'null_stats: {null_stats}')
    print(f'main_stat: {main_stat}')
    print(f'p_value: {p_value}')
    print(jnp.quantile(null_stats, jnp.array([0.25, .5, 0.95])))
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05

@pytest.mark.parametrize(
    "dim,train_size,cal_size,num_classifiers",
    [
        (1, 10_000, 1_000, 10),
    ],
)
def test_lc2st_on_learned_hierarchical_distribution_fmpe(dim, train_size, cal_size, num_classifiers):
    key = jr.PRNGKey(0)
    nnx_key, key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_epochs = 100
    n_phi = 2
    n_obs = 5 

    nn = MLPVectorField(
        theta_dim=dim + dim * n_phi,
        context_dim=dim * n_phi * n_obs,
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
    mu = jr.normal(sample_key, (train_size, dim))
    phi_sd = 1e-1
    phi_noise = jr.normal(sample_key, (train_size, dim * n_phi)) * phi_sd
    phi = mu.repeat(n_phi, axis=1) + phi_noise

    y_sd = 1e-2
    noise = jr.normal(sample_key, (train_size, dim * n_phi * n_obs)) * y_sd
    y = jnp.tile(phi, (1, n_obs)) + noise
    data = {
        'data': {
            'theta': jnp.concatenate([mu, phi], axis=1),
            'y': y
        }
    }

    batch_size = train_size // 10
    train, val = split_data(
        data,
        train_size,
        .8,
        shuffle_rng=key
    )

    losses = estim.fit(
        train,
        val,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        batch_size=batch_size
    )

    print(f'Training losses: {losses[0][-1]}')
    print(f'Validation losses: {losses[1][-1]}')

    # Create calibration data (x, theta, theta_q)
    key_theta, key_x, key = jr.split(key, 3)
    mu_cal = jr.normal(key_theta, (cal_size, dim))
    phi_cal = mu_cal.repeat(n_phi, axis=1) + \
              jr.normal(key_x, (cal_size, dim * n_phi)) * phi_sd
    x_cal = jnp.tile(phi_cal, (1, n_obs)) + \
            jr.normal(key_x, (cal_size, dim * n_phi * n_obs)) * y_sd
    theta_cal = jnp.concatenate([mu_cal, phi_cal], axis=1)
    
    # Generate posterior samples theta_q using the trained estimator (batched with vmap)
    def sample_single_posterior(x, key):
        theta_0 = jr.normal(key, (dim + dim * n_phi,))
        return estim.sample_posterior(
            x[None, ...],
            theta_shape = (dim + dim * n_phi,),
            n_samples=1,
            theta_0 = theta_0
        ).reshape((dim + dim * n_phi,))
    
    keys = jr.split(key, cal_size)
    theta_q = vmap(sample_single_posterior)(x_cal, keys)

    # Create calibration dataset (x, theta, theta_q)
    d_cal = (x_cal, theta_cal, theta_q)

    # Initialize classifiers for (x, theta) concatenated input
    n_layers = 1
    latent_dim = 16
    main = BinaryMLPClassifier(
        dim=dim * n_phi * n_obs + dim + dim * n_phi,  # x_dim + theta_dim
        latent_dim = latent_dim,
        n_layers=n_layers,
        activation=nnx.relu,
        rngs=rngs,
    )

    null_classifier = MultiBinaryMLPClassifier(
        dim=dim * n_phi * n_obs + dim + dim * n_phi,  # x_dim + theta_dim
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=nnx.relu,
        n=num_classifiers,
        rngs=rngs,
    )

    # Train both classifiers together
    train_key, key = jr.split(key)
    train_lc2st_classifiers(
        train_key,
        d_cal,
        main,
        null_classifier,
        n_epochs
    )

    # Evaluate Local-Classifier 2 Sample Test using posterior samples
    mu_key, phi_noise_key, obs_noise_key, key = jr.split(key, 4)
    mu_truth = jr.normal(mu_key, (dim,))
    phi_truth = mu_truth.repeat(n_phi) + \
               jr.normal(phi_noise_key, (dim * n_phi,)) * phi_sd
    observation = phi_truth.repeat(n_obs) + \
            jr.random.normal(obs_noise_key, (dim * n_phi * n_obs,)) * y_sd
    theta_truth = jnp.concatenate([mu_truth, phi_truth], axis=0)
    
    # Sample from posterior  
    posterior_samples = estim.sample_posterior(
        observation[None,...],
        theta_shape=(dim + dim * n_phi,),
        n_samples=100
    ).reshape((100, dim + dim * n_phi))
    print(f'truth: {theta_truth}')
    print(f'observation: {observation}')
    print(f'posterior mean: {jnp.mean(posterior_samples, axis=0)}')
    print(f'posterior std: {jnp.std(posterior_samples, axis=0)}')
    print(f'posterior: {posterior_samples[:10]}')
    
    ev_key, key = jr.split(key)
    null_stats, main_stat, p_value = evaluate_lc2st(
        observation,
        posterior_samples,
        main,
        null_classifier,
    )
    print(f'null_stats: {null_stats}')
    print(f'main_stat: {main_stat}')
    print(f'p_value: {p_value}')
    print(jnp.quantile(null_stats, jnp.array([0.25, .5, 0.95])))
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05
