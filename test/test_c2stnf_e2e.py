import pytest
from jax import numpy as jnp, random as jr, tree

from sfmpe.util.dataloader import flatten_structured
import flax.nnx as nnx
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.cnf import CNF
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.c2stnf import (
    BinaryMLPClassifier,
    train_c2st_nf_main_classifier,
    evaluate_c2st_nf
)
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.mlp import MLPVectorField
import optax

n_epochs_train = 1_000

def _split_data(data, size, split=.8):
    n_train = int(size * split)

    train = tree.map(lambda x: x[:n_train], data)
    val = tree.map(lambda x: x[n_train:], data)

    return train, val

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
    "dim,train_size,N_null,builder",
    [
        (1, 1_000, 10, create_transformer),
    ]
)
def test_c2stnf_on_learned_distribution_sfmpe(dim, train_size, N_null, builder):
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

    model = StructuredCNF(nn)

    estim = SFMPE(model)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.normal(sample_key, (train_size, dim))
    n_obs = 1

    sigma = 1e-1
    noise = jr.normal(sample_key, (train_size, dim * n_obs)) * sigma
    y = jnp.tile(theta, (1, n_obs)) + noise
    data = {
        'theta': { 'theta': theta[..., None] },
        'y': { 'y': y[..., None] },
    }

    data, slices = flatten_structured(data)
    train, val = _split_data(data['data'], train_size, split=.8)
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

    # Create calibration data and generate z_samples
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (train_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + jr.normal(key_x, (train_size, dim * n_obs)) * sigma
    
    # Generate z_samples using the inverse function
    z_samples = estim.sample_base_dist(
        key,
        theta_cal[..., None],
        x_cal[..., None],
        labels,
        slices['theta'],
        masks=masks
    )['theta'].reshape((theta_cal.shape[0], -1))

    # Train main classifier (z-only, no x)
    n_layers = 1
    activation = nnx.relu
    latent_dim = 64
    main = BinaryMLPClassifier(
        dim=dim,  # Only z dimension (no x)
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs,
    )

    train_key, key = jr.split(key)
    train_c2st_nf_main_classifier(
        train_key,
        main,
        z_samples,
        n_epochs
    )

    # Evaluate C2ST-NF using posterior samples
    ev_key, key = jr.split(key)
    theta_truth = jr.normal(ev_key, (dim,))
    observation = jnp.tile(theta_truth, (n_obs,)) + jr.normal(ev_key, (dim * n_obs,)) * sigma
    
    # Sample from posterior for a specific observation
    posterior_samples = estim.sample_posterior(
        key,
        observation[None, ..., None],
        labels,
        slices['theta'],
        n_samples=100  
    )
    
    # Convert posterior samples to z samples using the base distribution sampling
    z_posterior_samples = estim.sample_base_dist(
        key,
        posterior_samples['theta'],
        jnp.broadcast_to(
            observation[None, ..., None],
            (100, dim * n_obs, 1)
        ),
        labels,
        slices['theta'],
        masks=masks
    )['theta'].reshape((100, -1))
    
    n_val = 100
    null_stats, main_stat, p_value = evaluate_c2st_nf(
        ev_key,
        z_posterior_samples,
        main,
        latent_dim=dim,
        Nv=n_val,
        N_null=N_null
    )
    print(f'null_stats: {null_stats}')
    print(f'main_stat: {main_stat}')
    print(f'p_value: {p_value}')
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05

@pytest.mark.parametrize(
    "dim,train_size,cal_size,N_null",
    [
        (1, 10_000, 1_000, 10),
    ],
)
def test_c2stnf_on_learned_distribution_fmpe(dim, train_size, cal_size, N_null):
    key = jr.PRNGKey(0)
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
    n_train = int(train_size * 0.8)

    train = tree.map(lambda x: x[:n_train], data)
    val = tree.map(lambda x: x[n_train:], data)

    estim.fit(
        train,
        val,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        batch_size=batch_size
    )

    # Create calibration data and generate z_samples
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (cal_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + \
            jr.normal(key_x, (cal_size, dim * n_obs)) * sigma
    
    # Generate z_samples using the inverse function
    z_samples = estim.sample_base_dist(
        theta_cal,
        x_cal,
        theta_shape = (dim,)
    )

    # Train main classifier (z-only, no x)
    n_layers = 1
    activation = nnx.relu
    latent_dim = 16
    main = BinaryMLPClassifier(
        dim=dim,  # Only z dimension (no x)
        latent_dim = latent_dim,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs,
    )

    train_key, key = jr.split(key)
    train_c2st_nf_main_classifier(
        train_key,
        main,
        z_samples,
        n_epochs
    )

    # Evaluate C2ST-NF using posterior samples
    ev_key, ev_noise_key, key = jr.split(key, 3)
    theta_truth = jr.normal(ev_key, (dim,))
    obs_noise = jr.normal(ev_noise_key, (dim * n_obs,)) * sigma
    observation = jnp.tile(theta_truth, (n_obs,)) + obs_noise
    
    # Sample from posterior  
    posterior_samples = estim.sample_posterior(
        observation[None,...],
        theta_shape=(dim,),
        n_samples=100
    )
    print(f'truth: {theta_truth}')
    print(f'observation: {observation}')
    print(f'posterior mean: {jnp.mean(posterior_samples)}')
    print(f'posterior std: {jnp.std(posterior_samples)}')
    
    # Convert posterior samples to z samples
    z_posterior_samples = estim.sample_base_dist(
        posterior_samples,
        observation[None, ...].repeat(100, axis=0),
        theta_shape=(dim,)
    )
    
    n_val = 100
    null_stats, main_stat, p_value = evaluate_c2st_nf(
        ev_key,
        z_posterior_samples,
        main,
        latent_dim=dim,
        Nv=n_val,
        N_null=N_null
    )
    print(f'null_stats: {null_stats}')
    print(f'main_stat: {main_stat}')
    print(f'p_value: {p_value}')
    print(jnp.quantile(null_stats, jnp.array([0.25, .5, 0.95])))
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05