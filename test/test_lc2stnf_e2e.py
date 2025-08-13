import pytest
from jax import numpy as jnp, random as jr

from sfmpe.util.dataloader import flatten_structured
from sfmpe.utils import split_data
import flax.nnx as nnx
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.cnf import CNF
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.metrics.lc2stnf import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    train_l_c2st_nf_main_classifier,
    precompute_null_distribution_nf_classifiers,
    evaluate_l_c2st_nf
)
from sfmpe.utils import split_data
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
        (1, 1_000, 100, create_transformer),
    ]
)
def test_lc2stnf_on_learned_distribution_sfmpe(dim, train_size, num_classifiers, builder):
    key = jr.PRNGKey(42)
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
    n_obs = 1

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

    estim.fit(
        train,
        val,
        optimizer=optimiser,
        n_iter=n_epochs_train
    )

    # Create calibration data
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (train_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + jr.normal(key_x, (train_size, dim * n_obs)) * sigma
    calibration_data = (theta_cal, x_cal)
    
    # Generate z_samples using the inverse function
    z_samples = estim.sample_base_dist(
        theta_cal[..., None],
        x_cal[..., None],
        labels,
        slices['theta'],
        masks=masks
    )['theta'].reshape((theta_cal.shape[0], -1))

    # Train main classifier
    n_layers = 1
    activation = nnx.relu
    latent_dim = 64
    main = BinaryMLPClassifier(
        dim=dim + dim * n_obs,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs,
    )

    train_key, key = jr.split(key)
    train_l_c2st_nf_main_classifier(
        train_key,
        main,
        calibration_data,
        z_samples,
        n_epochs
    )

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim + dim * n_obs,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        n=num_classifiers,
        rngs=rngs,
    )

    # Precompute null distribution classifiers
    precompute_null_distribution_nf_classifiers(
        rng_key=key,
        calibration_data=calibration_data,
        classifiers=null_classifier,
        num_epochs=100,
    )

    # Evaluate LC2ST-NF
    ev_key, key = jr.split(key)
    observation = jr.normal(ev_key, (dim * n_obs,))
    n_cal = 100
    null_stats, main_stat, p_value = evaluate_l_c2st_nf(
        ev_key,
        observation,
        main,
        null_classifier,
        latent_dim=dim,
        Nv=n_cal
    )
    print(null_stats, main_stat, p_value)
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05

@pytest.mark.parametrize(
    "dim,train_size,cal_size,num_classifiers",
    [
        (1, 10_000, 1_000, 100),
    ],
)
def test_lc2stnf_on_learned_distribution_fmpe(dim, train_size, cal_size, num_classifiers):
    key = jr.PRNGKey(42)
    nnx_key, key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_obs = 1

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

    split_key, key = jr.split(key)
    train, val = split_data(
        data,
        train_size,
        split=.8,
        shuffle_rng=split_key
    )

    batch_size = train_size // 10

    estim.fit(
        train,
        val,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        batch_size=batch_size
    )

    # Create calibration data
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (cal_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + \
            jr.normal(key_x, (cal_size, dim * n_obs)) * sigma
    calibration_data = (theta_cal, x_cal)
    
    # Generate z_samples using the inverse function
    z_samples = estim.sample_base_dist(
        theta_cal,
        x_cal,
        theta_shape = (dim,)
    )

    # Train main classifier
    n_layers = 1
    activation = nnx.relu
    latent_dim = 16
    main = BinaryMLPClassifier(
        dim=dim + dim * n_obs,
        latent_dim = latent_dim,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs,
    )

    train_key, key = jr.split(key)
    train_l_c2st_nf_main_classifier(
        train_key,
        main,
        calibration_data,
        z_samples,
        num_epochs=100
    )

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim + dim * n_obs,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        n=num_classifiers,
        rngs=rngs,
    )

    # Precompute null distribution classifiers
    precompute_null_distribution_nf_classifiers(
        rng_key=key,
        calibration_data=calibration_data,
        classifiers=null_classifier,
        num_epochs=100,
    )

    # Evaluate LC2ST-NF
    ev_key, ev_noise_key, key = jr.split(key, 3)
    theta_truth = jr.normal(ev_key, (dim,))
    obs_noise = jr.normal(ev_noise_key, (dim * n_obs,)) * sigma
    observation = jnp.tile(theta_truth, (n_obs,)) + obs_noise
    posterior = estim.sample_posterior(
        observation[None,...],
        theta_shape = (dim,),
        n_samples=100
    )
    print(f'truth: {theta_truth}')
    print(f'observation: {observation}')
    print(f'posterior mean: {jnp.mean(posterior)}')
    print(f'posterior std: {jnp.std(posterior)}')
    n_cal = 100
    null_stats, main_stat, p_value = evaluate_l_c2st_nf(
        ev_key,
        observation,
        main,
        null_classifier,
        latent_dim=dim,
        Nv=n_cal
    )
    print(null_stats, main_stat, p_value)
    print(jnp.quantile(null_stats, jnp.array([0.25, .5, 0.95])))
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05
