import pytest
from jax import numpy as jnp, random as jr, vmap

from sfmpe.util.dataloader import (
    flatten_structured,
    flat_as_batch_iterators,
    as_batch_iterators
)
import flax.nnx as nnx
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.make_vanilla_cnf import VanillaCNF
from sfmpe.fmpe import SFMPE
from sfmpe.vanilla_fmpe import FMPE
from sfmpe.lc2stnf import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    train_l_c2st_nf_main_classifier,
    precompute_null_distribution_nf_classifiers,
    evaluate_l_c2st_nf
)
from sfmpe.nn.mlp import MLPVectorField
from sfmpe.nn.vanilla_mlp import VanillaMLPVectorField
import optax

n_epochs_train = 1_000

def create_transformer_schedule(
    peak_lr: float = 1e-4,
    warmup_steps: int = 4000,
    total_steps: int = 100000,
    end_lr_factor: float = 0.01
):
    """
    Creates a learning rate schedule with linear warmup followed by cosine decay.
    
    Args:
        peak_lr: Maximum learning rate reached after warmup
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps
        end_lr_factor: Final LR as fraction of peak_lr (e.g., 0.01 = 1% of peak)
    
    Returns:
        optax schedule function
    """
    
    warmup_schedule = optax.constant_schedule(peak_lr)
    
    # Cosine decay from peak_lr to end_lr
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=end_lr_factor  # End LR = alpha * init_value
    )
    
    # Combine schedules
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps]
    )
    
    return schedule


def create_transformer(rngs, config, dim):
    estimator = Transformer(
        config,
        value_dim=dim,
        n_labels=2,
        index_dim=0,
        rngs=rngs
    )
    optimiser = optax.adam(3e-4)
    # optimiser = optax.adam(
        # create_transformer_schedule(
            # peak_lr=3e-4,
            # warmup_steps=int(0.5 * n_epochs_train),
            # total_steps=n_epochs_train
        # )
    # )

    return estimator, optimiser

def create_mlp(rngs, config, dim):
    estimator = MLPVectorField(
        config,
        in_dim=2 * dim,
        out_dim=dim,
        value_dim=dim,
        n_labels=2,
        rngs=rngs
    )
    optimiser = optax.adam(3e-4)

    return estimator, optimiser

@pytest.mark.parametrize(
    "dim,batch_size,num_classifiers,builder",
    [
        (1, 1_000, 100, create_transformer),
        (1, 1_000, 100, create_mlp),
    ],
    ids=[
        "transformer",
        "mlp"
    ]
)
def test_lc2stnf_on_learned_distribution_sfmpe(dim, batch_size, num_classifiers, builder):
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

    model = CNF(transform=nn)

    estim = SFMPE(model)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.normal(sample_key, (batch_size, 1, dim))

    sigma = 1e-5
    noise = jr.normal(sample_key, theta.shape) * sigma
    y = theta + noise
    data = {
        'theta': { 'theta': theta },
        'y': { 'y': y },
    }
    train_data, slices = flatten_structured(data)
    labels = train_data['labels']
    masks = None #train_data['masks']

    train_key, itr_key, key = jr.split(key, 3)
    train_iter, val_iter = flat_as_batch_iterators(
        itr_key,
        train_data
    )
    estim.fit(
        train_key,
        train_iter,
        val_iter,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        n_early_stopping_patience=100,
    )

    def inverse(theta, y):
        theta, y = theta[..., None], y[..., None]
        def sample_pair(theta, y):
            return estim.sample_structured_posterior(
                key,
                y[None, ...],
                labels,
                slices['theta'],
                masks=masks,
                n_samples=1,
                theta_0=theta[None, ...],
                direction='backward'
            )

        z = vmap(sample_pair)(theta, y)
        return z['theta'].reshape((y.shape[0], -1))

    # Create calibration data
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (batch_size, dim))
    x_cal = theta_cal + jr.normal(key_x, theta_cal.shape) * sigma
    calibration_data = (theta_cal, x_cal)

    # Train main classifier
    n_layers = 1
    activation = nnx.relu
    main = BinaryMLPClassifier(
        dim=dim * 2,
        latent_dim=64,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs,
    )

    train_key, key = jr.split(key)
    train_l_c2st_nf_main_classifier(
        train_key,
        main,
        calibration_data,
        inverse,
        n_epochs
    )

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim * 2,
        latent_dim=64,
        n_layers=2,
        activation=nnx.relu,
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
    observation = jr.normal(ev_key, (dim,))
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
    key = jr.PRNGKey(0)
    nnx_key, key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_epochs = 100
    n_obs = 10

    nn = VanillaMLPVectorField(
        theta_dim=dim,
        context_dim=dim * n_obs,
        latent_dim=64,
        n_layers=2,
        dropout=.1,
        activation=nnx.relu
    )
    optimiser = optax.adam(3e-4)

    model = VanillaCNF(transform=nn)

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

    train_key, itr_key, key = jr.split(key, 3)
    train_iter, val_iter = as_batch_iterators(
        itr_key,
        data,
        batch_size=train_size // 10,
        split=.8,
        shuffle=True
    )
    estim.fit(
        train_key,
        train_iter,
        val_iter,
        n_iter=n_epochs_train,
        optimizer = optimiser,
        n_early_stopping_patience=100,
    )

    def inverse(theta, y):
        def sample_pair(theta, y):
            return estim.sample_posterior(
                key,
                y[None, ...],
                theta_shape = (dim,),
                n_samples=1,
                theta_0=theta[None, ...],
                direction='backward'
            )

        z = vmap(sample_pair)(theta, y)
        return z[..., 0]

    # Create calibration data
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (cal_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + \
            jr.normal(key_x, (cal_size, dim * n_obs)) * sigma
    calibration_data = (theta_cal, x_cal)

    # Train main classifier
    n_layers = 1
    activation = nnx.relu
    main = BinaryMLPClassifier(
        dim=dim + dim * n_obs,
        latent_dim=64,
        n_layers=n_layers,
        activation=activation,
        rngs=rngs,
    )

    train_key, key = jr.split(key)
    train_l_c2st_nf_main_classifier(
        train_key,
        main,
        calibration_data,
        inverse,
        n_epochs
    )

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim + dim * n_obs,
        latent_dim=64,
        n_layers=2,
        activation=nnx.relu,
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
        key,
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
    assert main_stat < jnp.quantile(null_stats, 0.99)
    assert p_value > 0.01
