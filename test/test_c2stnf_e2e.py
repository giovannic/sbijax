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
from sfmpe.c2stnf import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    train_c2st_nf_main_classifier,
    precompute_c2st_nf_null_classifiers,
    evaluate_c2st_nf
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
def test_c2stnf_on_learned_distribution_sfmpe(dim, train_size, num_classifiers, builder):
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

    # Create calibration data and generate z_samples
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (train_size, dim))
    x_cal = jnp.tile(theta_cal, (1, n_obs)) + jr.normal(key_x, (train_size, dim * n_obs)) * sigma
    
    # Generate z_samples using the inverse function
    z_samples = estim.sample_base_dist(
        theta_cal[..., None],
        x_cal[..., None],
        labels,
        slices['theta'],
        masks=masks
    )['theta'].reshape((theta_cal.shape[0], -1))

    # Train main classifier (z-only, no x)
    n_layers = 1
    activation = nnx.relu
    latent_dim = 16
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

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim,  # Only z dimension (no x)
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        n=num_classifiers,
        rngs=rngs,
    )

    # Precompute null distribution classifiers
    null_key, key = jr.split(key)
    precompute_c2st_nf_null_classifiers(
        rng_key=null_key,
        classifiers=null_classifier,
        latent_dim=dim,
        N_cal=train_size,
        num_epochs=n_epochs,
    )

    # Evaluate C2ST-NF using posterior samples
    ev_key, key = jr.split(key)
    theta_truth = jr.normal(ev_key, (dim,))
    observation = jnp.tile(theta_truth, (n_obs,)) + jr.normal(ev_key, (dim * n_obs,)) * sigma
    
    # Sample from posterior for a specific observation
    posterior_samples = estim.sample_posterior(
        observation[None, ..., None],
        labels,
        slices['theta'],
        n_samples=100  
    )
    
    # Convert posterior samples to z samples using the base distribution sampling
    z_posterior_samples = estim.sample_base_dist(
        posterior_samples['theta'],
        jnp.broadcast_to(
            observation[None, ..., None],
            (100, dim * n_obs, 1)
        ),
        labels,
        slices['theta'],
        masks=masks
    )['theta'].reshape((100, -1))
    
    null_stats, main_stat, p_value = evaluate_c2st_nf(
        ev_key,
        z_posterior_samples,
        main,
        null_classifier,
        latent_dim=dim,
    )
    print(f'null_stats: {null_stats}')
    print(f'main_stat: {main_stat}')
    print(f'p_value: {p_value}')
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05

@pytest.mark.parametrize(
    "dim,train_size,cal_size,num_classifiers,builder",
    [
        (1, 1_000, 1_000, 10, create_transformer),
    ]
)
def test_c2stnf_on_learned_hierarchical_distribution_sfmpe(dim, train_size, cal_size, num_classifiers, builder):
    key = jr.PRNGKey(0)
    nnx_key, key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_epochs = 100
    n_phi = 2
    n_obs = 5
    
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

    # Create training data with hierarchical structure
    sample_key, key = jr.split(key)
    mu = jr.normal(sample_key, (train_size, 1, dim))
    phi_sd = 1e-1
    phi_noise = jr.normal(sample_key, (train_size, n_phi, dim)) * phi_sd
    phi = jnp.broadcast_to(mu, (train_size, n_phi, dim)) + phi_noise

    y_sd = 1e-2
    noise = jr.normal(sample_key, (train_size, n_phi, n_obs, dim)) * y_sd
    y = jnp.broadcast_to(phi[:, :, None, :], (train_size, n_phi, n_obs, dim)) + noise
    
    data = {
        'theta': { 
            'mu': mu[..., None], 
            'phi': phi[..., None]
        },
        'y': { 
            'y': y[..., None] 
        },
    }

    # Define independence structure
    independence = {
        'local': ['y'],  # y observations independent of each other
        'cross': [('mu', 'y')],  # mu completely independent of y  
        'cross_local': [('phi', 'y', (0, 0))]  # phi[i] connects to y[i]
    }

    data, slices = flatten_structured(data, independence=independence)
    split_key, key = jr.split(key)
    train, val = split_data(data['data'], train_size, split=.8, shuffle_rng=split_key)
    train = { 'data': train, 'labels': data['labels'] }
    val = { 'data': val, 'labels': data['labels'] }
    labels = data['labels']
    masks = data.get('masks', None)
    
    # Add masks to train/val data if they exist
    if masks is not None:
        train['masks'] = masks
        val['masks'] = masks

    train_key, key = jr.split(key)

    losses = estim.fit(
        train,
        val,
        optimizer=optimiser,
        n_iter=n_epochs_train
    )
    print(f'Training losses: {losses[0][-1]}')
    print(f'Validation losses: {losses[1][-1]}')

    # Create calibration data and generate z_samples
    key_theta, key_x, key = jr.split(key, 3)
    mu_cal = jr.normal(key_theta, (cal_size, 1, dim))
    phi_cal = jnp.broadcast_to(mu_cal, (cal_size, n_phi, dim)) + jr.normal(key_x, (cal_size, n_phi, dim)) * phi_sd
    x_cal = jnp.broadcast_to(phi_cal[:, :, None, :], (cal_size, n_phi, n_obs, dim)) + jr.normal(key_x, (cal_size, n_phi, n_obs, dim)) * y_sd
    
    # Generate z_samples using the inverse function
    z_samples_dict = estim.sample_base_dist(
        jnp.concatenate([mu_cal[..., None], phi_cal[..., None]], axis=1),
        x_cal[..., None],
        labels,
        slices['theta'],
        masks=masks
    )
    # Flatten z_samples for classifier
    z_samples = jnp.concatenate([
        z_samples_dict['mu'].reshape((cal_size, -1)),
        z_samples_dict['phi'].reshape((cal_size, -1))
    ], axis=1)

    # Train main classifier
    n_layers = 1
    activation = nnx.relu
    latent_dim = 16
    main = BinaryMLPClassifier(
        dim=dim * (1 + n_phi),  # mu: 1*dim + phi: n_phi*dim
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

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim * (1 + n_phi),
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        n=num_classifiers,
        rngs=rngs,
    )

    # Precompute null distribution classifiers
    null_key, key = jr.split(key)
    precompute_c2st_nf_null_classifiers(
        rng_key=null_key,
        classifiers=null_classifier,
        latent_dim=dim * (1 + n_phi),
        N_cal=cal_size,
        num_epochs=n_epochs,
    )

    # Evaluate C2ST-NF using posterior samples
    mu_key, phi_noise_key, obs_noise_key, key = jr.split(key, 4)
    mu_truth = jr.normal(mu_key, (1, dim))
    phi_truth = jnp.broadcast_to(mu_truth, (n_phi, dim)) + jr.normal(phi_noise_key, (n_phi, dim)) * phi_sd
    observation = jnp.broadcast_to(phi_truth[:, None, :], (n_phi, n_obs, dim)) + jr.normal(obs_noise_key, (n_phi, n_obs, dim)) * y_sd
    
    # Sample from posterior for a specific observation
    posterior_samples = estim.sample_posterior(
        observation[..., None],
        labels,
        slices['theta'],
        n_samples=100,
        masks=masks
    )
    
    # Convert posterior samples to z samples using the base distribution sampling
    z_posterior_samples_dict = estim.sample_base_dist(
        jnp.concatenate([
            posterior_samples['mu'],
            posterior_samples['phi']
        ], axis=1),
        jnp.broadcast_to(
            observation[None, ..., None],
            (100, n_phi, n_obs, 1)
        ),
        labels,
        slices['theta'],
        masks=masks
    )
    # Flatten z_posterior_samples for classifier
    z_posterior_samples = jnp.concatenate([
        z_posterior_samples_dict['mu'].reshape((100, -1)),
        z_posterior_samples_dict['phi'].reshape((100, -1))
    ], axis=1)
    
    null_stats, main_stat, p_value = evaluate_c2st_nf(
        key,
        z_posterior_samples,
        main,
        null_classifier,
        latent_dim=dim * (1 + n_phi),
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
        (3, 1_000, 1_000, 10),
    ],
)
def test_c2stnf_on_learned_distribution_fmpe(dim, train_size, cal_size, num_classifiers):
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

    print(f'correlation: {jnp.corrcoef(z_samples, rowvar=False)}')

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

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim,  # Only z dimension (no x)
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        n=num_classifiers,
        rngs=rngs,
    )

    # Precompute null distribution classifiers
    null_key, key = jr.split(key)
    precompute_c2st_nf_null_classifiers(
        rng_key=null_key,
        classifiers=null_classifier,
        latent_dim=dim,
        N_cal=cal_size,
        num_epochs=n_epochs,
    )

    # Evaluate C2ST-NF using posterior samples
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
    )
    print(f'truth: {theta_truth}')
    print(f'observation: {observation}')
    print(f'posterior mean: {jnp.mean(posterior_samples)}')
    print(f'posterior std: {jnp.std(posterior_samples)}')
    
    # Convert posterior samples to z samples
    z_posterior_samples = estim.sample_base_dist(
        posterior_samples,
        observation[None, ...].repeat(n_post, axis=0),
        theta_shape=(dim,)
    )
    
    null_stats, main_stat, p_value = evaluate_c2st_nf(
        ev_key,
        z_posterior_samples,
        main,
        null_classifier,
        latent_dim=dim,
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
def test_c2stnf_on_learned_hierarchical_distribution_fmpe(dim, train_size, cal_size, num_classifiers):
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
    theta = jnp.concatenate([mu, phi], axis=1)

    y_sd = 1e-2
    noise = jr.normal(sample_key, (train_size, dim * n_phi * n_obs)) * y_sd
    y = jnp.tile(phi, (1, n_obs)) + noise
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

    # Create calibration data and generate z_samples
    key_theta, key_x, key = jr.split(key, 3)
    mu_cal = jr.normal(key_theta, (cal_size, dim))
    phi_cal = mu_cal.repeat(n_phi, axis=1) + \
              jr.normal(key_x, (cal_size, dim * n_phi)) * phi_sd
    x_cal = jnp.tile(phi_cal, (1, n_obs)) + \
            jr.normal(key_x, (cal_size, dim * n_phi * n_obs)) * y_sd
    theta_cal = jnp.concatenate([mu_cal, phi_cal], axis=1)
    
    # Generate z_samples using the inverse function
    z_samples = estim.sample_base_dist(
        theta_cal,
        x_cal,
        theta_shape = (dim + dim * n_phi,)
    )

    print(theta[:10])
    print(y[:10])

    print(f'correlation: {jnp.corrcoef(z_samples, rowvar=False)}')

    # Train main classifier
    n_layers = 1
    activation = nnx.relu
    latent_dim = 16
    main = BinaryMLPClassifier(
        dim=dim + dim * n_phi,
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

    # Initialize null classifiers
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim + dim * n_phi,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        n=num_classifiers,
        rngs=rngs,
    )

    # Precompute null distribution classifiers
    null_key, key = jr.split(key)
    precompute_c2st_nf_null_classifiers(
        rng_key=null_key,
        classifiers=null_classifier,
        latent_dim=dim + dim * n_phi,
        N_cal=cal_size,
        num_epochs=n_epochs,
    )

    # Evaluate C2ST-NF using posterior samples
    mu_key, phi_noise_key, obs_noise_key, key = jr.split(key, 4)
    mu_truth = jr.normal(mu_key, (dim,))
    phi_truth = mu_truth.repeat(n_phi) + \
               jr.normal(phi_noise_key, (dim * n_phi,)) * phi_sd
    observation = phi_truth.repeat(n_obs) + \
            jr.normal(obs_noise_key, (dim * n_phi * n_obs,)) * y_sd
    theta_truth = jnp.concatenate([mu_truth, phi_truth], axis=0)
    
    # Sample from posterior  
    n_post = 1_000
    posterior_samples = estim.sample_posterior(
        observation[None,...],
        theta_shape=(dim + dim * n_phi,),
        n_samples=n_post
    )
    print(f'truth: {theta_truth}')
    print(f'observation: {observation}')
    print(f'posterior mean: {jnp.mean(posterior_samples, axis=0)}')
    print(f'posterior std: {jnp.std(posterior_samples, axis=0)}')
    
    # Convert posterior samples to z samples
    z_posterior_samples = estim.sample_base_dist(
        posterior_samples,
        observation[None, ...].repeat(n_post, axis=0),
        theta_shape=(dim + dim * n_phi,)
    )
    print('z_post_mean', jnp.mean(z_posterior_samples, axis=0))
    print('z_post_std', jnp.std(z_posterior_samples, axis=0))
    print('z_post_correlation', jnp.corrcoef(z_posterior_samples, rowvar=False))
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    m = dim + dim * n_phi
    z_samples_df = pd.DataFrame(z_posterior_samples, columns=[f'z_{i}' for i in range(m)])
    g2 = sns.PairGrid(z_samples_df, diag_sharey=False)
    g2.map_lower(sns.scatterplot, alpha=0.6)
    g2.map_diag(sns.histplot, bins=30)
    g2.fig.suptitle('z_samples distribution (Estimated Posterior)', y=1.02)
    plt.savefig('z_post_sampled.png', bbox_inches='tight', dpi=150)
    plt.close()

    z_base = jr.normal(key, shape=(n_post, dim + dim * n_phi))
    print('z_base_mean', jnp.mean(z_base, axis=0))
    print('z_base_std', jnp.std(z_base, axis=0))
    print('z_base_correlation', jnp.corrcoef(z_base, rowvar=False))
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    m = dim + dim * n_phi
    z_samples_df = pd.DataFrame(z_base, columns=[f'z_{i}' for i in range(m)])
    g2 = sns.PairGrid(z_samples_df, diag_sharey=False)
    g2.map_lower(sns.scatterplot, alpha=0.6)
    g2.map_diag(sns.histplot, bins=30)
    g2.fig.suptitle('z_samples distribution (Estimated Posterior)', y=1.02)
    plt.savefig('z_base_sampled.png', bbox_inches='tight', dpi=150)
    plt.close()

    ev_key, key = jr.split(key)
    null_stats, main_stat, p_value = evaluate_c2st_nf(
        ev_key,
        z_posterior_samples,
        main,
        null_classifier,
        latent_dim=dim + dim * n_phi,
    )
    print(f'null_stats: {null_stats}')
    print(f'main_stat: {main_stat}')
    print(f'p_value: {p_value}')
    print(jnp.quantile(null_stats, jnp.array([0.25, .5, 0.95])))
    assert main_stat < jnp.quantile(null_stats, 0.95)
    assert p_value > 0.05
