import pytest
from jax import (
    numpy as jnp,
    random as jr,
    vmap
)
import flax.nnx as nnx
import optax
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.util.dataloader import flatten_structured
from sfmpe.utils import split_data
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.cnf import CNF
from sfmpe.sfmpe import SFMPE
from sfmpe.fmpe import FMPE
from sfmpe.nn.mlp import MLPVectorField
from utils.helpers import ks_norm_test
from sfmpe.metrics.mmd import mmd_test

n_epochs_train = 100

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

    model = StructuredCNF(nn, rngs=rngs)

    estim = SFMPE(model, rngs=rngs)
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
    split_key, key = jr.split(key)
    train, val = split_data(data['data'], train_size, .8, shuffle_rng=split_key)
    train = { 'data': train, 'labels': data['labels'] }
    val = { 'data': val, 'labels': data['labels'] }
    labels = data['labels']
    masks = None

    estim.fit(
        train,
        val,
        optimizer=optimiser,
        n_iter=n_epochs_train
    )

    z = estim.sample_base_dist(
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

    model = StructuredCNF(nn, rngs=rngs)

    estim = SFMPE(model, rngs=rngs)

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
    split_key, key = jr.split(key)
    train, val = split_data(data['data'], train_size, .8, shuffle_rng=split_key)
    train = { 'data': train, 'labels': data['labels'] }
    val = { 'data': val, 'labels': data['labels'] }

    estim.fit(
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
        test_y[None, ...],
        data['labels'],
        slices['theta'],
        n_samples=n_post
    )
    theta_truth = theta[0]
    print(f'theta_truth: {theta_truth}')
    print(f'mean theta_hat: {jnp.mean(theta_hat["theta"])}')

    obs_sigma = sigma * jnp.eye(dim)
    ref_scale = jnp.linalg.inv(n_obs * obs_sigma)
    ref_loc = ref_scale @ n_obs * obs_sigma @ test_y + theta_truth # If reusing, theta needs to be scaled by std
    reference_posterior = tfd.Normal(
        loc=ref_loc,
        scale=ref_scale
    )

    test_key, key = jr.split(key)
    mmd = mmd_test(
        theta_hat['theta'],
        reference_posterior,
        test_key
    )

    assert mmd['reject_null']

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

    split_key, key = jr.split(key)
    train, val = split_data(data, train_size, .8, shuffle_rng=split_key)

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
    print(f'mean: {jnp.mean(z, axis=0)}, std: {jnp.std(z, axis=0)}')
    ks_response = vmap(
        lambda x: ks_norm_test(x, alpha=0.01),
        in_axes=(1,)
    )(z)
    print(f'Critical value: {ks_response["critical_value"]}, statistic: {ks_response["statistic"]}')
    assert jnp.all(~ks_response['reject_null'])

    if dim > 1:
        # Compute correlation matrix
        corr_matrix = jnp.corrcoef(z, rowvar=False)
        
        # Off-diagonal elements should be close to zero (independence)
        off_diagonal = corr_matrix[jnp.triu_indices(dim, k=1)]
        max_correlation = jnp.max(jnp.abs(off_diagonal))
        
        assert max_correlation < 0.1, (
            f"Dimensions are not independent. "
            f"Maximum correlation: {max_correlation}, "
            f"Correlation matrix: {corr_matrix}"
        )


# Helper function to create correlation matrices
def _create_corr_matrix(dim: int, strength: float = 0.6) -> jnp.ndarray:
    """Create correlation matrix with specified off-diagonal strength."""
    return jnp.full((dim, dim), strength) + jnp.eye(dim) * (1 - strength)

@pytest.mark.parametrize(
    "dim,train_size,theta_cov",
    [
        # No theta correlation (baseline)
        (2, 10_000, jnp.eye(2)),
        (2, 10_000, _create_corr_matrix(2, 0.6)),
        (3, 10_000, jnp.eye(3)),
        (3, 10_000, _create_corr_matrix(3, 0.6)),
    ],
)
def test_cnf_can_recover_base_distribution_from_correlated_training_set(
    dim, train_size, theta_cov
):
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

    # Create training data using provided theta covariance matrix
    sample_key, key = jr.split(key)
    theta_key, y_key = jr.split(sample_key)
    
    # Sample theta from multivariate normal with specified covariance
    theta = jr.multivariate_normal(
        theta_key, 
        mean=jnp.zeros(dim), 
        cov=theta_cov, 
        shape=(train_size,)
    )
    
    # Generate y with independent normal noise
    sigma = 1e-1
    noise = jr.normal(y_key, (train_size, dim * n_obs)) * sigma
    y = jnp.tile(theta, (1, n_obs)) + noise

    data = {
        'data': {
            'theta': theta,
            'y': y
        }
    }

    split_key, key = jr.split(key)
    train, val = split_data(data, train_size, .8, shuffle_rng=split_key)

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

    # Check each dimension is normal
    print(f'mean: {jnp.mean(z, axis=0)}, std: {jnp.std(z, axis=0)}')
    ks_response = vmap(
        lambda x: ks_norm_test(x, alpha=0.01),
        in_axes=(1,)
    )(z)
    print(f'Critical value: {ks_response["critical_value"]}, statistic: {ks_response["statistic"]}')
    assert jnp.all(~ks_response['reject_null'])

    # Compute correlation matrix - dimensions should be independent
    corr_matrix = jnp.corrcoef(z, rowvar=False)
    
    # Off-diagonal elements should be close to zero (independence)
    off_diagonal = corr_matrix[jnp.triu_indices(dim, k=1)]
    max_correlation = jnp.max(jnp.abs(off_diagonal))
    
    assert max_correlation < 0.1, (
        f"Dimensions are not independent. "
        f"Maximum correlation: {max_correlation}, "
        f"Correlation matrix: {corr_matrix}"
    )

@pytest.mark.parametrize(
    "dim,train_size,cov",
    [
        (1, 10_000, None),
        (3, 10_000, jnp.eye(3)),
        (3, 10_000, _create_corr_matrix(3, 0.6)),
    ],
)
def test_cnf_can_recover_base_distribution_from_posterior_estimate(
    dim,
    train_size,
    cov
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
    if cov is None:
        theta = jr.normal(sample_key, (train_size, dim))
    else:
        theta = jr.multivariate_normal(
            sample_key, 
            mean=jnp.zeros(dim), 
            cov=cov, 
            shape=(train_size,)
        )

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
    train, val = split_data(data, train_size, .8, shuffle_rng=split_key)

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

    sim_precision = sigma * jnp.eye(dim)
    theta_cov = cov if cov is not None else 1.
    ref_scale = jnp.linalg.inv(n_obs * sim_precision)
    ref_loc = ref_scale @ (n_obs * (sim_precision @ test_y)) + theta_cov @ theta_truth
    print(ref_loc.shape, ref_scale.shape)
    reference_posterior = tfd.Normal(
        loc=ref_loc,
        scale=ref_scale
    )

    test_key, key = jr.split(key)
    mmd = mmd_test(
        theta_hat,
        reference_posterior,
        test_key
    )

    assert mmd['reject_null']

@pytest.mark.parametrize(
    "dim,train_size,cov",
    [
        (2, 10_000, jnp.eye(2)),
        (2, 10_000, _create_corr_matrix(2, 0.6)),
    ],
)
def test_cnf_mixture_of_posteriors_recovers_base_distribution(
    dim, train_size, cov
):
    """
    Sanity check: Test that a mixture of posteriors projected to base distribution 
    remains normally distributed. This validates the assumption that if individual
    posteriors map to normal base distributions, their mixture should also be normal.
    """
    key = jr.PRNGKey(0)
    key, nnx_key = jr.split(key)
    rngs = nnx.Rngs(nnx_key)
    n_obs = 10
    n_contexts = 1_000 # Number of different observations to test
    n_samples_per_context = 1 # Samples per posterior

    nn = MLPVectorField(
        theta_dim=dim,
        context_dim=dim * n_obs,
        latent_dim=16,
        n_layers=1,
        dropout=.1,
        activation=nnx.relu,
        rngs=rngs
    )
    optimiser = optax.adam(3e-4)

    model = CNF(nn)
    estim = FMPE(model)

    # Create training data
    sample_key, key = jr.split(key)
    theta = jr.multivariate_normal(
        sample_key, 
        mean=jnp.zeros(dim), 
        cov=cov, 
        shape=(train_size,)
    )

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
    train, val = split_data(data, train_size, .8, shuffle_rng=split_key)

    estim.fit(
        train,
        val,
        n_iter=n_epochs_train,
        optimizer=optimiser,
        batch_size=train_size // 10
    )

    # Select n_contexts different observations for testing
    test_indices = jnp.arange(n_contexts)
    test_y = y[test_indices]  # Shape: (n_contexts, dim * n_obs)
    
    # Generate different RNG keys for theta_0 sampling in each context
    context_keys = jr.split(key, n_contexts)

    # Vectorized function to sample posterior and compute z for each context
    def sample_and_compute_z(single_test_y, context_key):
        # Generate theta_0 from normal distribution with unique RNG key
        theta_0 = jr.normal(context_key, (n_samples_per_context, dim))
        
        # Sample posterior for this context
        theta_hat = estim.sample_posterior(
            single_test_y[None, :],  # Add batch dimension
            theta_shape=(dim,),
            n_samples=n_samples_per_context,
            theta_0=theta_0
        )

        # Project posterior samples to base distribution
        z = estim.sample_base_dist(
            theta_hat,
            jnp.broadcast_to(
                single_test_y[None, :],
                (theta_hat.shape[0], n_obs * dim)
            ),
            (dim,)
        )
        return z

    def compute_z(single_theta, single_test_y):
        # Project posterior samples to base distribution
        z = estim.sample_base_dist(
            single_theta[None, :],
            single_test_y[None, :],
            (dim,)
        )
        return z

    # Vectorized function to sample posterior for each context (returning theta_hat)
    def sample_posterior_for_context(single_test_y, context_key):
        # Generate theta_0 from normal distribution with unique RNG key
        theta_0 = jr.normal(context_key, (n_samples_per_context, dim))
        
        # Sample posterior for this context
        theta_hat = estim.sample_posterior(
            single_test_y[None, :],  # Add batch dimension
            theta_shape=(dim,),
            n_samples=n_samples_per_context,
            theta_0=theta_0
        )
        return theta_hat

    # Use vmap to sample posteriors for all contexts
    all_theta_hat = vmap(sample_posterior_for_context, in_axes=(0, 0))(test_y, context_keys)  # Shape: (n_contexts, n_samples_per_context, dim)
    
    # Reshape theta_hat to combine all contexts
    theta_hat_combined = all_theta_hat.reshape(n_contexts * n_samples_per_context, dim)  # Shape: (n_contexts * n_samples_per_context, dim)

    # Create pairplots for theta_hat and theta
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    
    # # Create pairplot for theta_hat
    # theta_hat_df = pd.DataFrame(theta_hat_combined, columns=[f'theta_{i}' for i in range(dim)])
    # g1 = sns.PairGrid(theta_hat_df, diag_sharey=False)
    # g1.map_lower(sns.scatterplot, alpha=0.6)
    # g1.map_diag(sns.histplot, bins=30)
    # g1.fig.suptitle('theta_hat distribution (Posterior Samples)', y=1.02)
    # plt.savefig('theta_hat.png', bbox_inches='tight', dpi=150)
    # plt.close()
    
    # # Create pairplot for theta
    # theta_df = pd.DataFrame(theta[:test_y.shape[0]], columns=[f'theta_{i}' for i in range(dim)])
    # g2 = sns.PairGrid(theta_df, diag_sharey=False)
    # g2.map_lower(sns.scatterplot, alpha=0.6)
    # g2.map_diag(sns.histplot, bins=30)
    # g2.fig.suptitle('theta distribution (Ground Truth)', y=1.02)
    # plt.savefig('theta.png', bbox_inches='tight', dpi=150)
    # plt.close()

    # Use vmap to compute z values for all contexts at once
    all_z = vmap(sample_and_compute_z, in_axes=(0, 0))(test_y, context_keys)  # Shape: (n_contexts, n_samples_per_context, dim)


    # Reshape to concatenate all z samples into mixture
    z_mixture = all_z.reshape(n_contexts * n_samples_per_context, dim)  # Shape: (n_contexts * n_samples_per_context, dim)

    all_z_direct = vmap(compute_z, in_axes=(0, 0))(theta[:test_y.shape[0]], test_y)  # Shape: (n_contexts, n_samples_per_context, dim)
    z_direct_mixture = all_z_direct.reshape(n_contexts * n_samples_per_context, dim)  # Shape: (n_contexts * n_samples_per_context, dim)
    mixture_corr_matrix = jnp.corrcoef(z_direct_mixture, rowvar=False)
    mixture_off_diagonal = mixture_corr_matrix[jnp.triu_indices(dim, k=1)]
    mixture_max_correlation = jnp.max(jnp.abs(mixture_off_diagonal))
    print(f"Direct Mixture correlation matrix:\n{mixture_corr_matrix}")
    print(f"Direct Mixture max correlation: {mixture_max_correlation}")

    all_z_batch = vmap(compute_z, in_axes=(0, 0))(theta_hat_combined, test_y)  # Shape: (n_contexts, n_samples_per_context, dim)
    z_direct_mixture = all_z_batch.reshape(n_contexts * n_samples_per_context, dim)  # Shape: (n_contexts * n_samples_per_context, dim)
    mixture_corr_matrix = jnp.corrcoef(z_direct_mixture, rowvar=False)
    mixture_off_diagonal = mixture_corr_matrix[jnp.triu_indices(dim, k=1)]
    mixture_max_correlation = jnp.max(jnp.abs(mixture_off_diagonal))
    print(f"Batch Mixture correlation matrix:\n{mixture_corr_matrix}")
    print(f"Batch Mixture max correlation: {mixture_max_correlation}")

    # Statistical comparison between theta_hat and theta
    print("\n=== Comparison between theta_hat (combined) and theta ===")
    
    # Compute statistics for both distributions
    theta_hat_mean = jnp.mean(theta_hat_combined, axis=0)
    theta_hat_std = jnp.std(theta_hat_combined, axis=0)
    theta_mean = jnp.mean(theta, axis=0)
    theta_std = jnp.std(theta, axis=0)
    
    print(f"theta_hat mean: {theta_hat_mean}")
    print(f"theta mean: {theta_mean}")
    print(f"theta_hat std: {theta_hat_std}")
    print(f"theta std: {theta_std}")
    
    # Compute correlation matrices
    if dim > 1:
        theta_hat_corr = jnp.corrcoef(theta_hat_combined, rowvar=False)
        theta_corr = jnp.corrcoef(theta, rowvar=False)
        
        print(f"theta_hat correlation matrix:\n{theta_hat_corr}")
        print(f"theta correlation matrix:\n{theta_corr}")
        
        # Compare correlation matrices
        corr_diff = jnp.abs(theta_hat_corr - theta_corr)
        max_corr_diff = jnp.max(corr_diff)
        print(f"Max difference in correlation matrices: {max_corr_diff}")

    print(f"\nMixture: mean={jnp.mean(z_mixture, axis=0)}, "
          f"std={jnp.std(z_mixture, axis=0)}")

    # Test mixture normality
    mixture_ks_results = vmap(
        lambda x: ks_norm_test(x, alpha=0.01),
        in_axes=(1,)
    )(z_mixture)

    print(f"Mixture normality passed: {jnp.all(~mixture_ks_results['reject_null'])}")

    # Test mixture independence (for multi-dimensional case)
    mixture_corr_matrix = jnp.corrcoef(z_mixture, rowvar=False)
    mixture_off_diagonal = mixture_corr_matrix[jnp.triu_indices(dim, k=1)]
    mixture_max_correlation = jnp.max(jnp.abs(mixture_off_diagonal))
    
    print(f"Mixture correlation matrix: {mixture_corr_matrix}")
    print(f"Mixture max correlation: {mixture_max_correlation}")

    # Test mixture mean and std
    mixture_mean = jnp.mean(z_mixture, axis=0)
    mixture_std = jnp.std(z_mixture, axis=0)

    print(f"Mixture mean: {mixture_mean}")
    print(f"Mixture std: {mixture_std}")

    assert mixture_max_correlation < 0.1, (
        f"Mixture dimensions are not independent. "
        f"Maximum correlation: {mixture_max_correlation}, "
        f"Correlation matrix: {mixture_corr_matrix}"
    )

    assert jnp.allclose(mixture_mean, 0.0, atol=0.1), (
        f"Mixture mean not close to 0: {mixture_mean}"
    )
    assert jnp.allclose(mixture_std, 1.0, atol=0.1), (
        f"Mixture standard deviation not close to 1: {mixture_std}"
    )
