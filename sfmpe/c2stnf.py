from typing import Tuple
from jax import random as jr, numpy as jnp

from .lc2stnf import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    fit_classifier,
)

def train_c2st_nf_main_classifier(
        rng_key: jnp.ndarray,
        classifier: BinaryMLPClassifier,
        z_samples: jnp.ndarray,
        num_epochs: int = 100,
        batch_size: int = 100
    ):
    """
    C2ST-NF – training the main classifier to differentiate z_samples from normal distribution
    Input:
        rng_key: JAX PRNG key for reproducibility.
        classifier: Binary classifier for z-only input
        z_samples: Z samples from the estimated posterior
        num_epochs: Number of epochs to train the classifier.
    """
    N_cal = z_samples.shape[0]
    m = z_samples.shape[1]  # Dimension of the latent space Z

    # Construct Classification Training Set
    # Class C=0: Z_n where Z_n ~ N(0, Im)
    rng_key, z_key = jr.split(rng_key)
    z_base = jr.normal(z_key, shape=(N_cal, m))
    print(f'z base mean: {jnp.mean(z_base, axis=0)}')
    print(f'z base std: {jnp.std(z_base, axis=0)}')

    # Class C=1: Z_q_n from the estimated posterior
    print(f'z samples mean: {jnp.mean(z_samples, axis=0)}')
    print(f'z samples std: {jnp.std(z_samples, axis=0)}')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Create pairplot for z_base
    z_base_df = pd.DataFrame(z_base, columns=[f'z_{i}' for i in range(m)])
    g1 = sns.PairGrid(z_base_df, diag_sharey=False)
    g1.map_lower(sns.scatterplot, alpha=0.6)
    g1.map_diag(sns.histplot, bins=30)
    g1.fig.suptitle('z_base distribution (Standard Normal)', y=1.02)
    plt.savefig('z_base.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Create pairplot for z_samples
    z_samples_df = pd.DataFrame(z_samples, columns=[f'z_{i}' for i in range(m)])
    g2 = sns.PairGrid(z_samples_df, diag_sharey=False)
    g2.map_lower(sns.scatterplot, alpha=0.6)
    g2.map_diag(sns.histplot, bins=30)
    g2.fig.suptitle('z_samples distribution (Estimated Posterior)', y=1.02)
    plt.savefig('z_sampled.png', bbox_inches='tight', dpi=150)
    plt.close()

    # Combine data and labels for training (z only, no x)
    u = jnp.concatenate([z_base, z_samples], axis=0)  # u = z (no concatenation with x)
    labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0) # 0 for C=0, 1 for C=1

    fit_classifier(
        rng_key,
        classifier, 
        u,
        labels,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Evaluate classifier performance
    d_score = classifier(z_samples)
    d_score_base = classifier(z_base)
    print('d score mean: ', jnp.mean(d_score))
    print('d score base mean: ', jnp.mean(d_score_base))

def precompute_c2st_nf_null_classifiers(
    rng_key: jnp.ndarray,
    classifiers: MultiBinaryMLPClassifier,
    latent_dim: int,
    N_cal: int,
    num_epochs: int = 100,
    batch_size: int = 100
):
    """
    C2ST-NF – precompute the null distribution classifiers.
    Input:
        rng_key: JAX PRNG key.
        classifiers: MultiBinaryMLPClassifier for null distribution.
        latent_dim: Dimension of the latent space Z.
        N_cal: Number of calibration samples per classifier.
        num_epochs: Number of epochs for each null classifier.
        batch_size: Batch size for training.
    """
    n_classifiers = classifiers.n

    rng_key, z_key = jr.split(rng_key)

    # Construct classification training set under the Null
    # Both classes (C=0 and C=1) now draw their Z samples from N(0, I_m)
    # Shape: (batch_dim, n_classifiers, z_dim)
    z_per_classifier = jr.normal(
        z_key,
        shape=(N_cal * 2, n_classifiers, latent_dim)
    )
    
    labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)
    # Broadcast labels to match the new input format: (batch_dim, n_classifiers)
    labels_per_classifier = jnp.broadcast_to(
        labels[:, None],
        (N_cal * 2, n_classifiers)
    )

    # Train all classifiers together using the 3D input format
    losses = fit_classifier(
        rng_key,
        classifiers,
        z_per_classifier,  # 3D input: (batch_dim, n_classifiers, z_dim)
        labels_per_classifier,  # 2D labels: (batch_dim, n_classifiers)
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    return losses

def evaluate_c2st_nf(
    rng_key: jnp.ndarray,
    z_posterior_samples: jnp.ndarray,  # Z samples from estimated posterior
    main_classifier: BinaryMLPClassifier,
    null_classifier: MultiBinaryMLPClassifier,
    latent_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    C2ST-NF, evaluating test statistics and p-values using z samples from estimated posterior.
    Input:
        rng_key: JAX PRNG key.
        z_posterior_samples: Z samples from the estimated posterior
        main_classifier: main classifier for z-only input (trained to differentiate z_samples from normal)
        null_classifier: null classifiers trained under null hypothesis
        latent_dim: dimension of the latent space
        Nv: Number of samples for validation.
    Output:
        null_test_statistics: test statistics for each null classifier
        t_mse_val: The calculated MSE test statistic for z_posterior_samples.
        p_value: The p-value for the consistency test.
    """
    Nv = z_posterior_samples.shape[0]
    
    # Generate different evaluation samples from N(0, I) for each null classifier
    # Shape: (batch_dim, n_classifiers, z_dim)
    rng_key, z_eval_key = jr.split(rng_key)
    z_eval = jr.normal(z_eval_key, shape=(Nv, null_classifier.n, latent_dim))

    # Compute test statistic on z_posterior_samples using main classifier
    # t̂MSE(z_posterior) = (1/N_posterior) * sum((d_main(Z_posterior_n) - 1/2)^2)
    d_posterior = main_classifier(z_posterior_samples)
    t_mse_val = jnp.mean((d_posterior - 0.5)**2)

    # Compute null test statistics using null classifiers on different z_eval per classifier
    # t̂null_h = (1/Nv) * sum((d_null_h(Z_eval_h_n) - 1/2)^2)
    # d_null will have shape (Nv, n_classifiers)
    d_null = null_classifier(z_eval)
    null_test_statistics = jnp.mean((d_null - 0.5)**2, axis=0)

    # Compute p-value
    # p̂ = (1/N_null) * sum(I(t̂null_h >= t̂MSE))
    p_value = jnp.mean(null_test_statistics >= t_mse_val)
    # Add a small value to prevent p_value from being 0
    p_value = jnp.maximum(p_value, 1.0 / (null_classifier.n + 1))

    return null_test_statistics, t_mse_val, p_value
