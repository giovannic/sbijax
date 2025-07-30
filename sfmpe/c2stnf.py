from typing import Callable, Tuple
from pathlib import Path
from jaxtyping import PyTree
from jax import random as jr, numpy as jnp, tree
from flax import nnx
import optax
from .train import fit_model_no_branch

import numpy as np
from matplotlib import pyplot as plt

# Import shared classifier components from lc2stnf
from .lc2stnf import (
    FFLayer,
    BinaryMLPClassifier,
    _ce_loss,
    fit_classifier,
    lc2st_quant_plot
)

def train_c2st_nf_main_classifier(
        rng_key: jnp.ndarray,
        classifier: BinaryMLPClassifier,
        z_samples: jnp.ndarray,  # Z samples from the estimated posterior
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
    print(f'z base mean: {jnp.mean(z_base)}')
    print(f'z base std: {jnp.std(z_base)}')

    # Class C=1: Z_q_n from the estimated posterior
    print(f'z samples mean: {jnp.mean(z_samples)}')
    print(f'z samples std: {jnp.std(z_samples)}')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create scatter plot with marginal histograms for z_base
    fig1 = plt.figure(figsize=(8, 8))
    if m > 1:
        g1 = sns.jointplot(x=z_base[:, 0], y=z_base[:, 1], 
                          kind='scatter', marginal_kws={'bins': 30})
    else:
        g1 = sns.jointplot(x=z_base[:, 0], y=z_base[:, 0], 
                          kind='scatter', marginal_kws={'bins': 30})
    g1.fig.suptitle('z_base distribution')
    plt.savefig('z_base.png')
    plt.close()
    
    # Create scatter plot with marginal histograms for z_samples
    fig2 = plt.figure(figsize=(8, 8))
    if m > 1:
        g2 = sns.jointplot(x=z_samples[:, 0], y=z_samples[:, 1], 
                          kind='scatter', marginal_kws={'bins': 30})
    else:
        g2 = sns.jointplot(x=z_samples[:, 0], y=z_samples[:, 0], 
                          kind='scatter', marginal_kws={'bins': 30})
    g2.fig.suptitle('z_samples distribution')
    plt.savefig('z_sampled.png')
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

def evaluate_c2st_nf(
    rng_key: jnp.ndarray,
    z_posterior_samples: jnp.ndarray,  # Z samples from estimated posterior
    classifier: BinaryMLPClassifier,
    latent_dim: int,
    Nv: int,  # Number of validation samples per null evaluation
    N_null: int  # Number of null evaluations
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    C2ST-NF, evaluating test statistics and p-values using z samples from estimated posterior.
    Input:
        rng_key: JAX PRNG key.
        z_posterior_samples: Z samples from the estimated posterior
        classifier: main classifer for z-only input (trained to differentiate z_samples from normal)
        latent_dim: dimension of the latent space
        Nv: Number of samples for validation per null evaluation.
        N_null: Number of null evaluations to perform
    Output:
        null_test_statistics: test statistics for each null evaluation (N_null,)
        t_mse_val: The calculated MSE test statistic for z_posterior_samples.
        p_value: The p-value for the consistency test.
    """
    
    # Compute test statistic on z_posterior_samples
    # t̂MSE(z_posterior) = (1/N_posterior) * sum((d(Z_posterior_n) - 1/2)^2)
    d_posterior = classifier(z_posterior_samples)
    t_mse_val = jnp.mean((d_posterior - 0.5)**2)

    # Generate null test statistics by evaluating classifier on normal samples N_null times
    rng_keys = jr.split(rng_key, N_null)
    
    def compute_null_stat(key):
        # Generate Nv samples from N(0, I)
        z_eval = jr.normal(key, shape=(Nv, latent_dim))
        # Evaluate classifier
        d_null = classifier(z_eval)
        # Compute test statistic
        return jnp.mean((d_null - 0.5)**2)
    
    # Vectorized computation of null statistics
    null_test_statistics = jnp.array([compute_null_stat(key) for key in rng_keys])

    # Compute p-value
    # p̂ = (1/N_null) * sum(I(t̂null_i >= t̂MSE))
    p_value = jnp.mean(null_test_statistics >= t_mse_val)
    # Add a small value to prevent p_value from being 0
    p_value = jnp.maximum(p_value, 1.0 / (N_null + 1))

    return null_test_statistics, t_mse_val, p_value