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
    MultiBinaryMLPClassifier,
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
    z_per_classifier = jr.normal(
        z_key,
        shape=(n_classifiers, N_cal * 2, latent_dim)
    )
    
    labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)
    labels_per_classifier = jnp.broadcast_to(
        labels[None, ...],
        (n_classifiers, N_cal * 2)
    )

    # TODO: why do I have to do this?
    graphdef, params = nnx.split(classifiers)

    @nnx.vmap
    def fit_multi_classifier(rng_key, params, z, labels):
        # add singleton dimension for multi-classifier forward pass
        params = tree.map(lambda x: x[None, ...], params)
        classifier = nnx.merge(graphdef, params)
        losses = fit_classifier(
            rng_key,
            classifier,
            z,  # z-only input (no x)
            labels,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        # remove singleton dimension
        params = nnx.state(classifier)
        params = tree.map(
            lambda leaf: leaf[0],
            params
        )
        return losses, params

    losses, params = fit_multi_classifier(
        jr.split(rng_key, n_classifiers),
        params,
        z_per_classifier,
        labels_per_classifier,
    )
    nnx.update(classifiers, params)
    return losses

def evaluate_c2st_nf(
    rng_key: jnp.ndarray,
    z_posterior_samples: jnp.ndarray,  # Z samples from estimated posterior
    main_classifier: BinaryMLPClassifier,
    null_classifier: MultiBinaryMLPClassifier,
    latent_dim: int,
    Nv: int  # Number of validation samples
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
    
    # Generate evaluation samples from N(0, I)
    rng_key, z_eval_key = jr.split(rng_key)
    z_eval = jr.normal(z_eval_key, shape=(Nv, latent_dim))

    # Compute test statistic on z_posterior_samples using main classifier
    # t̂MSE(z_posterior) = (1/N_posterior) * sum((d_main(Z_posterior_n) - 1/2)^2)
    d_posterior = main_classifier(z_posterior_samples)
    t_mse_val = jnp.mean((d_posterior - 0.5)**2)

    # Compute null test statistics using null classifiers on z_eval
    # t̂null_h = (1/Nv) * sum((d_null_h(Z_eval_n) - 1/2)^2)
    d_null = null_classifier(z_eval)
    null_test_statistics = jnp.mean((d_null - 0.5)**2, axis=1)

    # Compute p-value
    # p̂ = (1/N_null) * sum(I(t̂null_h >= t̂MSE))
    p_value = jnp.mean(null_test_statistics >= t_mse_val)
    # Add a small value to prevent p_value from being 0
    p_value = jnp.maximum(p_value, 1.0 / (null_classifier.n + 1))

    return null_test_statistics, t_mse_val, p_value