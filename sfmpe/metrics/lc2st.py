from typing import Tuple
from jax import random as jr, numpy as jnp
from jaxtyping import Array

# Import shared classifier components from lc2stnf
from .lc2stnf import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    fit_classifier,
)

def train_lc2st_classifiers(
    rng_key: Array,
    d_cal: Tuple[Array, Array, Array], 
    classifier: BinaryMLPClassifier,
    null_classifier: MultiBinaryMLPClassifier,
    num_epochs: int,
    batch_size: int = 100
) -> None:
    """
    Local-Classifier 2 Sample Test – training both main and null classifiers.
    
    Input:
        rng_key: JAX PRNG key for reproducibility.
        d_cal: Calibration data tuple (x, theta, theta_q) where:
            - x: observations from p(x|theta)
            - theta: parameters from joint distribution p(theta, x)  
            - theta_q: parameters from estimated posterior q(theta|x)
        classifier: Main binary classifier for (x, theta) concatenated input
        null_classifier: Multiple null classifiers for permuted label training
        num_epochs: Number of epochs to train both classifiers.
        batch_size: Batch size for training.
    """
    x_cal, theta_cal, theta_q = d_cal
    N_cal = x_cal.shape[0]

    # Train main classifier
    # Class C=0: (x, theta) from joint distribution p(theta, x)
    # Class C=1: (x, theta_q) from posterior q(theta|x) p(x)
    u_joint = jnp.concatenate([x_cal, theta_cal], axis=-1)
    u_posterior = jnp.concatenate([x_cal, theta_q], axis=-1)
    u_main = jnp.concatenate([u_joint, u_posterior], axis=0)
    labels_main = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)

    rng_key, main_key = jr.split(rng_key)
    fit_classifier(
        main_key,
        classifier,
        u_main,
        labels_main,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    # Train null classifiers with permuted labels
    n_null = null_classifier.n
    rng_key, null_key = jr.split(rng_key)

    # Create 3D input for null classifiers: (batch_size, n_classifiers, feature_dim)
    u_null_base = jnp.concatenate([u_joint, u_posterior], axis=0)  # (2*N_cal, feature_dim)
    u_null = jnp.broadcast_to(
        u_null_base[None, :, :], 
        (n_null, 2*N_cal, u_null_base.shape[-1])
    ).transpose(1, 0, 2)  # (2*N_cal, n_classifiers, feature_dim)

    # Generate different permuted labels for each null classifier
    base_labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)
    null_keys = jr.split(null_key, n_null)
    
    def permute_labels(key):
        return jr.permutation(key, base_labels)
    
    # Create permuted labels for each classifier
    permuted_labels = jnp.stack([
        permute_labels(key)
        for key in null_keys
    ], axis=1)  # (2*N_cal, n_classifiers)
    
    fit_classifier(
        null_key,
        null_classifier,
        u_null,
        permuted_labels,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

def evaluate_lc2st(
    observation: Array,
    posterior_samples: Array, 
    main_classifier: BinaryMLPClassifier,
    null_classifier: MultiBinaryMLPClassifier,
) -> Tuple[Array, Array, Array]:
    """
    Local-Classifier 2 Sample Test evaluation in posterior space.
    
    Input:
        observation: The specific observation to evaluate consistency at.
        posterior_samples: Samples from estimated posterior q(theta|observation).
        main_classifier: Main classifier trained to distinguish joint vs posterior.
        null_classifier: Null classifiers trained with permuted labels.
    
    Output:
        null_test_statistics: Test statistics for each null classifier.
        t_mse_val: The calculated MSE test statistic for posterior samples.
        p_value: The p-value for the Local-Classifier 2 Sample Test.
    """
    n_samples = posterior_samples.shape[0]
    n_null = null_classifier.n
    
    # Create inputs for main classifier: (observation, posterior_samples)
    observation_broadcast = jnp.broadcast_to(
        observation[None, :], 
        (n_samples, observation.shape[0])
    )
    u_main = jnp.concatenate([observation_broadcast, posterior_samples], axis=-1)
    
    # Compute main test statistic
    # t̂MSE = (1/n_samples) * sum((d_main(observation, theta_i) - 1/2)^2)
    d_main = main_classifier(u_main)
    t_mse_val = jnp.mean((d_main - 0.5)**2)
    
    # Create 3D inputs for null classifiers
    u_null = jnp.broadcast_to(
        u_main[None, :, :],
        (n_null, n_samples, u_main.shape[-1])
    ).transpose(1, 0, 2)  # (n_samples, n_null, feature_dim)
    
    # Compute null test statistics
    # t̂null_h = (1/n_samples) * sum((d_null_h(observation, theta_i) - 1/2)^2)
    d_null = null_classifier(u_null)  # (n_samples, n_null)
    null_test_statistics = jnp.mean((d_null - 0.5)**2, axis=0)  # (n_null,)
    
    # Compute p-value
    # p̂ = (1/n_null) * sum(I(t̂null_h >= t̂MSE))
    p_value = jnp.mean(null_test_statistics >= t_mse_val)
    # Add small value to prevent p_value from being 0
    p_value = jnp.maximum(p_value, 1.0 / (n_null + 1))
    
    return null_test_statistics, t_mse_val, p_value
