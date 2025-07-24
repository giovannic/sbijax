from typing import Callable, Tuple
from pathlib import Path
from jaxtyping import PyTree
from jax import random as jr, numpy as jnp, tree, vmap
from flax import nnx
import optax
from .train import fit_model_no_branch

import numpy as np
from matplotlib import pyplot as plt

class FFLayer(nnx.Module):

    activation: Callable

    def __init__(self, dim, activation, rngs=nnx.Rngs(0)):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class BinaryMLPClassifier(nnx.Module):

    n_layers: int

    def __init__(
        self,
        dim,
        n_layers,
        activation,
        rngs=nnx.Rngs(0)
        ):

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs):
            return FFLayer(dim, activation, rngs=rngs)

        self.layers = create_layer(rngs)
        self.n_layers = n_layers
        self.output = nnx.Linear(dim, 1, rngs=rngs)

    def __call__(self, z, x):
        @nnx.split_rngs(splits=self.n_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x)
            return x

        x = jnp.concatenate([z, x], axis=-1)
        x = forward(x, self.layers)
        x = self.output(x)[..., 0]
        return nnx.sigmoid(x)

class MultiBinaryMLPClassifier(nnx.Module):

    def __init__(
        self,
        dim: int,
        n_layers: int,
        activation: Callable,
        n: int,
        rngs=nnx.Rngs(0)
        ):
        """
        dim: int the latent dimension
        n_layers: int number of layers
        activation: Callable activation function
        n: int number of classifiers to map over
        rngs: nnx.Rngs
        """
        @nnx.split_rngs(splits=n)
        @nnx.vmap
        def create_classifier(rngs):
            return BinaryMLPClassifier(dim, n_layers, activation, rngs=rngs)

        self.classifiers = create_classifier(rngs)
        self.n = n

    def __call__(self, z, x):
        @nnx.vmap
        def call_classifier(cls):
            return cls(z, x)

        return call_classifier(self.classifiers)

Classifier = BinaryMLPClassifier | MultiBinaryMLPClassifier

def train_l_c2st_nf_main_classifier(
        rng_key: jnp.ndarray,
        classifier: BinaryMLPClassifier,
        calibration_data: Tuple[jnp.ndarray, jnp.ndarray], # D_cal = (Theta_cal, X_cal)
        inverse_transform: Callable,
        num_epochs: int = 100,
        batch_size: int = 100
    ):
    """
    ℓ-C2ST-NF – training the main classifier
    Input:
        rng_key: JAX PRNG key for reproducibility.
        classifier: Binary classifier
        calibration_data: Tuple of (Theta_n, X_n) sampled from p(theta, x).
        inverse_transform: Callable that applies the inverse NF transformation .
        num_epochs: Number of epochs to train the classifier.
    """
    theta_cal, x_cal = calibration_data
    N_cal = x_cal.shape[0]
    m = theta_cal.shape[1] # Dimension of the latent space Z

    # Construct Classification Training Set
    # Class C=0: (Z_n, X_n) where Z_n ~ N(0, Im), X_n from D_cal
    rng_key, z_c0_key = jr.split(rng_key)
    z_c0_samples = jr.normal(z_c0_key, shape=(N_cal, m))
    x_c0_samples = x_cal # Use original X_n from D_cal

    # Class C=1: (Z_q_n, X_n) where Z_q_n = T_phi_inv(Theta_n, X_n), X_n from D_cal
    # This involves applying the inverse NF transformation to the true (theta, x) pairs.
    z_q_c1_samples = inverse_transform(theta_cal, x_cal)
    x_c1_samples = x_cal # Use original X_n from D_cal

    # Combine data and labels for training
    z = jnp.concatenate([z_c0_samples, z_q_c1_samples], axis=0)
    x = jnp.concatenate([x_c0_samples, x_c1_samples], axis=0)
    labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0) # 0 for C=0, 1 for C=1

    fit_classifier(
        rng_key,
        classifier, 
        x,
        z,
        labels,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

def precompute_null_distribution_nf_classifiers(
		rng_key,
		calibration_data: Tuple[jnp.ndarray, jnp.ndarray], # D_cal = (Theta_cal, X_cal)
		classifiers: MultiBinaryMLPClassifier,
		num_epochs: int = 100,
        batch_size: int = 100
	):
    """
    ℓ-C2ST-NF – precompute the null distribution for any estimator.
    Input:
        rng_key: JAX PRNG key.
        calibration_data: Tuple of (Theta_n, X_n) sampled from p(theta, x).
        classifiers: List of classifiers.
        num_epochs: Number of epochs for each null classifier.
    """
    theta_cal, x_cal = calibration_data
    N_cal = x_cal.shape[0]
    m = theta_cal.shape[1]
    n_classifiers = classifiers.n

    rng_key, z_key = jr.split(rng_key)

    # Construct classification training set under the Null [2]
    # Both classes (C=0 and C=1) now draw their latent samples (Z) from N(0, Im)
    z_per_classifier = jr.normal(
        z_key,
        shape=(n_classifiers, N_cal * 2, m)
    )
    x_per_classifier = jnp.broadcast_to(
        jnp.tile(x_cal, (2, 1))[None, ...],
        (n_classifiers, N_cal * 2, m)
    ) # X_n from D_cal

    labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)
    labels_per_classifier = jnp.broadcast_to(
        labels[None, ...],
        (n_classifiers, N_cal * 2)
    )

    graphdef, params = nnx.split(classifiers)

    @nnx.vmap
    def fit_multi_classifier(rng_key, params, x, z, labels):
        # add singleton dimension for multi-classifier forward pass
        params = tree.map(lambda x: x[None, ...], params)
        classifier = nnx.merge(graphdef, params)
        losses = fit_classifier(
            rng_key,
            classifier,
            x,
            z,
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
        x_per_classifier,
        z_per_classifier,
        labels_per_classifier,
    )
    nnx.update(classifiers, params)
    return losses

def _ce_loss(model: Classifier, _: jnp.ndarray, batch: PyTree) -> jnp.ndarray:
    """Calculates binary cross-entropy loss for the classifier."""
    preds = model(batch['data']['z'], batch['data']['x'])
    labels = batch['data']['y']
    preds = jnp.clip(preds, 1e-7, 1 - 1e-7) # Clip to avoid log(0)
    loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))
    return loss

def fit_classifier(
        seed: jnp.ndarray,
        classifier: Classifier, 
        x: jnp.ndarray,
        z: jnp.ndarray,
        labels: jnp.ndarray,
        num_epochs=100,
        batch_size=100,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
    ):

    data = {"data": {"x": x, "z": z, "y": labels}}
    split = .8
    n_train = int(x.shape[0] * split)
    train = tree.map(lambda x: x[:n_train], data)
    val = tree.map(lambda x: x[n_train:], data)

    assert n_train % batch_size == 0, f"Batch size ({batch_size}) must divide n_train ({n_train})"

    return fit_model_no_branch(
        classifier,
        seed,
        loss_fn=_ce_loss,
        train=train,
        val=val,
        optimizer=optimizer,
        n_iter=num_epochs,
        batch_size=batch_size
    )

def evaluate_l_c2st_nf(
    rng_key: jnp.ndarray,
    xo: jnp.ndarray, # The specific observation for local evaluation
    main_classifier: BinaryMLPClassifier,
    null_classifier: MultiBinaryMLPClassifier,
    latent_dim: int,
    Nv: int # Number of validation samples
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    ℓ-C2ST-NF, evaluating test statistics and p-values for a given xo.
    Input:
        rng_key: JAX PRNG key.
        xo: The specific observation (x_0) to evaluate consistency at.
        main_classifier: main classifer
        null_classifiers: List of null classifiers
        Nv: Number of samples for validation.
    Output:
        null_test_statistics: test statistics for each null classifier
        t_mse0_val: The calculated MSE test statistic for xo.
        p_value: The p-value for the local consistency at xo.
    """
    # Create a batched version of the classifier's apply function for efficiency
    # This maps over the Nv samples (z_samples) and repeats xo for each.

    # Generate Samples and Predicted Probabilities
    # For ℓ-C2ST-NF evaluation, the classifier is evaluated on Nv samples from N(0, Im)
    rng_key, z_eval_key = jr.split(rng_key)
    z_eval = jr.normal(
        z_eval_key,
        shape=(Nv, latent_dim)
    )
    
    # xo needs to be broadcast to (Nv, d)
    xo_expanded = xo[None, ...]
    xo_expanded = jnp.broadcast_to(
        xo_expanded,
        (Nv, xo.shape[-1])
    )

    d_null = null_classifier(z_eval, xo_expanded)
    null_test_statistics = jnp.mean((d_null - 0.5)**2, axis=0)

    # Compute Test Statistics
    # t̂MSE0(xo) = (1/Nv) * sum((d(Z_n, xo) - 1/2)^2)
    d_main = main_classifier(z_eval, xo_expanded)
    t_mse0_val = jnp.mean((d_main - 0.5)**2)

    # Compute p-value
    # p̂(xo) = (1/NH) * sum(I(t̂h(xo) > t̂MSE0(xo)))
    p_value = jnp.mean(null_test_statistics >= t_mse0_val)
    # Add a small value to the denominator to prevent p_value from being 0 if all nulls are smaller
    # This also helps to ensure the lowest p-value is 1/(NH+1) rather than 0.
    p_value = jnp.maximum(p_value, 1.0 / (null_classifier.n + 1))

    return null_test_statistics, t_mse0_val, p_value

def lc2st_quant_plot(
    T_data: jnp.ndarray,
    T_null: jnp.ndarray,
    p_value: jnp.ndarray,
    reject: bool,
    conf_alpha: float,
    out_path: Path
    ):

    _, axes = plt.subplots(1, 1, figsize=(20, 10))

    # plot 95% confidence interval
    quantiles = np.quantile(T_null, [0, 1-conf_alpha])
    axes.hist(T_null, bins=50, density=True, alpha=0.5, label="Null")
    axes.axvline(float(T_data), color="red", label="Observed")
    axes.axvline(quantiles[0], color="black", linestyle="--", label="95% CI")
    axes.axvline(quantiles[1], color="black", linestyle="--")
    axes.set_xlabel("Test statistic")
    axes.set_ylabel("Density")
    axes.set_xlim(-0.01,0.25)
    axes.set_title(
            f"p-value = {p_value:.3f}, reject = {reject}"
            )
    axes.legend(bbox_to_anchor=(1.1, .5), loc='center left')
    plt.savefig(out_path)
