from typing import Callable, Tuple
from pathlib import Path
from jaxtyping import PyTree
from jax import random as jr, numpy as jnp, tree
from flax import nnx
import optax
from .train import fit_model_no_branch
from .utils import split_data

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
        latent_dim,
        n_layers,
        activation,
        rngs=nnx.Rngs(0)
        ):

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs):
            return FFLayer(latent_dim, activation, rngs=rngs)

        self.layers = create_layer(rngs)
        self.n_layers = n_layers
        self.in_layer = nnx.Linear(dim, latent_dim, rngs=rngs)
        self.output = nnx.Linear(latent_dim, 1, rngs=rngs)

    def __call__(self, u):
        @nnx.split_rngs(splits=self.n_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x)
            return x

        x = self.in_layer(u)
        x = forward(x, self.layers)
        x = self.output(x)[..., 0]
        return nnx.sigmoid(x)

class MultiBinaryMLPClassifier(nnx.Module):

    def __init__(
        self,
        dim: int,
        latent_dim: int,
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
            return BinaryMLPClassifier(dim, latent_dim, n_layers, activation, rngs=rngs)

        self.classifiers = create_classifier(rngs)
        self.n = n

    def __call__(self, u):
        assert u.ndim == 3, f"MultiBinaryMLPClassifier expects 3D input, got {u.ndim}D with shape {u.shape}"
        assert u.shape[1] == self.n, f"Second dimension must match number of classifiers ({self.n}), got shape {u.shape}"
        
        # Input shape is (batch_dim, n_classifiers, z_dim)
        # We want each classifier to process all batch samples for its corresponding slice
        @nnx.vmap(in_axes=(0, 1), out_axes=1)
        def call_classifier_on_slice(cls, u_slice):
            return cls(u_slice)
        
        return call_classifier_on_slice(self.classifiers, u)

Classifier = BinaryMLPClassifier | MultiBinaryMLPClassifier

def train_l_c2st_nf_main_classifier(
        rng_key: jnp.ndarray,
        classifier: BinaryMLPClassifier,
        calibration_data: Tuple[jnp.ndarray, jnp.ndarray], # D_cal = (Theta_cal, X_cal)
        z_samples: jnp.ndarray,
        num_epochs: int = 100,
        batch_size: int = 100
    ):
    """
    ℓ-C2ST-NF – training the main classifier
    Input:
        rng_key: JAX PRNG key for reproducibility.
        classifier: Binary classifier
        calibration_data: Tuple of (Theta_n, X_n) sampled from p(theta, x).
        z_samples: Pre-computed z samples from the inverse NF transformation.
        num_epochs: Number of epochs to train the classifier.
    """
    theta_cal, x_cal = calibration_data
    N_cal = x_cal.shape[0]
    m = theta_cal.shape[1] # Dimension of the latent space Z

    # Construct Classification Training Set
    # Class C=0: (Z_n, X_n) where Z_n ~ N(0, Im), X_n from D_cal
    rng_key, z_key = jr.split(rng_key)
    z_base = jr.normal(z_key, shape=(N_cal, m))
    print(f'z base mean: {jnp.mean(z_base)}')
    print(f'z base std: {jnp.std(z_base)}')

    # Class C=1: (Z_q_n, X_n) where Z_q_n = T_phi_inv(Theta_n, X_n), X_n from D_cal
    # Use the pre-computed z_samples
    print(f'z mean: {jnp.mean(z_samples)}')
    print(f'z std: {jnp.std(z_samples)}')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create scatter plot with marginal histograms for z_base
    fig1 = plt.figure(figsize=(8, 8))
    g1 = sns.jointplot(x=jnp.mean(x_cal, axis=1), y=z_base[:, 1], kind='scatter', 
                       marginal_kws={'bins': 30})
    g1.fig.suptitle('z_base distribution')
    plt.savefig('z_base.png')
    plt.close()
    
    # Create scatter plot with marginal histograms for z_samples
    fig2 = plt.figure(figsize=(8, 8))
    g2 = sns.jointplot(x=jnp.mean(x_cal, axis=1), y=z_samples[:, 1], kind='scatter',
                       marginal_kws={'bins': 30})
    g2.fig.suptitle('z_samples distribution')
    plt.savefig('z_sampled.png')
    plt.close()

    # Combine data and labels for training
    z = jnp.concatenate([z_base, z_samples], axis=0)
    x = jnp.concatenate([x_cal, x_cal], axis=0)
    labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0) # 0 for C=0, 1 for C=1

    # Create u = concatenate([z, x]) for the refactored classifier
    u = jnp.concatenate([z, x], axis=-1)

    fit_classifier(
        rng_key,
        classifier, 
        u,
        labels,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Evaluate classifier performance
    u_samples = jnp.concatenate([z_samples, x_cal], axis=-1)
    u_base = jnp.concatenate([z_base, x_cal], axis=-1)
    d_score = classifier(u_samples)
    d_score_base = classifier(u_base)
    print('d score mean: ', jnp.mean(d_score))
    print('d score base mean: ', jnp.mean(d_score_base))

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
    x_dim = x_cal.shape[1]
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
        (n_classifiers, N_cal * 2, x_dim)
    ) # X_n from D_cal
    
    # Create u = concatenate([z, x]) for each classifier
    u_per_classifier = jnp.concatenate([z_per_classifier, x_per_classifier], axis=-1)

    labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)
    labels_per_classifier = jnp.broadcast_to(
        labels[None, ...],
        (n_classifiers, N_cal * 2)
    )

    #TODO: why do I have to do this?
    graphdef, params = nnx.split(classifiers)

    @nnx.vmap
    def fit_multi_classifier(rng_key, params, u, labels):
        # add singleton dimension for multi-classifier forward pass
        params = tree.map(lambda x: x[None, ...], params)
        classifier = nnx.merge(graphdef, params)
        losses = fit_classifier(
            rng_key,
            classifier,
            u,
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
        u_per_classifier,
        labels_per_classifier,
    )
    nnx.update(classifiers, params)
    return losses

def _ce_loss(model: Classifier, _: jnp.ndarray, batch: PyTree) -> jnp.ndarray:
    """Calculates binary cross-entropy loss for the classifier."""
    preds = model(batch['data']['u'])
    labels = batch['data']['y']
    preds = jnp.clip(preds, 1e-7, 1 - 1e-7) # Clip to avoid log(0)
    loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))
    return loss

def fit_classifier(
        seed: jnp.ndarray,
        classifier: Classifier, 
        u: jnp.ndarray,
        labels: jnp.ndarray,
        num_epochs=100,
        batch_size=100,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
    ):

    data = {"data": {"u": u, "y": labels}}
    rng_key, shuffle_key = jr.split(seed)
    train, val = split_data(data, u.shape[0], split=0.8, shuffle_rng=shuffle_key)
    
    n_train = int(u.shape[0] * 0.8)
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

    # Create u = concatenate([z, x]) for evaluation
    u_eval = jnp.concatenate([z_eval, xo_expanded], axis=-1)

    d_null = null_classifier(u_eval)
    null_test_statistics = jnp.mean((d_null - 0.5)**2, axis=0)

    # Compute Test Statistics
    # t̂MSE0(xo) = (1/Nv) * sum((d(Z_n, xo) - 1/2)^2)
    d_main = main_classifier(u_eval)
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
