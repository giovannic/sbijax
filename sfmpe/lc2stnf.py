from typing import Callable, List, Tuple
from jaxtyping import PyTree
from jax import random as jr, numpy as jnp
from flax import nnx
import optax
from .util.dataloader import as_batch_iterators
from .train import fit_model

class FFLayer(nnx.Module):

    activation: Callable

    def __init__(self, dim, activation, rngs=nnx.Rngs(0)):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class BinaryMLPlassifier(nnx.Module):

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
        self.output = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, z, x):
        @nnx.split_rngs(splits=self.n_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x)
            return x

        x = jnp.concatenate([z, x], axis=-1)
        x = forward(x, self.layers)
        x = self.output(x)
        return nnx.sigmoid(x)

def train_l_c2st_nf_main_classifier(
        rng_key: jnp.ndarray,
        classifier: BinaryMLPlassifier,
        calibration_data: Tuple[jnp.ndarray, jnp.ndarray], # D_cal = (Theta_cal, X_cal)
        inverse_transform: Callable,
        latent_dim: int,
        num_epochs: int = 100
    ):
    """
    ℓ-C2ST-NF – training the main classifier
    Input:
        rng_key: JAX PRNG key for reproducibility.
        classifier: Binary classifier
        calibration_data: Tuple of (Theta_n, X_n) sampled from p(theta, x).
        num_epochs: Number of epochs to train the classifier.
    """
    theta_cal, x_cal = calibration_data
    N_cal = x_cal.shape[0]
    m = latent_dim # Dimension of the latent space Z

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
        num_epochs=num_epochs
    )

def precompute_null_distribution_nf_classifiers(
		rng_key,
		calibration_data: Tuple[jnp.ndarray, jnp.ndarray], # D_cal = (Theta_cal, X_cal)
		classifiers: List[BinaryMLPlassifier],
        latent_dim: int,
		num_epochs: int = 100
	):
    """
    ℓ-C2ST-NF – precompute the null distribution for any estimator.
    Input:
        rng_key: JAX PRNG key.
        calibration_data: Tuple of (Theta_n, X_n) sampled from p(theta, x).
        classifiers: List of classifiers.
        num_epochs: Number of epochs for each null classifier.
    Output:
        null_classifier_params_list: List of trained parameters for NH null classifiers.
    """
    _ , x_cal = calibration_data
    N_cal = x_cal.shape[0]
    m = latent_dim

    for classifier in classifiers:
        rng_key, z_c0_key, z_c1_key = jr.split(rng_key, 3)

        # Construct classification training set under the Null [2]
        # Both classes (C=0 and C=1) now draw their latent samples (Z) from N(0, Im)
        z_c0_null_samples = jr.normal(z_c0_key, shape=(N_cal, m))
        x_c0_null_samples = x_cal # X_n from D_cal

        z_c1_null_samples = jr.normal(z_c1_key, shape=(N_cal, m))
        x_c1_null_samples = x_cal # X_n from D_cal

        z = jnp.concatenate([z_c0_null_samples, z_c1_null_samples], axis=0)
        x = jnp.concatenate([x_c0_null_samples, x_c1_null_samples], axis=0)
        labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)

        fit_classifier(
            rng_key,
            classifier, 
            x,
            z,
            labels,
            num_epochs=num_epochs
        )


def _ce_loss(model: BinaryMLPlassifier, batch: PyTree) -> jnp.ndarray:
    """Calculates binary cross-entropy loss for the classifier."""
    preds = model(batch['z'], batch['x'])
    labels = batch['y'].astype(jnp.float32) # Ensure shape matches
    preds = jnp.clip(preds, 1e-7, 1 - 1e-7) # Clip to avoid log(0)
    loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))
    return loss

def fit_classifier(
        seed: jnp.ndarray,
        classifier: BinaryMLPlassifier, 
        x: jnp.ndarray,
        z: jnp.ndarray,
        labels: jnp.ndarray,
        num_epochs=100,
        batch_size=100,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
    ):

    # make iterator from z and x
    iter_key, seed = jr.split(seed)
    train_iter, val_iter = as_batch_iterators(
        iter_key,
        {"x": x, "z": z, "y": labels},
        batch_size,
        0.8,
        shuffle=True
    )
    fit_model(
        seed,
        classifier,
        loss_fn=_ce_loss,
        train_iter=train_iter,
        val_iter=val_iter,
        optimizer=optimizer,
        n_iter=num_epochs,
        n_early_stopping_patience=5,
        n_early_stopping_delta=0.001
    )


#TODO: sanitize
def evaluate_l_c2st_nf(
    rng_key: jax.random.PRNGKey,
    xo: jnp.ndarray, # The specific observation for local evaluation
    trained_main_classifier_params: hk.Params,
    null_classifier_params_list: List[hk.Params],
    q_phi_params: hk.Params, # Parameters for your pre-trained NF (for latent_dim)
    classifier_forward_fn: hk.Transformed,
    Nv: int # Number of validation samples
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Adapts Algorithm 2 for ℓ-C2ST-NF, evaluating test statistics and p-values for a given xo.
    Input:
        rng_key: JAX PRNG key.
        xo: The specific observation (x_0) to evaluate consistency at.
        trained_main_classifier_params: Parameters of the main classifier (from Algorithm 3).
        null_classifier_params_list: List of parameters for null classifiers (from Algorithm 4).
        q_phi_params: Parameters of the pre-trained Normalizing Flow (used here for latent_dim).
        classifier_forward_fn: Haiku transformed classifier function.
        Nv: Number of samples for validation.
    Output:
        t_mse0_val: The calculated MSE test statistic for xo.
        p_value: The p-value for the local consistency at xo.
    """
    m = q_phi_params['latent_dim'] # Latent dimension of the NF
    d = xo.shape # Dimension of the observation x

    # Create a batched version of the classifier's apply function for efficiency
    # This maps over the Nv samples (z_samples) and repeats xo for each.
    batched_classifier_apply = jax.vmap(classifier_forward_fn.apply, in_axes=(None, None, 0, None))

    # 1. Generate Samples and Predicted Probabilities [10]
    # For ℓ-C2ST-NF evaluation, the classifier is evaluated on Nv samples from N(0, Im) [3].
    rng_key, z_eval_key = jax.random.split(rng_key)
    z_eval_samples = jax.random.normal(z_eval_key, shape=(Nv, m))
    
    # xo needs to be broadcast or repeated for each z_eval_sample
    # If xo is (d,), make it (1, d) then vmap will handle it if in_axes for x_input is None
    xo_expanded = jnp.expand_dims(xo, axis=0) # (1, d)

    # Compute d(Z_n, xo) for the main classifier
    d_preds_main = batched_classifier_apply(trained_main_classifier_params, None, z_eval_samples, xo_expanded)
    d_preds_main = jnp.squeeze(d_preds_main) # Remove singleton dimension if output_dim was 1

    # 2. Compute Test Statistics [10]
    # t̂MSE0(xo) = (1/Nv) * sum((d(Z_n, xo) - 1/2)^2)
    t_mse0_val = jnp.mean((d_preds_main - 0.5)**2)

    null_test_statistics = []
    for null_params in null_classifier_params_list:
        d_preds_null = batched_classifier_apply(null_params, None, z_eval_samples, xo_expanded)
        d_preds_null = jnp.squeeze(d_preds_null)
        null_test_statistics.append(jnp.mean((d_preds_null - 0.5)**2))

    null_test_statistics = jnp.array(null_test_statistics)

    # 3. Compute p-value [10]
    # p̂(xo) = (1/NH) * sum(I(t̂h(xo) > t̂MSE0(xo)))
    p_value = jnp.mean(null_test_statistics >= t_mse0_val)
    # Add a small value to the denominator to prevent p_value from being 0 if all nulls are smaller
    # This also helps to ensure the lowest p-value is 1/(NH+1) rather than 0.
    p_value = jnp.maximum(p_value, 1.0 / (len(null_classifier_params_list) + 1))

    return t_mse0_val, p_value
