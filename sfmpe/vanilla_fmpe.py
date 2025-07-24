import optax
from jax import jit
from jax import numpy as jnp
from jax import random as jr

from ._ne_base import NE

from .train import fit_model

from flax import nnx

def _sample_theta_t(rng_key, times, theta, sigma_min):
    mus = times * theta
    sigma= 1.0 - (1.0 - sigma_min) * times

    noise = jr.normal(rng_key, shape=theta.shape)
    theta_t = noise * sigma + mus
    return theta_t

def _ut(theta_t, theta, times, sigma_min):
    num = theta - (1.0 - sigma_min) * theta_t
    denom = 1.0 - (1.0 - sigma_min) * times
    return num / denom

def _cfm_loss(
    model,
    rng_key,
    batch,
    sigma_min=0.001,
):
    theta = batch["data"]["theta"]
    n = theta.shape[0]

    t_key, rng_key = jr.split(rng_key)
    times = jr.uniform(t_key, shape=(n,))

    theta_key, rng_key = jr.split(rng_key)
    theta_t = _sample_theta_t(
        theta_key,
        times,
        theta,
        sigma_min
    )

    vs = model.vector_field(
        theta=theta_t,
        time=times,
        context=batch["data"]["y"]
    )
    uts = _ut(theta_t, theta, times, sigma_min)

    return jnp.mean(jnp.square(vs - uts))

class FMPE(NE):

    def __init__(
        self,
        density_estimator,
        **kwargs
        ):
        super().__init__(
            density_estimator,
            **kwargs
        )

    def fit(
        self,
        rng_key,
        train_iter,
        val_iter,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
        n_iter: int = 1000,
        n_early_stopping_patience: int = 10,
        n_early_stopping_delta: float = 0.001,
    ):
        """Fit the model.

        Args:
            rng_key: a jax random key
            data: data set for training, a dictionary with `y`, `y_index`, `theta`, `theta_index`.
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size:  batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated
                data that is used for validation and early stopping
            n_early_stopping_patience: number of iterations of no improvement
                of training the flow before stopping optimisation
        Returns:
            a tuple of parameters and a tuple of the training information
        """

        losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
            n_early_stopping_delta=n_early_stopping_delta,
        )

        return losses

    def _fit_model_single_round(
        self,
        seed,
        train_iter,
        val_iter,
        optimizer,
        n_iter,
        n_early_stopping_patience,
        n_early_stopping_delta,
    ):
        fit_model(
            seed,
            self.model,
            _cfm_loss,
            train_iter,
            val_iter,
            optimizer,
            n_iter,
            n_early_stopping_patience,
            n_early_stopping_delta,
        )

    def sample_posterior( #type: ignore
        self,
        rng_key,
        context,
        n_samples=4_000,
        theta_0=None,
        direction='forward'
    ):
        self.model.eval()

        # NOTE: nnx.jit somehow leaks tracers. Need to investigate
        @jit
        def _sample_theta(
            graphdef,
            state
            ):
            model = nnx.merge(graphdef, state)
            res = model.sample(
                rngs=nnx.Rngs(rng_key),
                context=context,
                sample_size=n_samples,
                theta_0=theta_0,
                direction=direction
            )

            return res

        graphdef, state = nnx.split(self.model)
        thetas = _sample_theta(graphdef, state)
        return thetas
