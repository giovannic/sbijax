import optax
from jax import jit
from jax import numpy as jnp
from jax import random as jr

from ._ne_base import NE

from .train import fit_model, fit_model_no_branch

from flax import nnx

Array = jnp.ndarray


def _theta_t_gauss(theta_0, times, theta, sigma_min):
    return times * theta  + jnp.square(1.0 - times) * theta_0

def _theta_t_linear(theta_0, times, theta, sigma_min):
    sigma = 1.0 - (1.0 - sigma_min) * times
    return theta_0 * sigma + theta * times

def _ut_gauss(theta_t, theta, times, sigma_min):
    return (theta - theta_t) / (1. - times)

def _ut_linear(theta_t, theta, times, sigma_min):
    num = theta - (1.0 - sigma_min) * theta_t
    denom = 1.0 - (1.0 - sigma_min) * times
    return num / denom

def _cfm_loss(
    model,
    rng_key,
    batch,
    sigma_min=1e-5,
):
    theta = batch["data"]["theta"]
    n = theta.shape[0]

    t_key, rng_key = jr.split(rng_key)
    times = jr.uniform(t_key, shape=(n, 1))
    theta_key, rng_key = jr.split(rng_key)
    theta_0 = jr.normal(theta_key, shape=theta.shape)

    theta_t = _theta_t_linear(
        theta_0,
        times,
        theta,
        sigma_min
    )

    vs = model.vector_field(
        theta=theta_t,
        time=times,
        context=batch["data"]["y"]
    )
    us = _ut_linear(
        theta_t,
        theta,
        times,
        sigma_min
    )

    return jnp.mean(jnp.square(vs - us))

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

        Returns:
            a tuple of parameters and a tuple of the training information
        """

        fit_model(
            rng_key,
            self.model,
            _cfm_loss,
            train_iter,
            val_iter,
            optimizer,
            n_iter,
            n_early_stopping_patience,
            n_early_stopping_delta,
        )

    def fit_fast(
        self,
        rng_key: Array,
        train: Array,
        val: Array,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
        n_iter: int = 1000,
        batch_size: int = 100
    ):
        """Fit the model.

        Returns:
            a tuple of parameters and a tuple of the training information
        """

        fit_model_no_branch(
            self.model,
            rng_key,
            _cfm_loss,
            train,
            val,
            optimizer,
            n_iter,
            batch_size
        )




    def sample_posterior( #type: ignore
        self,
        rng_key,
        context,
        theta_shape,
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
                theta_shape=theta_shape,
                sample_size=n_samples,
                theta_0=theta_0,
                direction=direction
            )

            return res

        graphdef, state = nnx.split(self.model)
        thetas = _sample_theta(graphdef, state)
        return thetas
