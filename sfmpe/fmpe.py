import optax
from jax import (
    jit,
    numpy as jnp,
    random as jr,
    vmap
)

from .train import fit_model_no_branch
from .cnf import CNF

from flax import nnx
from jaxtyping import Array
from typing import Tuple

def theta_t_linear(theta_0, times, theta, sigma_min):
    sigma = 1.0 - (1.0 - sigma_min) * times
    return theta_0 * sigma + theta * times

def ut_linear(theta_t, theta, times, sigma_min):
    return theta - (1.0 - sigma_min) * theta_t

def _cfm_loss(
    model,
    rng_key,
    batch,
    sigma_min = 1e-5,
):
    theta = batch["data"]["theta"]
    n = theta.shape[0]

    t_key, rng_key = jr.split(rng_key)
    times = jr.uniform(t_key, shape=(n, 1))
    theta_key, rng_key = jr.split(rng_key)
    theta_0 = jr.normal(theta_key, theta.shape)

    theta_t = theta_t_linear(
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
    us = ut_linear(
        theta_t,
        theta,
        times,
        sigma_min
    )

    return jnp.mean(jnp.square(vs - us))

class FMPE:

    def __init__(self, model: CNF, rngs=nnx.Rngs(0)):
        self.model = model
        self.rngs = rngs

    def fit(
        self,
        train: Array,
        val: Array,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
        n_iter: int = 1000,
        batch_size: int = 100
    ) -> Tuple[Array, Array]:
        """Fit the model.

        Returns:
            a tuple of loss values
        """

        return fit_model_no_branch(
            self.model,
            self.rngs.permutations(),
            _cfm_loss,
            train,
            val,
            optimizer,
            n_iter,
            batch_size
        )

    def sample_posterior(
        self,
        context,
        theta_shape,
        n_samples=1_000,
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

    def sample_base_dist(
        self,
        theta: Array,
        context: Array,
        theta_shape: Tuple
        ):
        def sample_pair(theta, y):
            return self.sample_posterior(
                y[None, ...],
                theta_shape = theta_shape,
                n_samples=1,
                theta_0=theta[None, ...],
                direction='backward'
            )

        z = vmap(sample_pair)(theta, context)
        return z[..., 0, :]
