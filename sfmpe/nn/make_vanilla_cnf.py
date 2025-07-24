from jax import numpy as jnp, random, vmap
from jax.experimental.ode import odeint
from flax import nnx
from .vanilla_mlp import VectorFieldModel

class VanillaCNF(nnx.Module):
    """Conditional continuous normalizing flow."""

    def __init__(self, transform: VectorFieldModel):
        """Conditional continuous normalizing flow."""
        super().__init__()
        self._network = transform

    def sample(
        self,
        rngs,
        context,
        theta_shape,
        sample_size=1,
        theta_0=None,
        direction='forward',
        ) -> jnp.ndarray:
        if theta_0 is None:
            theta_0 = random.normal(
                rngs.base_dist(),
                (sample_size,) + theta_shape
            )
        else:
            assert theta_0.shape == (sample_size,) + theta_shape, f'Expected theta_0 to have shape {(sample_size,) + theta_shape} but got {theta_0.shape}'


        if direction == 'forward':
            t = jnp.array([0.0, 1.0])
            vector_sign = 1.
        elif direction == 'backward':
            vector_sign = -1.
            t = vector_sign * jnp.array([1.0, 0.0])
        else:
            raise ValueError(f'Unknown direction: {direction}')

        def ode_func(theta_t, time, *_):
            theta_t = theta_t.reshape((1,) + theta_0.shape[1:]) # sample x token x feature
            time = vector_sign * jnp.full((1, 1), time)
            ret = vector_sign * self.vector_field(
                theta=theta_t,
                time=time,
                context=context
            )
            return ret.reshape(-1)

        # vmap odeint over sample shape
        # NOTE: how can this happen?
        def solve(theta_0):
            return odeint(
                ode_func,
                theta_0,
                t,
                rtol=1e-5,
                atol=1e-5
            )

        res = vmap(solve)(
            theta_0.reshape((sample_size, -1))
        )

        return res[:, -1].reshape((sample_size,) + theta_shape)

    def vector_field(
        self,
        theta,
        time,
        context,
        ) -> jnp.ndarray:
        """Compute the vector field.

        Args:
            theta: array of parameters
            time: time variables
            context: array of conditioning variables

        Keyword Args:
            keyword arguments that are passed to the neural network
        """
        return self._network(
            theta=theta,
            time=time,
            context=context
        )
