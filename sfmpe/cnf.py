from jax import numpy as jnp, random, vmap
from jax.experimental.ode import odeint
from flax import nnx
from .nn.mlp import VectorFieldModel
from jaxtyping import Array
from typing import Optional

class CNF(nnx.Module):

    def __init__(self, vfm: VectorFieldModel, rngs=nnx.Rngs(0)):
        self.vfm = vfm
        self.rngs = rngs

    def sample(
        self,
        context,
        theta_shape,
        sample_size=1,
        theta_0=None,
        direction='forward',
        return_times: Optional[Array] = None,
        ) -> jnp.ndarray:
        if theta_0 is None:
            theta_0 = random.normal(
                self.rngs.base_dist(),
                (sample_size,) + theta_shape
            )
        else:
            assert theta_0.shape == (sample_size,) + theta_shape, \
                    f'Expected theta_0 to have shape {(sample_size,) + theta_shape} but got {theta_0.shape}'


        if direction == 'forward':
            if return_times is not None:
                t = return_times
            else:
                t = jnp.array([0.0, 1.0])
            vector_sign = 1.
        elif direction == 'backward':
            vector_sign = -1.
            if return_times is not None:
                t = vector_sign * return_times[::-1]
            else:
                t = vector_sign * jnp.array([1.0, 0.0])
        else:
            raise ValueError(f'Unknown direction: {direction}')

        # NOTE: Take this up with nnx. @nnx.scan (in mlp) changes the state and there's no way to make it pure
        graphdef, state = nnx.split(self)

        @vmap
        def solve(theta_0):
            def ode_func(theta_t, time):
                model = nnx.merge(graphdef, state)
                theta_t = theta_t.reshape((1,) + theta_shape)
                time = vector_sign * jnp.full((1, 1), time)
                ret = vector_sign * model.vector_field(
                    theta=theta_t,
                    time=time,
                    context=context
                )
                return ret.reshape(-1)
                
            return odeint(
                ode_func,
                theta_0,
                t,
                rtol=1e-5,
                atol=1e-5
            )

        res = solve(theta_0.reshape((sample_size, -1)))

        if return_times is not None:
            # res shape: (sample_size, n_times, flattened_theta)
            # Return all time points, shape: (sample_size, n_times, *theta_shape)
            return res.reshape((sample_size, len(return_times)) + theta_shape)
        else:
            # Return only final point
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
        return self.vfm(
            theta=theta,
            time=time,
            context=context,
        )
