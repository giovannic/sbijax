from jax import numpy as jnp, random, vmap
from jax.experimental.ode import odeint
from flax import nnx

__all__ = ["CNF"]

# ruff: noqa: PLR0913,D417
class CNF(nnx.Module):
    """Conditional continuous normalizing flow.

    Args:
        n_dimension: the dimensionality of the modelled space
        transform: a haiku module. The transform is a callable that has to
            take as input arguments named 'theta', 'time', 'context' and
            **kwargs. Theta, time and context are two-dimensional arrays
            with the same batch dimensions.
    """

    def __init__(self, theta_shape, transform: nnx.Module):
        """Conditional continuous normalizing flow.

        Args:
            transform: an nnx module. The transform is a callable that has to
                take as input arguments named 'theta', 'time', 'context' and
                **kwargs. Theta, time and context are two-dimensional arrays
                with the same batch dimensions.
        """
        super().__init__()
        self._theta_shape= theta_shape
        self._network = transform

    def sample(
        self,
        rngs,
        context,
        context_label,
        context_index,
        theta_label,
        theta_index,
        sample_size=1
        ):
        """Sample from the pushforward.

        Args:
            context: array of conditioning variables
        """
        theta_0= random.normal(
            rngs.base_dist(),
            (sample_size,) + self._theta_shape
        )

        def ode_func(theta_t, time, *_):
            theta_t = theta_t.reshape((1,) + theta_0.shape[1:]) # sample x token x feature
            time = jnp.full((1, 1, 1), time)
            ret = self.vector_field(
                theta=theta_t,
                theta_label=theta_label,
                theta_index=theta_index,
                time=time,
                context=context,
                context_label=context_label,
                context_index=context_index
            )
            return ret.reshape(-1)

        # vmap odeint over sample shape
        def solve(theta_0):
            return odeint(
                ode_func,
                theta_0,
                jnp.array([0.0, 1.0]),
                rtol=1e-5,
                atol=1e-5
            )

        res = vmap(solve)(
            theta_0.reshape((sample_size, -1))
        )

        return res[:, -1].reshape((sample_size,) + self._theta_shape)

    def vector_field(
        self,
        theta,
        theta_label,
        theta_index,
        time,
        context,
        context_label,
        context_index
        ):
        """Compute the vector field.

        Args:
            theta: array of parameters
            time: time variables
            context: array of conditioning variables

        Keyword Args:
            keyword arguments that are passed to the neural network
        """
        return self._network( #type: ignore
            theta=theta,
            theta_label=theta_label,
            theta_index=theta_index,
            time=time,
            context=context,
            context_label=context_label,
            context_index=context_index,
        )
