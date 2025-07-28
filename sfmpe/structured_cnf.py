from jax import numpy as jnp, random, vmap
from jax.experimental.ode import odeint
from flax import nnx
from .svf import StructuredVectorFieldModel

class StructuredCNF(nnx.Module):
    """
    A structured CNF:
    Provides an interface for sampling and producing vector fields
    for models which require labels, indices and masks

    """

    def __init__(self, vf_model: StructuredVectorFieldModel):
        super().__init__()
        self.vf_model = vf_model

    def sample(
        self,
        rngs,
        context,
        context_label,
        context_index,
        context_mask,
        theta_shape,
        theta_label,
        theta_index,
        theta_mask,
        cross_mask,
        sample_size=1,
        theta_0=None,
        direction='forward',
        ):
        if theta_0 is None:
            theta_0 = random.normal(
                rngs.base_dist(),
                (sample_size,) + theta_shape
            )
        else:
            assert theta_0.shape == (sample_size,) + theta_shape, \
                    f'Expected theta_0 to have shape {(sample_size,) + theta_shape} but got {theta_0.shape}'


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
            time = vector_sign * jnp.full((1, 1, 1), time)
            ret = vector_sign * self.vector_field(
                theta=theta_t,
                theta_label=theta_label,
                theta_index=theta_index,
                theta_mask=theta_mask,
                time=time,
                context=context,
                context_label=context_label,
                context_index=context_index,
                context_mask=context_mask,
                cross_mask=cross_mask,
            )
            return ret.reshape(-1)

        # vmap odeint over sample shape
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
        theta_label,
        theta_index,
        theta_mask,
        time,
        context,
        context_label,
        context_index,
        context_mask,
        cross_mask,
        ):
        """Compute the vector field.

        Args:
            theta: array of parameters
            time: time variables
            context: array of conditioning variables

        Keyword Args:
            keyword arguments that are passed to the neural network
        """
        return self.vf_model(
            theta=theta,
            theta_label=theta_label,
            theta_index=theta_index,
            theta_mask=theta_mask,
            time=time,
            context=context,
            context_label=context_label,
            context_index=context_index,
            context_mask=context_mask,
            cross_mask=cross_mask
        )
