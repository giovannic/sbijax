from jax import numpy as jnp, random, vmap, vjp
from jax.experimental.ode import odeint
from jax.scipy.stats import norm
from flax import nnx
from .svf import StructuredVectorFieldModel

class StructuredCNF(nnx.Module):
    """
    A structured CNF:
    Provides an interface for sampling and producing vector fields
    for models which require labels, indices and masks

    """

    def __init__(self, vf_model: StructuredVectorFieldModel, rngs=nnx.Rngs(0)):
        super().__init__()
        self.vf_model = vf_model
        self.rngs = rngs

    def sample(
        self,
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
                self.rngs.base_dist(),
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

    def log_prob(
        self,
        theta,
        theta_label,
        theta_index,
        theta_mask,
        context,
        context_label,
        context_index,
        context_mask,
        cross_mask,
        n_epsilon=10,
        ):
        """
        Compute the log probability of theta using the FFJORD algorithm.

        Implementation of Algorithm 1 from Grathwohl and Chen 2019:
        "FFJORD: Free-form Continuous Dynamics for Scalable Reversible
        Generative Models"

        Uses unbiased stochastic log-density estimation by augmenting the ODE
        with log-density dynamics computed via vector-Jacobian products.

        Parameters
        ----------
        theta : jnp.ndarray
            Target samples to evaluate density, shape (batch_size, *theta_shape)
        theta_label : jnp.ndarray
            Labels for theta
        theta_index : jnp.ndarray or None
            Indices for theta
        theta_mask : jnp.ndarray or None
            Mask for theta
        context : jnp.ndarray
            Context/conditioning variables
        context_label : jnp.ndarray
            Labels for context
        context_index : jnp.ndarray or None
            Indices for context
        context_mask : jnp.ndarray or None
            Mask for context
        cross_mask : jnp.ndarray or None
            Cross attention mask
        n_epsilon : int, optional
            Number of epsilon samples to average for stochastic trace estimation.
            Higher values reduce variance but increase computation. Default: 10.

        Returns
        -------
        jnp.ndarray
            Log probabilities for each sample in theta, shape (batch_size,)
        """
        batch_size = theta.shape[0]
        theta_shape = theta.shape[1:]
        flat_size = int(jnp.prod(jnp.array(theta_shape)))

        # Sample epsilon for stochastic trace estimation
        # Shape: (batch_size, *theta_shape, n_epsilon)
        epsilon = random.normal(
            self.rngs.base_dist(),
            theta.shape + (n_epsilon,)
        )

        # Integration from t=1.0 to t=0.0 (backward)
        # Use negative time points for backward integration
        vector_sign = -1.0
        t = vector_sign * jnp.array([1.0, 0.0])

        def augmented_ode_func(state, time, epsilon_sample):
            """
            Augmented dynamics function f_aug([z_t, log_p_t], t).

            Computes both the vector field dynamics and the log-density
            change using stochastic trace estimation.
            """
            # Unpack state: [theta_t (flattened), log_p_t]
            theta_t_flat = state[:flat_size]

            # Reshape theta_t back to original shape
            theta_t = theta_t_flat.reshape((1,) + theta_shape)

            time_broadcast = jnp.full((1, 1, 1), time)

            # Define vector field function for vjp
            def vf_func(theta_input):
                return vector_sign * self.vector_field(
                    theta=theta_input,
                    theta_label=theta_label,
                    theta_index=theta_index,
                    theta_mask=theta_mask,
                    time=time_broadcast,
                    context=context,
                    context_label=context_label,
                    context_index=context_index,
                    context_mask=context_mask,
                    cross_mask=cross_mask,
                )

            # Compute vector field and vjp
            f_t, vjp_fn = vjp(vf_func, theta_t)

            # Compute vector-Jacobian product for all epsilon samples
            # epsilon_sample shape: (*theta_shape, n_epsilon)
            # Vectorize over the epsilon dimension
            def compute_trace_single_eps(eps):
                epsilon_batch = eps.reshape((1,) + theta_shape)
                g = vjp_fn(epsilon_batch)[0]
                return jnp.sum(g * epsilon_batch)

            # Vectorize over n_epsilon dimension
            trace_estimates = vmap(compute_trace_single_eps, in_axes=-1)(epsilon_sample)
            # Average over all epsilon samples
            trace_estimate = jnp.mean(trace_estimates)

            # Return augmented dynamics: [f_t, -trace_estimate]
            f_t_flat = f_t.reshape(-1)
            return jnp.concatenate([f_t_flat, jnp.array([-trace_estimate])])

        def solve_single(theta_sample, epsilon_sample):
            """Solve ODE for a single sample."""
            # Initial state: [theta_flat, 0.0]
            theta_flat = theta_sample.reshape(-1)
            initial_state = jnp.concatenate([theta_flat, jnp.array([0.0])])

            # Create ODE function with epsilon_sample bound
            def ode_func_with_epsilon(state, time, *_):
                return augmented_ode_func(state, time, epsilon_sample)

            # Solve augmented ODE
            solution = odeint(
                ode_func_with_epsilon,
                initial_state,
                t,
                rtol=1e-5,
                atol=1e-5
            )

            # Extract final state
            final_state = solution[-1]
            z_final_flat = final_state[:flat_size]
            delta_log_p = final_state[flat_size]

            # Reshape z_final
            z_final = z_final_flat.reshape(theta_shape)

            # Compute log probability of base distribution (standard normal)
            log_p_z0 = jnp.sum(norm.logpdf(z_final))

            # Final log probability: log p(z_0) - Δ_log_p
            return log_p_z0 - delta_log_p

        # Vectorize over batch
        log_probs = vmap(solve_single)(theta, epsilon)

        return log_probs

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
