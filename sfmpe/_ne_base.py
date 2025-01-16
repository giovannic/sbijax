import abc
from abc import ABC

from jax import random as jr
from typing import Tuple
from jaxtyping import Array
import arviz as az

from ._sbi_base import SBI
from .util.data import inference_data_as_dictionary as flatten
from .util.data import stack_data


# ruff: noqa: PLR0913
class NE(SBI, ABC):
    """Sequential neural estimation base class."""

    def __init__(
        self,
        model_fns,
        network
        ):
        """Construct an SNE object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            network: a neural network
            indices: a dictionary of indices for infinite dimensional
                    priors or observations
        """
        super().__init__(model_fns)
        self.model = network
        self.n_total_simulations = 0

    def simulate_data_and_possibly_append(
        self,
        rng_key,
        params,
        observable,
        data=None,
        n_simulations=1000,
        **kwargs,
    ):
        """Simulate data and paarameters from the prior or posterior and append.

        Args:
            rng_key: a random key
            params: a dictionary of neural network parameters
            observable: an observation
            data: existing data set
            n_simulations: number of newly simulated data
            kwargs: dictionary of ey value pairs passed to `sample_posterior`

        Returns:
            returns a NamedTuple of two axis, y and theta
        """
        new_data, diagnostics = self.simulate_data(
            rng_key,
            params=params,
            observable=observable,
            n_simulations=n_simulations,
            **kwargs,
        )
        if data is None:
            d_new = new_data
        else:
            d_new = stack_data(data, new_data)
        return d_new, diagnostics

    @abc.abstractmethod
    def sample_posterior(
        self,
        rng_key,
        params,
        observable,
        observed_index=None,
        *args,
        **kwargs
        ) -> Tuple[az.InferenceData, Array]:
        """Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: a data point
            *args: argument list
            **kwargs: keyword arguments
        """

    def simulate_parameters(
        self,
        rng_key,
        *,
        params=None,
        observable=None,
        theta_index=None,
        context_index=None,
        n_simulations=1000,
        **kwargs,
    ):
        r"""Simulate parameters from the posterior or prior.

        Args:
            rng_key: a random key
            params:a dictionary of neural network parameters. If None, will
                draw from prior. If parameters given, will draw from amortized
                posterior using 'observable'.
            observable: an observation. Needs to be given if posterior draws
                are desired
            observed_index: (optional) the index at which an observation
                was made
            n_simulations: number of newly simulated data
            kwargs: dictionary of ey value pairs passed to `sample_posterior`

        Returns:
            a NamedTuple of two axis, y and theta
        """
        if params is None or len(params) == 0:
            diagnostics = None
            self.n_total_simulations += n_simulations
            new_thetas = self.prior_sampler_fn(
                index=theta_index,
                seed=rng_key,
                sample_shape=(n_simulations,),
            )
        else:
            if observable is None:
                raise ValueError(
                    "need to have access to 'observable' "
                    "when sampling from posterior"
                )
            if "n_samples" not in kwargs:
                kwargs["n_samples"] = n_simulations
            inference_data, diagnostics = self.sample_posterior(
                rng_key=rng_key,
                params=params,
                observable=observable,
                context_index=context_index,
                theta_index=theta_index,
                **kwargs,
            )
            new_thetas = flatten(inference_data.posterior) #type: ignore
        return (new_thetas, theta_index), diagnostics

    def simulate_data(
        self,
        rng_key,
        *,
        params=None,
        observable=None,
        n_simulations=1000,
        **kwargs,
    ):
        r"""Simulate data from the posterior or prior and append.

        Args:
            rng_key: a random key
            params:a dictionary of neural network parameters. If None, will
                draw from prior. If parameters given, will draw from amortized
                posterior using 'observable;
            observable: an observation. Needs to be gfiven if posterior draws
                are desired
            n_simulations: number of newly simulated data
            kwargs: dictionary of ey value pairs passed to `sample_posterior`

        Returns:
            a NamedTuple of two axis, y and theta
        """
        theta_key, data_key = jr.split(rng_key)

        (new_thetas, theta_index), diagnostics = self.simulate_parameters(
            theta_key,
            params=params,
            observable=observable,
            n_simulations=n_simulations,
            **kwargs,
        )

        context_index = kwargs.get("context_index", {})

        new_obs = self.simulate_observations(
            data_key,
            new_thetas,
            context_index=context_index
        )

        new_data = {
            "y": new_obs,
            'y_index': context_index,
            "theta": new_thetas,
            "theta_index": theta_index,
        }

        return new_data, diagnostics

    def simulate_observations(
        self,
        rng_key,
        thetas,
        context_index=None
        ):
        new_obs = self.simulator_fn(
            seed=rng_key,
            theta=thetas,
            **context_index
        )
        return new_obs
