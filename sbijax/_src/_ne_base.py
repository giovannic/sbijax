import abc
from abc import ABC

import chex
from jax import numpy as jnp
from jax import random as jr

from sbijax._src._sbi_base import SBI
from sbijax._src.util.data import inference_data_as_dictionary as flatten
from sbijax._src.util.data import stack_data


# ruff: noqa: PLR0913
class NE(SBI, ABC):
    """Sequential neural estimation base class."""

    def __init__(self, model_fns, network, sample_index=None, index_shape=()):
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
        self._sample_index = sample_index
        self._index_shape = index_shape

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
        observable = jnp.atleast_2d(observable)
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
        ):
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
        observed_index=None,
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
            if observed_index is None:
                index = observed_index
            else:
                index_key, rng_key = jr.split(rng_key)
                index = self.sample_index(index_key, (n_simulations,))
            new_thetas = self.prior_sampler_fn(
                index=index,
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
                observable=jnp.atleast_2d(observable),
                observed_index=observed_index,
                **kwargs,
            )
            new_thetas = flatten(inference_data.posterior)
            perm_key, rng_key = jr.split(rng_key)
            first_key = list(new_thetas.keys())[0]
            idxs = jr.choice(
                perm_key,
                new_thetas[first_key].shape[0],
                shape=(n_simulations,),
                replace=False,
            )
            new_thetas = {k: v[idxs] for k, v in new_thetas.items()}

        return new_thetas, diagnostics

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

        new_thetas, diagnostics = self.simulate_parameters(
            theta_key,
            params=params,
            observable=observable,
            n_simulations=n_simulations,
            **kwargs,
        )

        new_obs = self.simulate_observations(data_key, new_thetas)
        for v in new_thetas.values():
            chex.assert_shape(v, [n_simulations, None])
        chex.assert_shape(new_obs, [n_simulations, None])
        new_data = {"y": new_obs, "theta": new_thetas}

        return new_data, diagnostics

    def simulate_observations(self, rng_key, thetas):
        # TODO: support infinite dimensional observations
        new_obs = self.simulator_fn(
            seed=rng_key,
            theta=thetas,
            **self.indices
        )
        return new_obs

    def sample_index(self, key, batch_shape):
        assert self._sample_index is not None
        index_shape = batch_shape + self._index_shape
        return self._sample_index(key, index_shape)
