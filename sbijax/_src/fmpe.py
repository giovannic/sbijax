from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree
from tqdm import tqdm

from sbijax._src._ne_base import NE
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.early_stopping import EarlyStopping
from sbijax._src.util.types import PyTree

from flax import nnx

def _sample_theta_t(rng_key, times, theta, sigma_min):
    mus = times * theta
    sigmata = 1.0 - (1.0 - sigma_min) * times
    sigmata = sigmata.reshape(times.shape[0], 1)

    noise = jr.normal(rng_key, shape=(*theta.shape,))
    theta_t = noise * sigmata + mus
    return theta_t


def _ut(theta_t, theta, times, sigma_min):
    num = theta - (1.0 - sigma_min) * theta_t
    denom = 1.0 - (1.0 - sigma_min) * times
    return num / denom

def _cfm_loss(
    model,
    rng_key,
    batch,
    labels,
    sigma_min=0.001,
):
    theta = batch["theta"]
    n, _ = theta.shape

    t_key, rng_key = jr.split(rng_key)
    times = jr.uniform(t_key, shape=(n, 1))

    theta_key, rng_key = jr.split(rng_key)
    theta_t = _sample_theta_t(theta_key, times, theta, sigma_min)

    vs = model.vector_field(
        theta=theta_t,
        theta_label=labels["theta"],
        theta_index=batch["theta_index"],
        time=times,
        context=batch["y"],
        context_label=labels["y"],
        context_index=batch["y_index"],
    )
    uts = _ut(theta_t, theta, times, sigma_min)

    loss = jnp.mean(jnp.square(vs - uts))
    return loss

class FMPE(NE):
    r"""Flow matching posterior estimation.

    Implements the FMPE algorithm introduced in :cite:t:`wilderberger2023flow`.

    Args:
        model_fns: a tuple of calalbles. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        density_estimator: a continuous normalizing flow model

    Examples:
        >>> from sbijax import FMPE
        >>> from sbijax.nn import make_cnf
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...     dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_cnf(1)
        >>> model = FMPE(fns, neural_network)

    References:
        Wildberger, Jonas, et al. "Flow Matching for Scalable Simulation-Based Inference." Advances in Neural Information Processing Systems, 2024.
    """

    def __init__(
        self,
        model_fns,
        density_estimator,
        **kwargs
        ):
        super().__init__(
            model_fns,
            density_estimator,
            **kwargs
        )

    def fit(
        self,
        rng_key,
        data: PyTree,
        labels: PyTree,
        *,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
        n_iter: int = 1000,
        batch_size: int = 100,
        percentage_data_as_validation_set: float = 0.1,
        n_early_stopping_patience: int = 10,
        n_early_stopping_delta: float = 0.001,
    ):
        """Fit the model.

        Args:
            rng_key: a jax random key
            data: data set obtained from calling
                `simulate_data_and_possibly_append`
            labels: labels for each random variable ([u]int)
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
        itr_key, rng_key = jr.split(rng_key)
        train_iter, val_iter = self.as_iterators(
            itr_key, data, batch_size, percentage_data_as_validation_set
        )
        losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
            n_early_stopping_delta=n_early_stopping_delta,
            labels=labels
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
        labels
    ):
        nnx_optimizer = nnx.Optimizer(self.model, optimizer)

        # set model to train
        self.model.train()

        @nnx.jit
        def step(model, rng, optimizer, batch):
            loss, grads = nnx.value_and_grad(_cfm_loss)(
                model,
                rng,
                batch,
                labels
            )
            optimizer.update(grads)
            return loss

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(
            n_early_stopping_delta,
            n_early_stopping_patience
        )
        best_state = nnx.state(self.model)
        best_loss = np.inf
        logging.info("training model")
        for i in tqdm(range(n_iter)):
            train_loss = 0.0
            rng_key = jr.fold_in(seed, i)
            for batch in train_iter:
                train_key, rng_key = jr.split(rng_key)
                batch_loss = step(
                    self.model,
                    train_key,
                    nnx_optimizer,
                    batch
                )
                train_loss += batch_loss * (
                    batch["y"].shape[0] / train_iter.num_samples
                )
            val_key, rng_key = jr.split(rng_key)
            self.model.eval()
            validation_loss = self._validation_loss(
                val_key,
                val_iter,
                labels
            )
            self.model.train()
            losses[i] = jnp.array([train_loss, validation_loss])

            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stopping criterion found")
                break
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = nnx.state(self.model)

        # set the model to the best state
        nnx.update(self.model, best_state)
        losses = jnp.vstack(losses)[: (i + 1), :] #type: ignore
        return losses

    def _validation_loss(self, rng_key, val_iter, labels):

        @nnx.jit
        def body_fn(batch_key, batch):
            loss = _cfm_loss(self.model, batch_key, batch, labels)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        loss = 0.0
        for batch in val_iter:
            val_key, rng_key = jr.split(rng_key)
            loss += body_fn(val_key, batch)
        return loss

    def sample_posterior(
        self,
        rng_key,
        observable,
        *,
        n_samples=4_000,
        observed_index=None,
        **_
    ):
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: observation to condition on
            n_samples: number of samples to draw

        Returns:
            returns an array of samples from the posterior distribution of
            dimension (n_samples \times p)
        """
        observable = jnp.atleast_2d(observable)

        _, unravel_fn = ravel_pytree(
            self.prior_sampler_fn(
                index=observed_index,
                seed=jr.PRNGKey(1)
            )
        )
        sample_key, rng_key = jr.split(rng_key)
        self.model.eval()
        thetas = nnx.jit(self.model.sample)(
            sample_key,
            context=jnp.tile(observable, [n_samples, 1]),
        )

        proposal_probs = self.prior_log_density_fn(
            jax.vmap(unravel_fn)(thetas),
            index=observed_index
        )
        ess = thetas.shape[0] / jnp.sum(
            jnp.isfinite(proposal_probs)
        )
        thetas = jax.tree_map(
            lambda x: x.reshape(1, *x.shape),
            jax.vmap(unravel_fn)(thetas[:n_samples]),
        )
        inference_data = as_inference_data(thetas, jnp.squeeze(observable))
        return inference_data, ess
