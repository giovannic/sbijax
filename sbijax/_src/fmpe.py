import numpy as np
import optax
from jax import tree
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src._ne_base import NE
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.early_stopping import EarlyStopping
from sbijax._src.util.types import PyTree

from .util.dataloader import (
    structured_as_batch_iterators,
    encode_context,
    encode_unknown_theta,
    decode_theta,
    PAD_VALUE
)

from flax import nnx

def _sample_theta_t(rng_key, times, theta, sigma_min):
    mus = times * theta
    sigma= 1.0 - (1.0 - sigma_min) * times

    noise = jr.normal(rng_key, shape=theta.shape)
    theta_t = noise * sigma + mus
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
    n = theta.shape[0]

    t_key, rng_key = jr.split(rng_key)
    times = jr.uniform(t_key, shape=(n, 1, 1)) # sample, token, time

    theta_key, rng_key = jr.split(rng_key)
    theta_t = _sample_theta_t(
        theta_key,
        times,
        theta,
        sigma_min
    )

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
        theta_batch_shapes,
        **kwargs
        ):
        super().__init__(
            model_fns,
            density_estimator,
            **kwargs
        )
        self.theta_batch_shapes = theta_batch_shapes

    def fit(
        self,
        rng_key,
        data: PyTree,
        *,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
        n_iter: int = 1000,
        batch_size: int = 100,
        percentage_data_as_validation_set: float = 0.1,
        n_early_stopping_patience: int = 10,
        n_early_stopping_delta: float = 0.001,
        data_batch_ndims = None
    ):
        """Fit the model.

        Args:
            rng_key: a jax random key
            data: data set obtained from calling
                `simulate_data_and_possibly_append`
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
        (
            train_iter,
            val_iter,
            labels
        ) = structured_as_batch_iterators(
            itr_key,
            data,
            batch_size,
            percentage_data_as_validation_set,
            True,
            data_batch_ndims=data_batch_ndims
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
        def body_fn(model, batch_key, batch):
            loss = _cfm_loss(model, batch_key, batch, labels)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        loss = 0.0
        for batch in val_iter:
            val_key, rng_key = jr.split(rng_key)
            loss += body_fn(self.model, val_key, batch)
        return loss

    def sample_posterior(
        self,
        rngs,
        observable,
        theta_index,
        context_index,
        *,
        n_samples=4_000,
        sample_ndims=1,
        batch_ndims=None,
        pad_value=PAD_VALUE,
        context_index_map=None,
        theta_index_map=None,
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
        if batch_ndims is None:
            context_batch_ndims = None
            theta_batch_ndims = None
        else:
            context_batch_ndims = { 'y': batch_ndims['y'] }
            theta_batch_ndims = { 'theta': batch_ndims['theta'] }

        context, context_label = encode_context(
            {
                'y': observable,
                'y_index': context_index
            },
            data_sample_ndims=1,
            data_batch_ndims=context_batch_ndims,
            pad_value=pad_value,
            index_map=context_index_map
        )

        # calculate event shapes from index
        event_shapes = tree.map(
            lambda x: x.shape[sample_ndims:-1],
            theta_index
        )

        flat_index, theta_label = encode_unknown_theta(
            { 'theta_index': theta_index },
            data_sample_ndims=sample_ndims,
            data_batch_ndims=theta_batch_ndims,
            pad_value=pad_value,
            index_map=theta_index_map
        )

        self.model.eval()
        thetas = nnx.jit(self.model.sample, static_argnames=['sample_size'])(
            rngs,
            sample_size=n_samples,
            context=context['y'],
            context_label=context_label,
            context_index=context['y_index'],
            theta_label=theta_label,
            theta_index=flat_index
        )
        thetas = decode_theta(
            theta=thetas,
            labels=theta_index.keys(),
            sample_shape=(n_samples,),
            event_shapes=event_shapes,
            batch_shapes=self.theta_batch_shapes
        )
        proposal_probs = self.prior_log_density_fn(
            thetas,
            index=theta_index
        )
        ess = n_samples / jnp.sum(
            jnp.isfinite(proposal_probs)
        )
        inference_data = as_inference_data(thetas, observable)
        return inference_data, ess
