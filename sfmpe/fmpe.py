import numpy as np
import optax
from jax import tree, jit
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from ._ne_base import NE
from .util.data import as_inference_data
from .util.early_stopping import EarlyStopping
from .util.types import PyTree

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

class SFMPE(NE):

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
        theta_event_sizes = None
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
            labels,
            mask
        ) = structured_as_batch_iterators(
            itr_key,
            data,
            batch_size,
            percentage_data_as_validation_set,
            True,
            data_batch_ndims=data_batch_ndims,
            event_sizes=theta_event_sizes
        )

        losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
            n_early_stopping_delta=n_early_stopping_delta,
            labels=labels,
            mask=mask
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
        key,
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
        thetas, diagnostics = self.sample_structured_posterior(
            key,
            observable,
            theta_index,
            context_index,
            n_samples=n_samples,
            sample_ndims=sample_ndims,
            batch_ndims=batch_ndims,
            pad_value=pad_value,
            context_index_map=context_index_map,
            theta_index_map=theta_index_map
        )

        inference_data = as_inference_data(thetas, observable)
        return inference_data, diagnostics

    def sample_structured_posterior(
        self,
        rng_key,
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

        # NOTE: nnx.jit somehow leaks tracers. Need to investigate
        @jit
        def _sample_theta(
            graphdef,
            state
            ):
            model = nnx.merge(graphdef, state)
            res = model.sample(
                rngs=nnx.Rngs(rng_key),
                context=context['y'],
                context_label=context_label,
                context_index=context['y_index'],
                theta_label=theta_label,
                theta_index=flat_index,
                sample_size=n_samples,
            )

            return res

        graphdef, state = nnx.split(self.model)

        thetas = _sample_theta(
            graphdef,
            state
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
        return thetas, ess

    def simulate_parameters(
        self,
        rng_key,
        *,
        first_round=True,
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
        if first_round:
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
            new_thetas, diagnostics = self.sample_structured_posterior(
                rng_key=rng_key,
                observable=observable,
                context_index=context_index,
                theta_index=theta_index,
                **kwargs,
            )
        return (new_thetas, theta_index), diagnostics
