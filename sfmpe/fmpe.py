import numpy as np
import optax
from jax import jit
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from ._ne_base import NE
from .util.data import as_inference_data
from .util.early_stopping import EarlyStopping

from .util.dataloader import (decode_theta, prod)

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

def make_self_attention_mask(ind_mask, padding_mask=None):
    mask= jnp.expand_dims(ind_mask, 0) # for batch dim
    if padding_mask is not None:
        mask= mask * padding_mask
    return jnp.expand_dims(mask, 1) # for num attention heads

def make_cross_attention_mask(
    ind_mask,
    theta_padding_mask=None,
    y_padding_mask=None
):
    # extra dim for batch and heads
    cross_mask = jnp.expand_dims(ind_mask, (0, 1))

    if theta_padding_mask is not None:
        # extra heads and y dim
        cross_mask = cross_mask * jnp.expand_dims(theta_padding_mask, (1, 3))
    if y_padding_mask is not None:
        # extra heads and theta dim
        cross_mask = cross_mask * jnp.expand_dims(y_padding_mask, (1, 2))
    return cross_mask

def _cfm_loss(
    model,
    rng_key,
    batch,
    labels,
    attention_masks,
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

    theta_mask = make_self_attention_mask(
        attention_masks['theta'],
        batch.get('theta_padding_mask')
    )

    context_mask = make_self_attention_mask(
        attention_masks['y'],
        batch.get('y_padding_mask')
    )

    cross_mask = make_cross_attention_mask(
        attention_masks['cross'],
        batch.get('theta_padding_mask'),
        batch.get('y_padding_mask')
    )

    vs = model.vector_field(
        theta=theta_t,
        theta_label=labels["theta"],
        theta_index=batch.get("theta_index", None),
        theta_mask=theta_mask,
        time=times,
        context=batch["y"],
        context_label=labels["y"],
        context_index=batch.get("y_index", None),
        context_mask=context_mask,
        cross_mask=cross_mask,
    )
    uts = _ut(theta_t, theta, times, sigma_min)

    # loss has to be masked with the padding mask `batch['theta_mask']`
    # the denominator has to also be derived from the padding mask
    if 'theta_padding_mask' in batch:
        loss = jnp.sum(
            jnp.square(vs - uts) * batch['theta_padding_mask']
        ) / jnp.sum(batch['theta_padding_mask'])
    else:
        loss = jnp.mean(jnp.square(vs - uts))

    return loss

class SFMPE(NE):

    def __init__(
        self,
        density_estimator,
        theta_batch_shapes,
        **kwargs
        ):
        super().__init__(
            density_estimator,
            **kwargs
        )
        self.theta_batch_shapes = theta_batch_shapes

    def fit(
        self,
        rng_key,
        train_iter,
        val_iter,
        labels,
        masks,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
        n_iter: int = 1000,
        n_early_stopping_patience: int = 10,
        n_early_stopping_delta: float = 0.001,
    ):
        """Fit the model.

        Args:
            rng_key: a jax random key
            data: data set for training, a dictionary with `y`, `y_index`, `theta`, `theta_index`.
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size:  batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated
                data that is used for validation and early stopping
            n_early_stopping_patience: number of iterations of no improvement
                of training the flow before stopping optimisation
            data_batch_ndims: PyTree with number of batch dimensions in the data
            event_shapes: shapes of the event dims in the data for attention and loss masking
            independence: dictionary specifying independence in the data, for attention masking. Keys are `local`, `cross` and `cross_local`.
            {
                "local": List[ key, ... ],
                "cross": List[ (key1, key2), ... ],
                "cross_local": List[ (key1, key2, map), ... ],
                # map=None => diagonal mask
                # map=List[ (i,j) ]
            }
        Returns:
            a tuple of parameters and a tuple of the training information
        """

        losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
            n_early_stopping_delta=n_early_stopping_delta,
            labels=labels,
            attention_masks=masks,
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
        labels,
        attention_masks,
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
                labels,
                attention_masks,
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
                labels,
                attention_masks,
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

    def _validation_loss(self, rng_key, val_iter, labels, masks):

        @nnx.jit
        def body_fn(model, batch_key, batch):
            loss = _cfm_loss(
                model,
                batch_key,
                batch,
                labels,
                masks,
            )
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
        labels,
        masks,
        theta_slices,
        index=None,
        n_samples=4_000,
    ):
        thetas = self.sample_structured_posterior(
            key,
            observable,
            labels,
            masks,
            theta_slices,
            index=index,
            n_samples=n_samples,
        )

        inference_data = as_inference_data(thetas, observable)
        return inference_data

    def sample_structured_posterior(
        self,
        rng_key,
        context,
        labels,
        masks,
        theta_slices,
        n_samples=4_000,
        index=None,
    ):
        if index is not None:
            context_index = {k: index[k] for k in context.keys()}
            theta_index = {k: index[k] for k in theta_slices.keys()}
        else:
            context_index = None
            theta_index = None

        theta_shape = (
            sum(prod(s['event_shape']) for s in theta_slices.values()),
            max(prod(s['batch_shape']) for s in theta_slices.values())
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
                context=context,
                context_label=labels['y'],
                context_index=context_index,
                context_mask=masks['y'],
                theta_shape=theta_shape,
                theta_label=labels['theta'],
                theta_index=theta_index,
                theta_mask=masks['theta'],
                cross_mask=masks['cross'],
                sample_size=n_samples,
            )

            return res

        graphdef, state = nnx.split(self.model)

        thetas = _sample_theta(graphdef, state)
        thetas = decode_theta(
            theta=thetas,
            theta_slices=theta_slices,
            sample_shape=(n_samples,),
        )
        return thetas
