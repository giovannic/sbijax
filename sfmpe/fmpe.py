import optax
from jax import jit
from jax import numpy as jnp
from jax import random as jr

from ._ne_base import NE
from .util.data import as_inference_data

from .util.dataloader import (decode_theta, prod)
from .train import fit_model

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

def cross_2d_masks(a, b):
    """ outer product of two 2d masks (sample_size x token_size)

    Args:
        a: (sample_size, token_size_1)
        b: (sample_size, token_size_2)

    Returns:
        (sample_size, token_size_1, token_size_2)
    """
    return jnp.expand_dims(a, 2) * jnp.expand_dims(b, 1)

def make_cross_attention_mask(
    ind_mask,
    theta_padding_mask=None,
    y_padding_mask=None
):
    # extra dim for heads
    cross_mask = jnp.expand_dims(ind_mask, 1)

    if theta_padding_mask is not None:
        # extra heads and y dim
        cross_mask = cross_mask * jnp.expand_dims(theta_padding_mask, (1, 3))
    if y_padding_mask is not None:
        # extra heads and theta dim
        cross_mask = cross_mask * jnp.expand_dims(y_padding_mask, (1, 2))
    return cross_mask

def make_attention_masks(masks):
    theta_mask, context_mask, cross_mask = None, None, None

    if masks is None:
        return (theta_mask, context_mask, cross_mask)

    if 'padding' in masks:
        p_masks = masks['padding']
        theta_padding_mask = p_masks['theta']
        theta_mask = cross_2d_masks(theta_padding_mask, theta_padding_mask)
        y_padding_mask = p_masks['y']
        context_mask = cross_2d_masks(y_padding_mask, y_padding_mask)
        cross_mask = cross_2d_masks(theta_padding_mask, y_padding_mask)

    if 'attention' in masks:
        a_masks = masks['attention']

        if theta_mask is not None:
            theta_mask = theta_mask * a_masks['theta']
        else:
            theta_mask = a_masks['theta']
        if context_mask is not None:
            context_mask = context_mask * a_masks['y']
        else:
            context_mask = a_masks['y']
        if cross_mask is not None:
            cross_mask = cross_mask * a_masks['cross']
        else:
            cross_mask = a_masks['cross']

    # add extra dim for heads
    if theta_mask is not None:
        theta_mask = jnp.expand_dims(theta_mask, 1)
    if context_mask is not None:
        context_mask = jnp.expand_dims(context_mask, 1)
    if cross_mask is not None:
        cross_mask = jnp.expand_dims(cross_mask, 1)

    return theta_mask, context_mask, cross_mask


def _cfm_loss(
    model,
    rng_key,
    batch,
    sigma_min=0.001,
):
    theta = batch["data"]["theta"]
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

    theta_mask, context_mask, cross_mask = make_attention_masks(batch.get('masks'))

    if 'index' in batch:
        theta_index = batch['index']['theta']
        y_index = batch['index']['y']
    else:
        theta_index = None
        y_index = None

    vs = model.vector_field(
        theta=theta_t,
        theta_label=batch['labels']["theta"],
        theta_index=theta_index,
        theta_mask=theta_mask,
        time=times,
        context=batch["data"]["y"],
        context_label=batch["labels"]["y"],
        context_index=y_index,
        context_mask=context_mask,
        cross_mask=cross_mask,
    )
    uts = _ut(theta_t, theta, times, sigma_min)

    # loss has to be masked with the padding mask `batch['theta_mask']`
    # the denominator has to also be derived from the padding mask
    if 'masks' in batch and 'padding' in batch['masks']:
        theta_padding_mask = jnp.expand_dims(batch['masks']['padding']['theta'], -1)
        return jnp.sum(
            jnp.square(vs - uts) * theta_padding_mask
        ) / jnp.sum(theta_padding_mask)

    return jnp.mean(jnp.square(vs - uts))

class SFMPE(NE):

    def __init__(
        self,
        density_estimator,
        **kwargs
        ):
        super().__init__(
            density_estimator,
            **kwargs
        )

    def fit(
        self,
        rng_key,
        train_iter,
        val_iter,
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
                # map=None => diagonal mask for single event dimension
                # map=([i],[j]) ] diagonal mask across event dimension i in key1 and corresponding event dimension j in key2
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
    ):
        fit_model(
            seed,
            self.model,
            _cfm_loss,
            train_iter,
            val_iter,
            optimizer,
            n_iter,
            n_early_stopping_patience,
            n_early_stopping_delta,
        )

    def sample_posterior(
        self,
        key,
        observable,
        labels,
        theta_slices,
        masks=None,
        index=None,
        n_samples=4_000,
    ):
        thetas = self.sample_structured_posterior(
            key,
            observable,
            labels,
            theta_slices,
            masks=masks,
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
        theta_slices,
        masks=None,
        n_samples=4_000,
        index=None,
        theta_0=None,
        direction='forward'
    ):
        if index is not None:
            context_index = {k: index[k] for k in context.keys()} #TODO: broken. Context will be flat
            theta_index = {k: index[k] for k in theta_slices.keys()}
        else:
            context_index = None
            theta_index = None

        theta_shape = (
            sum(prod(s['event_shape']) for s in theta_slices.values()),
            max(prod(s['batch_shape']) for s in theta_slices.values())
        )

        theta_mask, context_mask, cross_mask = make_attention_masks(masks)

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
                context_mask=context_mask,
                theta_shape=theta_shape,
                theta_label=labels['theta'],
                theta_index=theta_index,
                theta_mask=theta_mask,
                cross_mask=cross_mask,
                sample_size=n_samples,
                theta_0=theta_0,
                direction=direction
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
