from jaxtyping import Array, PyTree
from typing import Optional, Literal
import optax
from jax import jit, numpy as jnp, random as jr, vmap

from .util.dataloader import (decode_theta, prod)
from .train import fit_model_no_branch
from .structured_cnf import StructuredCNF
from .fmpe import theta_t_linear, ut_linear

from flax import nnx

Direction = Literal['forward', 'backward']

def _cross_2d_masks(a, b):
    """ outer product of two 2d masks (sample_size x token_size)

    Args:
        a: (sample_size, token_size_1)
        b: (sample_size, token_size_2)

    Returns:
        (sample_size, token_size_1, token_size_2)
    """
    return jnp.expand_dims(a, 2) * jnp.expand_dims(b, 1)

def _make_attention_masks(masks):
    theta_mask, context_mask, cross_mask = None, None, None

    if masks is None:
        return (theta_mask, context_mask, cross_mask)

    if 'padding' in masks:
        p_masks = masks['padding']
        theta_padding_mask = p_masks['theta']
        theta_mask = _cross_2d_masks(theta_padding_mask, theta_padding_mask)
        y_padding_mask = p_masks['y']
        context_mask = _cross_2d_masks(y_padding_mask, y_padding_mask)
        cross_mask = _cross_2d_masks(theta_padding_mask, y_padding_mask)

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
    theta_0 = jr.normal(theta_key, shape=theta.shape)
    theta_t = theta_t_linear(
        theta_0,
        times,
        theta,
        sigma_min
    )

    theta_mask, context_mask, cross_mask = _make_attention_masks(batch.get('masks'))

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
    us = ut_linear(theta_t, theta, times, sigma_min)

    # loss has to be masked with the padding mask `batch['theta_mask']`
    # the denominator has to also be derived from the padding mask
    if 'masks' in batch and 'padding' in batch['masks']:
        theta_padding_mask = jnp.expand_dims(batch['masks']['padding']['theta'], -1)
        return jnp.sum(
            jnp.square(vs - us) * theta_padding_mask
        ) / jnp.sum(theta_padding_mask)

    return jnp.mean(jnp.square(vs - us))

class SFMPE:

    def __init__(self, vector_field_model: StructuredCNF, rngs=nnx.Rngs(0)):
        self.model = vector_field_model
        self.rngs = rngs

    def fit(
        self,
        train: PyTree,
        val: PyTree,
        optimizer: optax.GradientTransformation = optax.adam(0.0003),
        n_iter: int = 1000,
        batch_size: int = 100
    ):
        """Fit the model"""

        losses = fit_model_no_branch(
            self.model,
            self.rngs.permutations(),
            _cfm_loss,
            train,
            val,
            optimizer,
            n_iter,
            batch_size
        )

        return losses

    def sample_posterior(
        self,
        context: Array,
        labels: Array,
        theta_slices: PyTree,
        masks: Optional[PyTree] = None,
        index: Optional[PyTree] = None,
        n_samples: int = 1_000,
        theta_0: Optional[PyTree]=None,
        direction: Direction = 'forward',
    ) -> PyTree:
        if index is not None:
            context_index = index['y']
            theta_index = index['theta']
        else:
            context_index = None
            theta_index = None

        theta_shape = (
            sum(prod(s['event_shape']) for s in theta_slices.values()),
            max(prod(s['batch_shape']) for s in theta_slices.values())
        )

        theta_mask, context_mask, cross_mask = _make_attention_masks(masks)

        self.model.eval()

        # NOTE: nnx.jit somehow leaks tracers. Need to investigate
        @jit
        def _sample_theta(
            graphdef,
            state
            ):
            model = nnx.merge(graphdef, state)
            res = model.sample(
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

    def sample_base_dist(
        self,
        theta: Array,
        context: Array,
        labels: Array,
        theta_slices: PyTree, #TODO: stricter typing
        masks: Optional[PyTree] = None,
        index: Optional[PyTree] = None,
    ) -> PyTree:
        def sample_pair(theta, y):
            return self.sample_posterior(
                y[None, ...],
                labels,
                theta_slices,
                masks=masks,
                index=index,
                n_samples=1,
                theta_0=theta[None, ...],
                direction='backward'
            )

        return vmap(sample_pair)(theta, context)
