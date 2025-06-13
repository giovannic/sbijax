from flax import nnx
import optax
from typing import Callable
import numpy as np
from tqdm import tqdm
from jax import random as jr, numpy as jnp

from .util.early_stopping import EarlyStopping

def fit_model(
        seed,
        model: nnx.Module,
        loss_fn: Callable,
        train_iter,
        val_iter,
        optimizer: optax.GradientTransformation,
        n_iter: int,
        n_early_stopping_patience: int,
        n_early_stopping_delta: float,
    ):
    nnx_optimizer = nnx.Optimizer(model, optimizer)

    # set model to train
    model.train()

    @nnx.jit
    def step(model, rng, optimizer, batch):
        loss, grads = nnx.value_and_grad(loss_fn)(
            model,
            rng,
            batch
        )
        optimizer.update(grads)
        return loss

    losses = np.zeros([n_iter, 2])
    early_stop = EarlyStopping(
        n_early_stopping_delta,
        n_early_stopping_patience
    )
    best_state = nnx.state(model)
    best_loss = np.inf
    for i in (pbar := tqdm(range(n_iter))):
        train_loss = 0.0
        rng_key = jr.fold_in(seed, i)
        for batch in train_iter:
            train_key, rng_key = jr.split(rng_key)
            batch_loss = step(
                model,
                train_key,
                nnx_optimizer,
                batch
            )
            train_loss += batch_loss * (
                batch['data']["y"].shape[0] / train_iter.num_samples
            )
        val_key, rng_key = jr.split(rng_key)
        model.eval()
        validation_loss = _validation_loss(
            val_key,
            model,
            loss_fn,
            val_iter,
        )
        model.train()
        losses[i] = jnp.array([train_loss, validation_loss])
        pbar.set_description(f"train_loss: {train_loss:.2f}, val_loss: {validation_loss:.2f}")

        _, early_stop = early_stop.update(validation_loss)
        if early_stop.should_stop:
            break
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = nnx.state(model)

    # set the model to the best state
    nnx.update(model, best_state)
    losses = jnp.vstack(losses)[: (i + 1), :] #type: ignore
    return losses

def _validation_loss(
        rng_key: jnp.ndarray,
        model: nnx.Module,
        loss_fn: Callable,
        val_iter):

    @nnx.jit
    def body_fn(model, batch_key, batch):
        loss = loss_fn(
            model,
            batch_key,
            batch,
        )
        return loss * (batch['data']['y'].shape[0] / val_iter.num_samples)

    loss = 0.0
    for batch in val_iter:
        val_key, rng_key = jr.split(rng_key)
        loss += body_fn(model, val_key, batch)
    return loss
