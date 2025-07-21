import pytest
from jax import numpy as jnp
from jax import random as jr
from flax import nnx

from sfmpe.fmpe import SFMPE
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer


def test_fmpe(prior_simulator_tuple):
    tol: float = 1e-3
    y_observed = jnp.array([-1.0, 1.0])
    rngs = nnx.Rngs(0)
    config = {
        'latent_dim': 64,
        'label_dim': 8,
        'index_out_dim': 0,
        'n_encoder': 1,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }
    nn = Transformer(
        config,
        value_dim=1,
        n_labels=3,
        index_dim=0,
        rngs=rngs
    )
    estim = SFMPE(prior_simulator_tuple, CNF(nn))
    data, params = None, {}
    for _ in range(2):
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        params, _ = estim.fit(jr.PRNGKey(2), data=data, n_iter=2)
    posterior, _ = estim.sample_posterior(
        jr.PRNGKey(3),
        params,
        y_observed,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
    assert posterior.posterior.mean() == pytest.approx(1., tol) # type: ignore
