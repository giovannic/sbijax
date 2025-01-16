import pytest
from jax import numpy as jnp
from jax import random as jr

from sfmpe.fmpe import SFMPE
from sfmpe.nn import make_cnf


def test_fmpe(prior_simulator_tuple):
    tol: float = 1e-3
    y_observed = jnp.array([-1.0, 1.0])
    estim = SFMPE(prior_simulator_tuple, make_cnf(2))
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
