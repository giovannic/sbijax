import pytest
from functools import reduce
from jax import numpy as jnp, random as jr, vmap
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.fmpe import SFMPE
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.util.dataloader import structured_as_batch_iterators

from flax import nnx

# TODO:
# - test that inference works on infinite case
# - test that inference works on hierarchical case

def prod(x):
    return reduce(lambda x, y: x * y, x)

def sample_index(key, shape):
    points = jnp.cumsum(
        jr.lognormal(key, shape=(3,) + shape),
        axis=0
    )
    return {
        'x': points[0],
        'y': -points[1],
        't': points[2]
    }

def test_structured_loader_flattens_theta_and_y():
    sample_size = 10
    event_size = 3
    batch_shape= (2, 4)
    index_size = 6

    data = {
        'theta': {
            'x': jnp.zeros((sample_size, event_size) + batch_shape),
        },
        'theta_index': {
            'x': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
        'y': {
            'obs': jnp.zeros((sample_size, event_size) + batch_shape),
        },
        'y_index': {
            'obs': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
    }

    train_iter, _, labels = structured_as_batch_iterators(
        jr.PRNGKey(0),
        data,
        3,
        .5,
        True,
        data_batch_ndims={
            'theta': {'x': 2},
            'y': {'obs': 2}
        }
    )

    for batch in train_iter:
        train_n = batch['theta'].shape[0]
        # check that theta and y have been flattened
        target_shape = (train_n, event_size, prod(batch_shape))
        assert batch['theta'].shape == target_shape
        assert batch['y'].shape == target_shape

        # check that indices are broadcasted
        target_shape = (train_n, event_size, index_size)
        assert batch['theta_index'].shape == target_shape
        assert batch['y_index'].shape == target_shape

    # check that labels are correct shape
    assert labels['theta'].shape == (1, event_size, 1)
    assert labels['y'].shape == (1, event_size, 1)

def test_structured_loader_labels_multiple_data():
    sample_size = 10
    event_size = 3
    batch_shape= (2, 4)
    index_size = 6

    data = {
        'theta': {
            'x': jnp.zeros((sample_size, event_size) + batch_shape),
            'y': jnp.zeros((sample_size, 1) + batch_shape),
        },
        'theta_index': {
            'x': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
        'y': {
            'obs_1': jnp.zeros((sample_size, event_size) + batch_shape),
            'obs_2': jnp.zeros((sample_size, 1) + batch_shape),
        },
        'y_index': {
            'obs_1': jnp.arange(sample_size * event_size * index_size).reshape((sample_size, event_size, index_size)),
        },
    }

    _, _, labels = structured_as_batch_iterators(
        jr.PRNGKey(0),
        data,
        3,
        .5,
        True,
        data_batch_ndims={
            'theta': {'x': 2, 'y': 2},
            'y': {'obs_1': 2, 'obs_2': 2}
        },
    )

    # check that labels are correct shape
    assert labels['theta'].shape == (1, event_size + 1, 1)
    assert labels['y'].shape == (1, event_size + 1, 1)

def test_encode_context():
    pass

def test_encode_unknown_theta():
    pass

def test_decode_theta():
    pass

def test_sample_structured_model():
    pass

def test_sampled_structured_model_after_first_round():
    pass

def test_posterior_sampling():
    pass

def test_infinite_parameters():
    tol: float = 1e-3

    theta_index= {
        's': jnp.array([
            [1., -1.],
            [2., -2.],
            [3., -3.],
        ])[jnp.newaxis, :]
    }
    y_index= {
        'obs': jnp.arange(1., 30., dtype=jnp.float32)[
            jnp.newaxis, #sample
            :, #time
            jnp.newaxis #x/y
        ],
    }

    def prior_fn(**kwargs):
        s = kwargs["s"][0]
        # return prior distribution of gaussian random walk at positions x and y
        prior = tfd.JointDistributionNamed(
            dict(
                s=tfd.Normal(s / 2., 1.),
            ),
            batch_ndims=1,
        )
        return prior
       
    def simulator_fn(seed, theta, **kwargs):
        t = kwargs['obs']
        sample_size= theta["s"].shape[0]
        t_n = t.shape[1]

        # sample from guassian mixture
        def sample_row(seed, s, t):
            
            choice_key, noise_key = jr.split(seed)
            return tfd.Normal(
                jnp.broadcast_to(
                    jr.choice(choice_key, s)[jnp.newaxis, :],
                    (t_n, 2)
                ),
                jnp.broadcast_to(1./t, (t_n, 2))
            ).sample(seed=noise_key)

        # vmap over sample size
        keys = jr.split(seed, sample_size)
        samples = vmap(sample_row, in_axes=(0, 0, 0))(
            keys,
            theta["s"],
            jnp.broadcast_to(t, (sample_size, t_n, 2))
        )

        return { 'obs': samples }

    theta_key, y_key = jr.split(jr.PRNGKey(0))

    theta_truth = prior_fn(**theta_index).sample((1,), seed=theta_key)
    y_observed = simulator_fn(y_key, theta_truth, **y_index)

    rngs = nnx.Rngs(0)
    config = {
        'latent_dim': 64,
        'label_dim': 64,
        'index_out_dim': 64,
        'n_encoder': 1,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }
    theta_value_dim = 2
    nn = Transformer(
        config,
        context_value_dim=2,
        n_context_labels=1,
        context_index_dim=1,
        theta_value_dim=theta_value_dim,
        n_theta_labels=2,
        theta_index_dim=2,
        rngs=rngs
    )
    theta_event_size= 3
    model = CNF(
        theta_shape=(theta_event_size, theta_value_dim),
        transform=nn
    )

    data_batch_ndims = {
        'theta': {'s': 1},
        'y': {'obs': 1}
    }

    theta_batch_shapes = { 's': (2,) }

    estim = SFMPE(
        (prior_fn, simulator_fn),
        model,
        theta_batch_shapes
    )
    data, params = None, {}
    for _ in range(1):
        # TODO: params will never be created!
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            context_index=y_index,
            theta_index=theta_index,
            data=data,
            n_simulations=1_000,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        estim.fit(
            jr.PRNGKey(2),
            data=data,
            n_iter=1_000,
            data_batch_ndims=data_batch_ndims,
        )
    posterior, _ = estim.sample_posterior( 
        rngs,
        y_observed,
        context_index=y_index,
        theta_index=theta_index,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
    s_hat = jnp.mean(jnp.array(posterior.posterior.s), keepdims=True, axis=0) # type: ignore
    s_truth = theta_truth['s'] # type: ignore
    print(s_hat)
    print(s_truth)
    assert s_hat == pytest.approx(s_truth, tol)
    assert False
