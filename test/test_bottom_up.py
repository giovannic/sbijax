import unittest.mock
import pytest
from unittest.mock import Mock
import optax
from flax import nnx
from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sfmpe.bottom_up import train_bottom_up
from sfmpe.sfmpe import SFMPE
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.util.dataloader import flatten_structured


def create_hierarchical_prior_fn(var_mu: float = 1.0, 
                                var_theta: float = 1.0):
    """Create hierarchical prior function similar to hierarchical_gaussian.py"""
    def prior_fn(n):
        prior = tfd.JointDistributionNamed(
            dict(
                mu=tfd.Normal(
                    jnp.zeros((1, 1)), 
                    jnp.full((1, 1), var_mu)
                ),
                theta=lambda mu: tfd.Independent(
                    tfd.Normal(
                        jnp.repeat(mu, n, axis=-2),
                        var_theta
                    ),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=1,
        )
        return prior
    return prior_fn


def create_hierarchical_simulator_fn(n_obs: int, var_obs: float = 1.0):
    """Create hierarchical simulator function similar to hierarchical_gaussian.py"""
    def simulator_fn(seed, theta):
        obs = tfd.Independent(
            tfd.Normal(theta['theta'], var_obs),
            reinterpreted_batch_ndims=1
        ).sample((n_obs,), seed=seed)
        obs = jnp.transpose(obs, (1, 2, 0, 3))  # type: ignore
        return {
            'obs': obs
        }
    return simulator_fn


def create_f_in_fn(n_obs: int, obs_rate: float = 1.0):
    """Create f_in function that returns TFP distribution similar to hierarchical_brownian.py"""
    def f_in_fn(n):
        return tfd.JointDistributionNamed(
            dict(
                obs=tfd.Exponential(jnp.full((n, 1), obs_rate)),
            ),
            batch_ndims=1
        )
    return f_in_fn


def create_hierarchical_simulator_fn_with_f_in(n_obs: int):
    """Create hierarchical simulator function that accepts f_in parameter"""
    def simulator_fn(seed, theta, f_in):
        # theta['theta'] has shape (n_simulations, n_theta, 1)
        # f_in['obs'] has shape (n_simulations, n_obs, 1)
        # We want n_theta means each with n_obs different variances
        # Expand theta to (n_simulations, n_theta, n_obs, 1) and f_in to (n_simulations, n_theta, n_obs, 1)
        theta_expanded = jnp.expand_dims(theta['theta'], -2)  # (n_simulations, n_theta, 1, 1)
        theta_expanded = jnp.broadcast_to(theta_expanded, (*theta_expanded.shape[:-2], n_obs, 1))  # (n_simulations, n_theta, n_obs, 1)
        
        f_in_expanded = jnp.expand_dims(f_in['obs'], 1)  # (n_simulations, 1, n_obs, 1)
        f_in_expanded = jnp.broadcast_to(f_in_expanded, (*f_in_expanded.shape[:-3], theta['theta'].shape[1], n_obs, 1))  # (n_simulations, n_theta, n_obs, 1)
        
        obs = tfd.Independent(
            tfd.Normal(theta_expanded, f_in_expanded),
            reinterpreted_batch_ndims=2
        ).sample(seed=seed)
        return {
            'obs': obs  # Shape: (n_simulations, n_theta, n_obs, 1)
        }
    return simulator_fn


def create_test_estimator(key: jnp.ndarray):
    """Create SFMPE estimator for testing"""
    rngs = nnx.Rngs(key)
    
    transformer_config = {
        'latent_dim': 16,
        'label_dim': 16,
        'index_out_dim': 0,
        'n_encoder': 2,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 32,
        'dropout': 0.0,
        'activation': nnx.relu,
    }
    
    nn = Transformer(
        transformer_config,
        value_dim=1,
        n_labels=3,
        index_dim=0,
        rngs=rngs
    )
    
    model = StructuredCNF(nn, rngs=rngs)
    estim = SFMPE(model, rngs=rngs)
    
    return estim


def create_test_estimator_with_index(key: jnp.ndarray):
    """Create SFMPE estimator with index support for f_in testing"""
    rngs = nnx.Rngs(key)
    
    transformer_config = {
        'latent_dim': 16,
        'label_dim': 16,
        'index_out_dim': 2,
        'n_encoder': 2,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 32,
        'dropout': 0.0,
        'activation': nnx.relu,
    }
    
    nn = Transformer(
        transformer_config,
        value_dim=1,
        n_labels=3,
        index_dim=1,
        rngs=rngs
    )
    
    model = StructuredCNF(nn, rngs=rngs)
    estim = SFMPE(model, rngs=rngs)
    
    return estim


@pytest.fixture
def hierarchical_setup():
    """Setup hierarchical Gaussian test scenario"""
    n_theta = 3
    n_obs = 5
    n_local = n_theta
    var_mu = 1.0
    var_theta = 0.5
    var_obs = 0.3
    
    independence = {
        'local': ['obs'],
        'cross': [('mu', 'obs')],
        'cross_local': [('theta', 'obs', (0, 0))]
    }
    
    prior_fn = create_hierarchical_prior_fn(var_mu, var_theta)
    simulator_fn = create_hierarchical_simulator_fn(n_obs, var_obs)
    
    # Generate observed data
    key = jr.PRNGKey(42)
    theta_key, y_key, estim_key = jr.split(key, 3)
    
    theta_truth = prior_fn(n_theta).sample((1,), seed=theta_key)
    y_observed = simulator_fn(y_key, theta_truth)
    
    estim = create_test_estimator(estim_key)
    
    return {
        'prior_fn': prior_fn,
        'simulator_fn': simulator_fn,
        'estim': estim,
        'y_observed': y_observed,
        'independence': independence,
        'n_theta': n_theta,
        'n_obs': n_obs,
        'n_local': n_local,
        'global_names': ['mu'],
        'local_names': ['theta']
    }


def test_train_bottom_up_single_round(hierarchical_setup):
    """Test train_bottom_up with a single training round"""
    setup = hierarchical_setup
    n_rounds = 1
    n_simulations = 50
    n_epochs = 1
    
    key = jr.PRNGKey(123)
    
    # Create spies using wraps
    prior_spy = Mock(wraps=setup['prior_fn'])
    simulator_spy = Mock(wraps=setup['simulator_fn'])
    fit_spy = Mock(wraps=setup['estim'].fit)
    flatten_spy = Mock(wraps=flatten_structured)
    
    # Patch the functions
    with unittest.mock.patch.object(setup['estim'], 'fit', fit_spy), \
         unittest.mock.patch('sfmpe.bottom_up.flatten_structured', flatten_spy):
        
        labels, slices, masks = train_bottom_up(
            key=key,
            estim=setup['estim'],
            prior_fn=prior_spy,
            simulator_fn=simulator_spy,
            global_names=setup['global_names'],
            local_names=setup['local_names'],
            n_local=setup['n_local'],
            n_rounds=n_rounds,
            n_simulations=n_simulations,
            n_epochs=n_epochs,
            y_observed=setup['y_observed'],
            independence=setup['independence'],
            optimiser=optax.adam(0.001),
            batch_size=10
        )
    
    # Verify prior_fn called once with correct argument
    assert prior_spy.call_count == 1
    call_args = prior_spy.call_args_list[0]
    # prior_fn is called with 1 as first argument (then .sample() is called)
    assert call_args[0][0] == 1
    
    # Verify simulator_fn called once
    assert simulator_spy.call_count == 1
    sim_call = simulator_spy.call_args_list[0]
    theta_samples = sim_call[0][1]  # Second argument is theta
    assert set(theta_samples.keys()) == {'mu', 'theta'}
    # theta shape is (n_simulations, 1, 1) because prior_fn(1) creates theta with n=1
    assert theta_samples['theta'].shape == (n_simulations, 1, 1)
    
    # Verify flatten_structured called exactly 3 times for single round
    assert flatten_spy.call_count == 3
    
    # 1st call: z_data flattening for initial simulation training data
    first_call = flatten_spy.call_args_list[0][0][0]  
    assert set(first_call.keys()) == {'theta', 'y'}
    assert set(first_call['theta'].keys()) == {'theta'}  # local parameters only
    assert set(first_call['y'].keys()) == {'obs', 'mu'}  # observations + global parameters
    
    # 2nd call: z_sim flattening for z sampling
    second_call = flatten_spy.call_args_list[1][0][0]
    assert set(second_call.keys()) == {'theta', 'y'}
    assert set(second_call['theta'].keys()) == {'theta'}  # local parameters only
    assert set(second_call['y'].keys()) == {'obs', 'mu'}  # observations + global parameters
    
    # 3rd call: final data flattening for full posterior training
    third_call = flatten_spy.call_args_list[2][0][0]
    assert set(third_call.keys()) == {'theta', 'y'}
    assert set(third_call['theta'].keys()) == {'theta', 'mu'}  # local + global parameters
    assert set(third_call['y'].keys()) == {'obs'}  # only observations
    
    # Verify estim.fit called twice (once for p(z|theta,y), once for p(theta,z_vec|y_vec))
    assert fit_spy.call_count == 2
    
    # Verify return values structure
    # labels should be dict with theta and y keys containing label arrays
    assert isinstance(labels, dict)
    assert set(labels.keys()) == {'theta', 'y'}
    # theta labels shape: (1, 1 + n_theta) where 1 is for mu and n_theta for theta
    assert labels['theta'].shape == (1, 1 + setup['n_theta'])
    # y labels shape: (1, n_theta * n_obs)
    assert labels['y'].shape == (1, setup['n_theta'] * setup['n_obs'])
    
    # slices should be dict with metadata for theta parameters  
    assert isinstance(slices, dict)
    assert set(slices.keys()) == {'theta', 'mu'}  # Should match final theta structure

    # masks should be dict with attention key
    assert masks is not None
    assert isinstance(masks, dict)
    assert 'attention' in masks
    # attention masks should have theta, y, and cross keys
    assert set(masks['attention'].keys()) == {'theta', 'y', 'cross'}  # type: ignore


def test_train_bottom_up_with_f_in_single_round():
    """Test train_bottom_up with f_in and f_in_args (single round)"""
    n_theta = 3
    n_obs = 5
    n_local = n_theta
    var_mu = 1.0
    var_theta = 0.5
    obs_rate = 1.0
    
    independence = {
        'local': ['obs'],
        'cross': [('mu', 'obs')],
        'cross_local': [('theta', 'obs', (0, 0))]
    }
    
    prior_fn = create_hierarchical_prior_fn(var_mu, var_theta)
    simulator_fn = create_hierarchical_simulator_fn_with_f_in(n_obs)
    f_in_fn = create_f_in_fn(n_obs, obs_rate)
    
    # Generate observed data
    key = jr.PRNGKey(42)
    theta_key, y_key, f_in_key, estim_key = jr.split(key, 4)
    
    theta_truth = prior_fn(n_theta).sample((1,), seed=theta_key)
    f_in_truth = f_in_fn(n_obs).sample((1,), seed=f_in_key)
    y_observed = simulator_fn(y_key, theta_truth, f_in_truth)
    
    estim = create_test_estimator_with_index(estim_key)
    
    # Test parameters
    n_rounds = 1
    n_simulations = 50
    n_epochs = 1
    
    key = jr.PRNGKey(123)
    
    # Create spies using wraps
    prior_spy = Mock(wraps=prior_fn)
    simulator_spy = Mock(wraps=simulator_fn)
    f_in_spy = Mock(wraps=f_in_fn)
    fit_spy = Mock(wraps=estim.fit)
    flatten_spy = Mock(wraps=flatten_structured)
    sample_posterior_spy = Mock(wraps=estim.sample_posterior)
    
    # Patch the functions
    with unittest.mock.patch.object(estim, 'fit', fit_spy), \
         unittest.mock.patch.object(estim, 'sample_posterior', sample_posterior_spy), \
         unittest.mock.patch('sfmpe.bottom_up.flatten_structured', flatten_spy):
        
        labels, slices, masks = train_bottom_up(
            key=key,
            estim=estim,
            prior_fn=prior_spy,
            simulator_fn=simulator_spy,
            global_names=['mu'],
            local_names=['theta'],
            n_local=n_local,
            n_rounds=n_rounds,
            n_simulations=n_simulations,
            n_epochs=n_epochs,
            y_observed=y_observed,
            independence=independence,
            optimiser=optax.adam(0.001),
            batch_size=10,
            f_in=f_in_spy,
            f_in_args=(n_obs,)
        )
    
    # 1. f_in Function Calls
    assert f_in_spy.call_count == 1
    f_in_call = f_in_spy.call_args_list[0]
    assert f_in_call[0][0] == n_obs  # Called with n_obs argument
    
    # 2. Simulator Function Calls
    assert simulator_spy.call_count == 1
    sim_call = simulator_spy.call_args_list[0]
    assert len(sim_call[0]) == 3  # Called with 3 arguments (seed, theta, f_in)
    f_in_sample = sim_call[0][2]  # Third argument is f_in_sample
    assert 'obs' in f_in_sample
    assert f_in_sample['obs'].shape == (n_simulations, n_obs, 1)
    
    # 3. flatten_structured Index Parameter
    # Verify flatten_structured called exactly 3 times for single round
    assert flatten_spy.call_count == 3
    
    # 1st call: z_data flattening for initial simulation training data
    first_call = flatten_spy.call_args_list[0]
    assert 'index' in first_call[1]  # Should have index kwarg
    index_first = first_call[1]['index']
    # Expected index shape: flattened f_in with shape (n_simulations, n_obs, 1)
    assert index_first.shape == (n_simulations, n_obs, 1)
    
    # 2nd call: z_sim flattening for z sampling  
    second_call = flatten_spy.call_args_list[1]
    assert 'index' in second_call[1]  # Should have index kwarg
    index_second = second_call[1]['index']
    # Expected index shape: (n_simulations, n_local * n_obs, 1) = (50, 15, 1)
    assert index_second.shape == (n_simulations, n_local * n_obs, 1)
    
    # 3rd call: final data flattening for full posterior training
    third_call = flatten_spy.call_args_list[2]
    assert 'index' in third_call[1]  # Should have index kwarg
    index_third = third_call[1]['index']
    # Expected index shape: (n_simulations, n_local * n_obs, 1) = (50, 15, 1)
    assert index_third.shape == (n_simulations, n_local * n_obs, 1)
    
    # 4. sample_posterior Integration
    # Check sample_posterior calls receive flattened f_in data via index kwarg
    assert sample_posterior_spy.call_count > 0
    posterior_call = sample_posterior_spy.call_args_list[0]
    assert 'index' in posterior_call[1]  # Should have index kwarg
    index_context = posterior_call[1]['index']['context']
    # Expected context index shape: (n_simulations, n_obs, 1) = (50, 5, 1)
    assert index_context.shape == (n_simulations, n_obs, 1)


def test_train_bottom_up_multiple_rounds(hierarchical_setup):
    """Test train_bottom_up with multiple training rounds"""
    setup = hierarchical_setup
    n_rounds = 2
    n_simulations = 50
    n_epochs = 1
    
    key = jr.PRNGKey(456)
    
    # Create spies using wraps
    prior_spy = Mock(wraps=setup['prior_fn'])
    simulator_spy = Mock(wraps=setup['simulator_fn'])
    fit_spy = Mock(wraps=setup['estim'].fit)
    flatten_spy = Mock(wraps=flatten_structured)
    
    # Patch the functions
    with unittest.mock.patch.object(setup['estim'], 'fit', fit_spy), \
         unittest.mock.patch('sfmpe.bottom_up.flatten_structured', flatten_spy):
        
        labels, slices, masks = train_bottom_up(
            key=key,
            estim=setup['estim'],
            prior_fn=prior_spy,
            simulator_fn=simulator_spy,
            global_names=setup['global_names'],
            local_names=setup['local_names'],
            n_local=setup['n_local'],
            n_rounds=n_rounds,
            n_simulations=n_simulations,
            n_epochs=n_epochs,
            y_observed=setup['y_observed'],
            independence=setup['independence'],
            optimiser=optax.adam(0.001),
            batch_size=10
        )
    
    # Verify prior_fn called only once (first round only)
    assert prior_spy.call_count == 1
    
    # Verify simulator_fn called n_rounds times
    assert simulator_spy.call_count == n_rounds
    
    # Verify both simulator calls have correct theta structure
    for i in range(n_rounds):
        sim_call = simulator_spy.call_args_list[i]
        theta_samples = sim_call[0][1]
        assert set(theta_samples.keys()) == {'mu', 'theta'}
        # In round 1: samples from prior, in round 2+: samples from posterior
        assert theta_samples['theta'].shape[0] == n_simulations
    
    # Verify flatten_structured call count for 2 rounds:
    # Round 1: 3 calls (z_data, z_sim, final data)
    # Round 2: 4 calls (full posterior sample, z_data, z_sim, final data) 
    # Total: 3 + 4 = 7 calls
    assert flatten_spy.call_count == 7
    
    # Round 1 calls (indices 0-2):
    # 1st call: z_data flattening
    round1_call1 = flatten_spy.call_args_list[0][0][0]
    assert set(round1_call1.keys()) == {'theta', 'y'}
    assert set(round1_call1['theta'].keys()) == {'theta'}
    assert set(round1_call1['y'].keys()) == {'obs', 'mu'}
    
    # 2nd call: z_sim flattening  
    round1_call2 = flatten_spy.call_args_list[1][0][0]
    assert set(round1_call2.keys()) == {'theta', 'y'}
    assert set(round1_call2['theta'].keys()) == {'theta'}
    assert set(round1_call2['y'].keys()) == {'obs', 'mu'}
    
    # 3rd call: final data flattening
    round1_call3 = flatten_spy.call_args_list[2][0][0]
    assert set(round1_call3.keys()) == {'theta', 'y'}
    assert set(round1_call3['theta'].keys()) == {'theta', 'mu'}
    assert set(round1_call3['y'].keys()) == {'obs'}
    
    # Round 2 calls (indices 3-6):
    # 1st call: full posterior sample flattening
    round2_call1 = flatten_spy.call_args_list[3][0][0]
    assert set(round2_call1.keys()) == {'theta', 'y'}
    assert set(round2_call1['theta'].keys()) == {'theta', 'mu'}
    assert set(round2_call1['y'].keys()) == {'obs'}
    
    # 2nd call: z_data flattening
    round2_call2 = flatten_spy.call_args_list[4][0][0]
    assert set(round2_call2.keys()) == {'theta', 'y'}
    assert set(round2_call2['theta'].keys()) == {'theta'}
    assert set(round2_call2['y'].keys()) == {'obs', 'mu'}
    
    # 3rd call: z_sim flattening
    round2_call3 = flatten_spy.call_args_list[5][0][0]
    assert set(round2_call3.keys()) == {'theta', 'y'}
    assert set(round2_call3['theta'].keys()) == {'theta'}
    assert set(round2_call3['y'].keys()) == {'obs', 'mu'}
    
    # 4th call: final data flattening
    round2_call4 = flatten_spy.call_args_list[6][0][0]
    assert set(round2_call4.keys()) == {'theta', 'y'}
    assert set(round2_call4['theta'].keys()) == {'theta', 'mu'}
    assert set(round2_call4['y'].keys()) == {'obs'}
    
    # Verify estim.fit called 2 * n_rounds times
    assert fit_spy.call_count == 2 * n_rounds
    
    # Verify return values have expected structure
    assert isinstance(labels, dict)
    assert set(labels.keys()) == {'theta', 'y'}
    # theta labels shape: (1, 1 + n_theta) where 1 is for mu and n_theta for theta
    assert labels['theta'].shape == (1, 1 + setup['n_theta'])
    # y labels shape: (1, n_theta * n_obs)
    assert labels['y'].shape == (1, setup['n_theta'] * setup['n_obs'])
    
    assert isinstance(slices, dict)
    assert set(slices.keys()) == {'theta', 'mu'}
    
    assert masks is not None
    assert isinstance(masks, dict)
    assert 'attention' in masks
    assert set(masks['attention'].keys()) == {'theta', 'y', 'cross'}  # type: ignore
