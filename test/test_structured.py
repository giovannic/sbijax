from functools import reduce
from jax import numpy as jnp

from sfmpe.util.dataloader import flatten_structured

def prod(x):
    return reduce(lambda x, y: x * y, x)

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

    # Extract index data
    index_data = {
        'x': data['theta_index']['x'],
        'obs': data['y_index']['obs']
    }
    
    flattened_data, slices = flatten_structured(
        data,
        data_batch_ndims={
            'theta': {'x': 2},
            'y': {'obs': 2}
        },
        index=index_data
    )

    # check that theta and y have been flattened
    target_shape = (sample_size, event_size, prod(batch_shape))
    assert flattened_data['data']['theta'].shape == target_shape
    assert flattened_data['data']['y'].shape == target_shape

    # check that indices are broadcasted
    target_shape = (sample_size, event_size, index_size)
    assert flattened_data['index']['theta'].shape == target_shape
    assert flattened_data['index']['y'].shape == target_shape

    # check that slices have correct structure and metadata
    assert 'theta' in slices and 'y' in slices
    assert 'x' in slices['theta'] and 'obs' in slices['y']
    
    # check theta slice metadata
    theta_slice = slices['theta']['x']
    assert 'offset' in theta_slice
    assert 'event_shape' in theta_slice
    assert 'batch_shape' in theta_slice
    assert theta_slice['offset'] == 0
    assert theta_slice['event_shape'] == (event_size,)
    assert theta_slice['batch_shape'] == batch_shape
    
    # check y slice metadata  
    y_slice = slices['y']['obs']
    assert 'offset' in y_slice
    assert 'event_shape' in y_slice
    assert 'batch_shape' in y_slice
    assert y_slice['offset'] == 0
    assert y_slice['event_shape'] == (event_size,)
    assert y_slice['batch_shape'] == batch_shape

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

    # Extract index data (only has obs_1 index)
    index_data = {
        'x': data['theta_index']['x'],
        'obs_1': data['y_index']['obs_1']
    }
    
    _, slices = flatten_structured(
        data,
        data_batch_ndims={
            'theta': {'x': 2, 'y': 2},
            'y': {'obs_1': 2, 'obs_2': 2}
        },
        index=index_data
    )

    # check that slices have correct structure
    assert 'theta' in slices and 'y' in slices
    assert 'x' in slices['theta'] and 'y' in slices['theta']
    assert 'obs_1' in slices['y'] and 'obs_2' in slices['y']
    
    # check theta slices have correct offsets (x should be at 0, y should be after x)
    theta_x_slice = slices['theta']['x']
    theta_y_slice = slices['theta']['y']
    assert theta_x_slice['offset'] == 0
    assert theta_y_slice['offset'] == event_size  # y comes after x
    assert theta_x_slice['event_shape'] == (event_size,)
    assert theta_y_slice['event_shape'] == (1,)
    
    # check y slices have correct offsets (obs_1 should be at 0, obs_2 should be after obs_1)
    y_obs1_slice = slices['y']['obs_1']
    y_obs2_slice = slices['y']['obs_2'] 
    assert y_obs1_slice['offset'] == 0
    assert y_obs2_slice['offset'] == event_size  # obs_2 comes after obs_1
    assert y_obs1_slice['event_shape'] == (event_size,)
    assert y_obs2_slice['event_shape'] == (1,)
