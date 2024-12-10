import pytest
from jax import numpy as jnp
from flax import nnx
import numpy as np
from .transformer import Transformer

@pytest.mark.parameterize('cache_size', 2)
@pytest.mark.parameterize('keys', [['layer_0', 'layer_1']])
@pytest.mark.parameterize('k_shape', [(1, 2, 3, 4)])
@pytest.mark.parameterize('v_shape', [(1, 2, 3, 4)])
def test_creates_cache(config, cache_size, keys, k_shape, v_shape):
    transformer = Transformer(
        config=config,
        rngs=nnx.Rngs(params=0)
    )
    cache = transformer.init_cache(
        cache_size=cache_size,
        batch_size=1,
        dtype=jnp.float32,
    )
    assert all(list(cache.keys()) == keys)
    assert cache['layer_0']['k'].shape == k_shape
    assert cache['layer_0']['v'].shape == v_shape

@pytest.mark.parameterize('batch_size', 4)
@pytest.mark.parameterize('seq_size', 10)
def test_forward_no_cache(
  self,
  config,
  batch_size: int,
  seq_size: int
):
    tol = 1e-5
    cache_size = 6

    token_input = jnp.ones((batch_size, seq_size), dtype=jnp.int32)
    transformer = Transformer(
        config=config,
        rngs=nnx.Rngs(params=0)
    )
    empty_cache = transformer.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    attention_mask = jnp.ones(
        (batch_size, seq_size, cache_size), dtype=jnp.bool
    )
    positions = build_positions_from_mask(token_input != 0)

    output_cache, _ = transformer(
        token_input, positions, empty_cache, attention_mask
    )

    attention_mask = jnp.ones(
        (batch_size, seq_size, seq_size),
        dtype=jnp.bool
    )
    output_none, cache_none = transformer(
        token_input, positions, None, attention_mask
    )

    self.assertIsNone(cache_none)
    np.testing.assert_array_almost_equal(output_cache, output_none, tol)
