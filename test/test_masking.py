import pytest
import jax.numpy as jnp
from sfmpe.util.dataloader import build_padding_mask, prod

def test_single_block_single_sample_2d():
    """
    A single block with padded shape (3, 4).
    We supply an actual shape (2, 2) for the single sample.
    The valid region is (r < 2) and (c < 2).
    So we expect a flattened dimension size=12, but only indices
    corresponding to (r=0,c=0), (0,1), (1,0), (1,1) => (4 tokens) to be 1.
    The rest are 0.
    """
    block_shape = (3, 4)
    block_size = prod(block_shape)  # 3*4=12
    block_info = {
        "offset": 0,
        "size": block_size,
        "shape": block_shape,
    }

    # Only 1 key in block_slices
    block_slices = {
        "my_block": block_info
    }

    # We have a single sample, actual shape is (2, 2)
    # => we store it in shape (1,2)? or just (2,). The code expects the
    # event_shapes["my_block"] to have shape [sample_shape + (n_event_dims,)].
    # We'll do sample_shape=() => a 0D "batch"
    # => final shape is (2,). So let's do:
    event_shapes = {
        "my_block": jnp.array([2,2])  # shape (2,) => n_event_dims=2
    }

    mask = build_padding_mask(event_shapes, block_slices)
    # We expect mask.shape = (sum_of_block_sizes,) => (12,)
    assert mask.shape == (12,)

    # Let's see which indices should be 1:
    # row-major flatten of (3,4):
    #   (r=0,c=0)-> idx=0, (0,1)->1, (0,2)->2, (0,3)->3,
    #   (1,0)->4, (1,1)->5, (1,2)->6, (1,3)->7,
    #   (2,0)->8, (2,1)->9, (2,2)->10, (2,3)->11
    # We only want (0,0),(0,1),(1,0),(1,1) => idx=0,1,4,5
    mask_np = jnp.array(mask)

    expected = jnp.zeros((12,), dtype=jnp.float32)
    expected = expected.at[jnp.array([0,1,4,5])].set(1.0)

    assert jnp.allclose(mask_np, expected), f"Got {mask_np}, expected {expected}"

def test_single_block_multi_sample_2d():
    """
    Two samples, one block.
    block_shape=(3,2)-> size=6, we supply a shape(3,2)

    sample0 => actual shape(1,2) => row in [0..0], col in [0..1]
    sample1 => actual shape(2,1) => row in [0..1], col in [0..0]
    We confirm that the mask is shape (2,6) and has the correct 1/0 entries.
    """
    block_shape = (3,2)
    block_size = prod(block_shape)  # 6
    block_info = {
        "offset": 0,
        "size": block_size,
        "shape": block_shape,
    }
    block_slices = {
        "my_block": block_info
    }

    # sample_shape=(2,) => 2 samples. event dim=2 for (h,w).
    # event_shapes["my_block"].shape=(2,2)
    # sample0 => (1,2), sample1 => (2,1)
    shapes = jnp.array([
        [1,2],
        [2,1]
    ])
    event_shapes = {
        "my_block": shapes
    }
    mask = build_padding_mask(event_shapes, block_slices)
    # shape => (2,6)
    assert mask.shape == (2,6)

    # For block(3,2), row-major flatten is:
    #   (r=0,c=0)->0, (0,1)->1, (1,0)->2, (1,1)->3, (2,0)->4, (2,1)->5
    # sample0 => (1,2): valid coords => (0,0),(0,1). => idx=0,1 => => mask=1. rest=0
    # sample1 => (2,1): valid coords => (0,0),(1,0). => => idx=0,2 => => mask=1, rest=0
    mask_np = jnp.array(mask)

    expected = jnp.array([
        [1,1,0,0,0,0],  # sample0
        [1,0,1,0,0,0],  # sample1
    ], dtype=jnp.float32)

    assert jnp.allclose(mask_np, expected)

def test_two_blocks_single_sample_2d():
    """
    We have 2 blocks, each with shapes like (2,2) => size=4.
    We'll pass them in offset order, and rely on your function
    to concatenate. There's 1 sample => shape is (4 + 4)=8 in the last dim.

    For blockA => actual shape = (2,1), valid => row<2,col<1 => coords=(0,0),(1,0)
    For blockB => actual shape = (1,2), valid => coords=(0,0),(0,1)
    We'll verify that we get indices for blockA first, blockB second.
    """
    blockA = {
        "offset": 0,
        "size": 4,
        "shape": (2,2),
    }
    blockB = {
        "offset": 4,
        "size": 4,
        "shape": (2,2),
    }
    # The dictionary is sorted by offset => blockA then blockB
    block_slices = {
        "blockA": blockA,
        "blockB": blockB,
    }

    # single sample => shape=(2,) => n_event_dims=2
    # actual shape for blockA => (2,1)
    # actual shape for blockB => (1,2)
    event_shapes = {
        "blockA": jnp.array([2,1]),
        "blockB": jnp.array([1,2]),
    }

    mask = build_padding_mask(event_shapes, block_slices)
    # shape => (8,). Because sample_shape=(), each block is size=4 => total=8
    assert mask.shape == (8,)

    # blockA flatten => row-major of (2,2):
    #   (0,0)->idx0, (0,1)->1, (1,0)->2, (1,1)->3
    # actual shape(2,1) => valid are idx=0,2 => rest=0
    # blockB flatten => also shape(2,2):
    #   (0,0)->idx0, (0,1)->1, (1,0)->2, (1,1)->3
    # actual shape(1,2) => valid are idx=0,1 => rest=0
    # final => [ blockA(4), blockB(4) ] => total=8
    mask_np = mask
    expected = jnp.array([1,0,1,0,   # blockA
                          1,1,0,0], dtype=jnp.float32)  # blockB
    assert jnp.allclose(mask_np, expected)

def test_check_axis_negative_two():
    """
    Just to confirm that axis=-2 does what we expect, let's try
    a multi-d sample_shape. E.g. sample_shape=(2,2). Then each block
    returns shape=(2,2, block_size). We want to see if axis=-2 merges
    them into shape=(2,2, total_size). If so, great. If not, we see an error.
    """
    block_slices = {
        "b0": {"offset":0, "size":4, "shape":(2,2)},
        "b1": {"offset":4, "size":6, "shape":(2,3)},
    }
    # sample_shape=(2,2), n_event_dims=2 => event_shapes["b0"].shape=(2,2,2)
    # We'll do an example with small actual shapes, ignoring correctness for brevity
    # This is just to see the final shape.
    shapes_b0 = jnp.array([
        [[2,2],[2,2]],   # sample=0 => shape(2,2) for both "rows" => nonsense but just a test
        [[1,1],[2,2]]    # sample=1
    ])
    shapes_b1 = jnp.array([
        [[2,3],[2,3]],
        [[2,3],[2,3]]
    ])
    event_shapes = {
        "b0": shapes_b0,
        "b1": shapes_b1
    }
    mask = build_padding_mask(event_shapes, block_slices)
    # If successful, mask.shape should be (2,2, 4+6) => (2,2,10)
    assert mask.shape == (2,2,10), f"Got shape {mask.shape}, expected (2,2,10)."


@pytest.mark.parametrize("use_unsupported_offset_order", [True, False])
def test_offset_order(use_unsupported_offset_order):
    """
    Checks that if we pass block_slices out-of-offset order, the result is
    not necessarily correct. If we pass them in correct offset order, we get
    correct shape.
    """
    # We'll do 2 blocks again, but reversed order if use_unsupported_offset_order
    blockA = {"offset":0, "size":4, "shape":(2,2)}
    blockB = {"offset":4, "size":4, "shape":(2,2)}
    if use_unsupported_offset_order:
        block_slices = {"blockB": blockB, "blockA": blockA}
    else:
        block_slices = {"blockA": blockA, "blockB": blockB}

    event_shapes = {
        "blockA": jnp.array([2,2]),
        "blockB": jnp.array([2,2]),
    }
    mask = build_padding_mask(event_shapes, block_slices)

    # shape => (8,) either way
    assert mask.shape == (8,)

def test_block_3d_single_sample():
    """
    Tests a single block with padded shape=(2,2,2), total size=8.
    The actual shape for this sample is (1,2,1).
    Valid coordinates are (z=0, y=0..1, x=0..0) => (0,0,0) => idx=0, (0,1,0) => idx=2.
    Final mask => [1, 0, 1, 0,  0, 0, 0, 0].
    """
    import jax.numpy as jnp
    
    # Padded shape => (2,2,2), so 8 tokens total in row-major flatten
    block_info = {
        "offset": 0,
        "size": 8,             # 2*2*2 = 8
        "shape": (2,2,2),      # (Z,Y,X)
    }
    # Single block dictionary
    block_slices = {
        "my_3d_block": block_info
    }
    
    # Single sample => no leading sample dims, event_dims=3 => shape=(3,).
    # The actual shape is (z=1, y=2, x=1).
    # So event_shapes["my_3d_block"] = jnp.array([1,2,1]).
    event_shapes = {
        "my_3d_block": jnp.array([1,2,1])
    }
    
    mask = build_padding_mask(event_shapes, block_slices)
    # We expect shape=(8,)
    assert mask.shape == (8,), f"Got {mask.shape}, expected (8,)."

    # (z=0,y=0,x=0) => flattened index=0 => valid
    # (z=0,y=1,x=0) => flattened index=2 => valid
    # everything else => 0
    expected = jnp.array([1,0,1,0,  0,0,0,0], dtype=jnp.float32)
    assert jnp.allclose(mask, expected), f"mask={mask}, expected={expected}"
