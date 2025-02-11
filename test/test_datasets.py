from sfmpe.util.dataloader import combine_data
import jax.numpy as jnp
import numpy as np

def assert_array_equal(a, b):
    np.testing.assert_array_equal(np.array(a), np.array(b))

def test_combine_data_all_fields():
    pad_value = 42
    # Both x and y have all required fields: "data", "labels", "masks", and "index".
    # For data, we choose "theta" and "y" blocks with different time dimensions.
    x = {
        "data": {
            "theta": jnp.array([[1, 2, 3],
                                  [4, 5, 6]]),       # shape (2,3)
            "y":     jnp.array([[10, 20, 30],
                                  [40, 50, 60]])
        },
        "labels": {
            "theta": jnp.array([[100, 200, 300]]),   # shape (1,3); will be broadcast to (2,3)
            "y":     jnp.array([[1000, 2000, 3000]])
        },
        "masks": {
            "padding": {
                "theta": jnp.array([[1, 1, 1]]),     # shape (1,3)
                "y":     jnp.array([[1, 1, 1]])
            },
            "attention": {
                "theta": jnp.array([[[1, 1],
                                     [1, 1]]]),       # shape (1,2,2)
                "y":     jnp.array([[[2, 2],
                                     [2, 2]]]),
                "cross": jnp.array([[[1, 1],
                                     [1, 1]]])
            }
        },
        "index": {
            "theta": jnp.array([[10, 20, 30]]),       # shape (1,3)
            "y":     jnp.array([[100, 200, 300]])
        }
    }

    y = {
        "data": {
            "theta": jnp.array([[7, 8],
                                  [9, 10]]),       # shape (2,2); will be padded to 2x3 with zeros
            "y":     jnp.array([[70, 80],
                                  [90, 100]])
        },
        "labels": {
            "theta": jnp.array([[400, 500]]),         # shape (1,2) -> padded to (1,3)
            "y":     jnp.array([[4000, 5000]])
        },
        "masks": {
            "padding": {
                "theta": jnp.array([[0, 0]]),         # shape (1,2) -> padded to (1,3) with 0
                "y":     jnp.array([[0, 0]])
            },
            "attention": {
                "theta": jnp.array([[[0, 0]]]),         # shape (1,1,2) -> padded to (1,2,2)
                "y":     jnp.array([[[3, 3]]]),
                "cross": jnp.array([[[5]]])             # shape (1,1,1) -> padded to (1,2,2)
            }
        },
        "index": {
            "theta": jnp.array([[40, 50]]),           # shape (1,2) -> padded to (1,3) with pad_value
            "y":     jnp.array([[400, 500]])
        }
    }

    # Expected outputs:

    # 1. Data is concatenated along the sample axis.
    expected_theta_data = jnp.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, pad_value],    # y padded from 2 -> 3 timesteps with pad_value
        [9, 10, pad_value]
    ])
    expected_y_data = jnp.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, pad_value],
        [90, 100, pad_value]
    ])

    # 2. Labels: padded similarly (pad value is explicitly 0 in the call).
    expected_theta_labels = jnp.array([
        [100, 200, 300],
        [100, 200, 300],
        [400, 500, 0],
        [400, 500, 0]
    ])
    expected_y_labels = jnp.array([
        [1000, 2000, 3000],
        [1000, 2000, 3000],
        [4000, 5000, 0],
        [4000, 5000, 0]
    ])

    # 3. Masks -> padding:
    expected_theta_padding = jnp.array([
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_y_padding = jnp.array([
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])

    # 4. Masks -> attention:
    # For "theta": x attention is (1,2,2) → broadcast to (2,2,2); y attention is padded from (1,1,2) to (1,2,2).
    expected_theta_attention = jnp.concatenate([
        jnp.broadcast_to(jnp.array([[[1, 1],
                                       [1, 1]]]), (2, 2, 2)),
        jnp.broadcast_to(jnp.array([[[0, 0],
                                       [0, 0]]]), (2, 2, 2))
    ], axis=0)
    # For "y": similar padding behavior.
    expected_y_attention = jnp.concatenate([
        jnp.broadcast_to(jnp.array([[[2, 2],
                                       [2, 2]]]), (2, 2, 2)),
        jnp.broadcast_to(jnp.array([[[3, 3],
                                       [0, 0]]]), (2, 2, 2))
    ], axis=0)
    # For "cross": y cross is padded from (1,1,1) to (1,2,2).
    expected_cross_attention = jnp.concatenate([
        jnp.broadcast_to(jnp.array([[[1, 1],
                                       [1, 1]]]), (2, 2, 2)),
        jnp.broadcast_to(jnp.array([[[5, 0],
                                       [0, 0]]]), (2, 2, 2))
    ], axis=0)

    # 5. Index: no explicit pad_value is passed so default pad_value is used.
    expected_theta_index = jnp.array([
        [10, 20, 30],
        [10, 20, 30],
        [40, 50, pad_value],
        [40, 50, pad_value]
    ])
    expected_y_index = jnp.array([
        [100, 200, 300],
        [100, 200, 300],
        [400, 500, pad_value],
        [400, 500, pad_value]
    ])

    result = combine_data(x, y, pad_value=pad_value)

    # Verify outputs.
    assert_array_equal(result["data"]["theta"], expected_theta_data)
    assert_array_equal(result["data"]["y"], expected_y_data)
    assert_array_equal(result["labels"]["theta"], expected_theta_labels)
    assert_array_equal(result["labels"]["y"], expected_y_labels)
    assert "masks" in result
    assert_array_equal(result["masks"]["padding"]["theta"], expected_theta_padding)
    assert_array_equal(result["masks"]["padding"]["y"], expected_y_padding)
    assert "attention" in result["masks"]
    assert_array_equal(result["masks"]["attention"]["theta"], expected_theta_attention)
    assert_array_equal(result["masks"]["attention"]["y"], expected_y_attention)
    assert_array_equal(result["masks"]["attention"]["cross"], expected_cross_attention)
    assert_array_equal(result["index"]["theta"], expected_theta_index)
    assert_array_equal(result["index"]["y"], expected_y_index)

def test_combine_data_missing_padding():
    # Test behavior when one of the datasets is missing the "padding" mask for a block.
    x = {
        "data": {
            "theta": jnp.array([[1, 2, 3]]),   # shape (1,3)
            "y":     jnp.array([[10, 20, 30]])
        },
        "labels": {
            "theta": jnp.array([[100, 200, 300]]),
            "y":     jnp.array([[1000, 2000, 3000]])
        },
        "index": {
            "theta": jnp.array([[10, 20, 30]]),
            "y":     jnp.array([[100, 200, 300]])
        }
    }
    y = {
        "data": {
            "theta": jnp.array([[7, 8]]),   # shape (1,2) → padded to (1,3)
            "y":     jnp.array([[70, 80]])   # shape (1,2) → padded to (1,3)
        },
        "labels": {
            "theta": jnp.array([[400, 500]]),  # shape (1,2) → padded to (1,3)
            "y":     jnp.array([[4000, 5000]])
        },
        "index": {
            "theta": jnp.array([[40, 50]]),
            "y":     jnp.array([[400, 500]])
        }
    }

    # For "theta": x padding is provided ([1,1,1]); y default will be zeros of shape (1,2) padded to (1,3).
    expected_theta_padding = jnp.array([
        [1, 1, 1],
        [1, 1, 0]
    ])
    expected_y_padding = jnp.array([
        [1, 1, 1],
        [1, 1, 0]
    ])

    result = combine_data(x, y)
    masks_padding = result["masks"]["padding"]

    assert_array_equal(masks_padding["theta"], expected_theta_padding)
    assert_array_equal(masks_padding["y"], expected_y_padding)

def test_combine_data_missing_index():
    # Here we test that if "index") is missing both inputs,
    # the output does not contain that field.
    x = {
        "data": {
            "theta": jnp.array([[1, 2, 3]]),
            "y":     jnp.array([[10, 20, 30]])
        },
        "labels": {
            "theta": jnp.array([[1, 1, 1]]),
            "y":     jnp.array([[1, 1, 1]])
        }
    }
    y = {
        "data": {
            "theta": jnp.array([[7, 8]]),
            "y":     jnp.array([[70, 80]])
        },
        "labels": {
            "theta": jnp.array([[0, 0]]),
            "y":     jnp.array([[0, 0]])
        }
    }
    result = combine_data(x, y)
    # Since neither x nor y has "index", the output should not contain a "index" key.
    assert "index" not in result
