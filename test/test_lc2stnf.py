import pytest
from jax import numpy as jnp, random as jr, tree
import flax.nnx as nnx
from sfmpe.lc2stnf import (
    BinaryMLPClassifier,
    train_l_c2st_nf_main_classifier,
    precompute_null_distribution_nf_classifiers,
    evaluate_l_c2st_nf
)

@pytest.mark.parametrize(
    "dim,n_layers,activation,batch_size",
    [
        (8, 1, nnx.relu, None),      # unbatched input
        (8, 1, nnx.relu, 4),         # small batch
        (16, 2, nnx.tanh, 8),        # larger model and batch
    ],
)
def test_binary_mlp_shape(dim, n_layers, activation, batch_size):
    """
    Test that BinaryMLPClassifier __call__ accepts both unbatched and batched inputs
    and returns an array of shape matching the input batch dimensions.
    """
    # Initialize RNG and model
    key = jr.PRNGKey(0)
    model = BinaryMLPClassifier(
        dim=dim * 2,
        n_layers=n_layers,
        activation=activation,
        rngs=nnx.Rngs(key),
    )

    # Generate test inputs
    key_z, key_x = jr.split(key)
    if batch_size is None:
        shape = (dim,)
    else:
        shape = (batch_size, dim)
    z = jr.normal(key_z, shape)
    x = jr.normal(key_x, shape)

    # Forward pass
    out = model(z, x)

    # Assertions
    assert isinstance(out, jnp.ndarray), "Output must be a JAX array"
    # Expect one output per sample: output.shape == input.shape[:-1]
    assert out.shape == z.shape[:-1], (
        f"Expected output shape {z.shape[:-1]}, got {out.shape}"
    )
    # Check output dtype is floating-point
    assert jnp.issubdtype(out.dtype, jnp.floating), (
        f"Expected floating dtype, got {out.dtype}"
    )

@pytest.mark.parametrize(
    "dim,batch_size",
    [
        (8, 100),  # 100 samples, single batch
        (16, 200)  # 200 samples
    ],
)
def test_train_main_classifier_updates_params(dim, batch_size):
    """
    Test that train_l_c2st_nf_main_classifier runs for 1 epoch and updates classifier parameters.
    """
    # Setup
    key = jr.PRNGKey(42)
    # Instantiate classifier
    classifier = BinaryMLPClassifier(
        dim=dim * 2,
        n_layers=2,
        activation=nnx.relu,
        rngs=nnx.Rngs(key),
    )
    # Generate calibration data
    key_theta, key_x = jr.split(key)
    theta_cal = jr.normal(key_theta, (batch_size, dim))
    x_cal = jr.normal(key_x, (batch_size, dim))
    calibration_data = (theta_cal, x_cal)

    # Inverse transform for testing
    inverse_transform = lambda theta, _: theta

    # Snapshot initial parameters
    initial_params = nnx.state(classifier)

    # Train for 1 epoch
    train_l_c2st_nf_main_classifier(
        rng_key=key,
        classifier=classifier,
        calibration_data=calibration_data,
        inverse_transform=inverse_transform,
        num_epochs=1,
    )

    # Check that at least one parameter has changed
    leaves_before = tree.leaves(initial_params)
    leaves_after = tree.leaves(nnx.state(classifier))
    params_changed = any(
        not jnp.allclose(b, a) for b, a in zip(leaves_before, leaves_after)
    )
    assert params_changed, "Expected classifier parameters to update after training"

@pytest.mark.parametrize(
    "dim,batch_size,num_classifiers",
    [
        (8, 100, 3),  # 3 null classifiers
        (16, 200, 5)  # 5 null classifiers
    ],
)
def test_precompute_null_distribution(dim, batch_size, num_classifiers):
    """
    Test that precompute_null_distribution_nf_classifiers trains multiple null classifiers
    with fresh RNGs and results differ.
    """
    key = jr.PRNGKey(0)
    # Split key for each classifier
    keys = jr.split(key, num_classifiers)
    # Initialize null classifiers
    null_cls_list = [
        BinaryMLPClassifier(
            dim=dim * 2,
            n_layers=2,
            activation=nnx.relu,
            rngs=nnx.Rngs(k),
        )
        for k in keys
    ]

    # Create calibration data
    key_theta, key_x = jr.split(key)
    theta_cal = jr.normal(key_theta, (batch_size, dim))
    x_cal = jr.normal(key_x, (batch_size, dim))
    calibration_data = (theta_cal, x_cal)

    # Precompute null distribution classifiers
    precompute_null_distribution_nf_classifiers(
        rng_key=key,
        calibration_data=calibration_data,
        classifiers=null_cls_list,
        num_epochs=1,
    )

    # Verify each has params and at least two differ
    all_params = [tree.leaves(nnx.state(c)) for c in null_cls_list]
    # Compare first vs others to ensure not all identical
    identical = all(
        all(jnp.allclose(p0, p1) for p0, p1 in zip(all_params[0], params))
        for params in all_params[1:]
    )
    assert not identical, "Expected at least two null classifiers to have different parameters"

@pytest.mark.parametrize(
    "dim,Nv",
    [
        (8, 10),  # small dim and validation size
        (16, 20),
    ],
)
def test_evaluate_l_c2st_nf_output(dim, Nv):
    """
    Test evaluate_l_c2st_nf returns proper-shaped, positive statistics for untrained classifiers.
    """
    key = jr.PRNGKey(123)
    # Observation xo
    xo = jr.normal(key, (dim,))

    # Main classifier
    key_main = jr.PRNGKey(1)
    main_clf = BinaryMLPClassifier(
        dim=dim * 2,
        n_layers=2,
        activation=nnx.relu,
        rngs=nnx.Rngs(key_main),
    )

    # Null classifiers
    num_null = 3
    keys_null = jr.split(key, num_null)
    null_clfs = [
        BinaryMLPClassifier(
            dim=dim * 2,
            n_layers=2,
            activation=nnx.relu,
            rngs=nnx.Rngs(k),
        )
        for k in keys_null
    ]

    # Evaluate
    t_stat, p_val = evaluate_l_c2st_nf(
        rng_key=key,
        xo=xo,
        main_classifier=main_clf,
        null_classifiers=null_clfs,
        latent_dim=dim,
        Nv=Nv,
    )

    # Assertions
    for arr in (t_stat, p_val):
        assert isinstance(arr, jnp.ndarray), "Expected JAX array"
        assert arr.shape == (), (
            f"Expected shape (), got {arr.shape}"
        )
        assert arr >= 0, "Expected non-negative values"
