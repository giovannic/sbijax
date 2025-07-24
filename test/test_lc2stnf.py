import pytest
from jax import numpy as jnp, random as jr, tree
import flax.nnx as nnx
from sfmpe.lc2stnf import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
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
    "dim,d_size,batch_size",
    [
        (8, 100, 10),
        (16, 200, 20)
    ],
)
def test_train_main_classifier_updates_params(dim, d_size, batch_size):
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
    theta_cal = jr.normal(key_theta, (d_size, dim))
    x_cal = jr.normal(key_x, (d_size, dim))
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
        batch_size=batch_size
    )

    # Check that at least one parameter has changed
    leaves_before = tree.leaves(initial_params)
    leaves_after = tree.leaves(nnx.state(classifier))
    params_changed = any(
        not jnp.allclose(b, a) for b, a in zip(leaves_before, leaves_after)
    )
    assert params_changed, "Expected classifier parameters to update after training"

@pytest.mark.parametrize(
    "dim,d_size,batch_size,num_classifiers",
    [
        (8, 100, 10, 3),  # 3 null classifiers
        (16, 200, 10, 5)  # 5 null classifiers
    ],
)
def test_precompute_null_distribution(dim, d_size, batch_size, num_classifiers):
    """
    Test that precompute_null_distribution_nf_classifiers trains multiple null classifiers
    with fresh RNGs and results differ.
    """
    key = jr.PRNGKey(0)
    # Split key for each classifier

    # Initialize null classifier
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim * 2,
        n_layers=2,
        activation=nnx.relu,
        n=num_classifiers,
        rngs=nnx.Rngs(0)
    )

    # Create calibration data
    key_theta, key_x = jr.split(key)
    theta_cal = jr.normal(key_theta, (d_size, dim))
    x_cal = jr.normal(key_x, (d_size, dim))
    calibration_data = (theta_cal, x_cal)

    pre_train_params = nnx.state(null_classifier)
    assert all(
        tree.leaves(
            tree.map(lambda leaf: leaf.shape[0] == num_classifiers, pre_train_params)
        )
    )
    delta = 1e-5

    # Compare first vs others to ensure not all identical
    identical = all(
        tree.leaves(
            tree.map(
                lambda leaf: jnp.all(jnp.diff(leaf, axis=0) < delta),
                pre_train_params
            )
        )
    )
    assert not identical, "Expected at least two null classifiers to have different parameters"

    # Precompute null distribution classifiers
    precompute_null_distribution_nf_classifiers(
        rng_key=key,
        calibration_data=calibration_data,
        classifiers=null_classifier,
        num_epochs=1,
        batch_size=batch_size
    )

    post_train_params = nnx.state(null_classifier)

    changed = tree.map(
        lambda leaf1, leaf2: jnp.any(jnp.abs(leaf1 - leaf2) > delta),
        pre_train_params,
        post_train_params
    )
    print(tree.map(lambda x, y: x - y, post_train_params, pre_train_params))
    assert all(tree.leaves(changed)), "Expected all classifiers to be updated"
    # Compare first vs others to ensure not all identical
    identical = all(
        tree.leaves(
            tree.map(
                lambda leaf: jnp.all(jnp.diff(leaf, axis=0) < delta),
                post_train_params
            )
        )
    )
    assert not identical, "Expected at least two null classifiers to have different parameters"

@pytest.mark.parametrize(
    "dim,Nv",
    [
        (8, 10),
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
    rngs = nnx.Rngs(0)
    null_clf = MultiBinaryMLPClassifier(
        dim=dim*2,
        n_layers=2,
        activation=nnx.relu,
        n=num_null,
        rngs=rngs,
    )
    # Evaluate
    _, t_stat, p_val = evaluate_l_c2st_nf(
        rng_key=key,
        xo=xo,
        main_classifier=main_clf,
        null_classifier=null_clf,
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

@pytest.mark.parametrize(
    "dim,n",
    [
        (8, 3),
        (16, 5),
    ],
)
def test_multi_classifier_shapes(dim, n):
    """
    Test that the multi classifier initialises parameters with the correct shapes
    and when called outputs probabilities with the correct shape.
    """
    # Initialize null classifier
    cls = MultiBinaryMLPClassifier(
        dim=dim * 2,
        n_layers=2,
        activation=nnx.relu,
        n=n,
        rngs=nnx.Rngs(0)
    )

    state = nnx.state(cls)

    assert all(
        tree.leaves(
            tree.map(lambda leaf: leaf.shape[0] == n, state)
        )
    )

    # Create input data
    x = jnp.zeros((100, dim))
    z = jnp.zeros((100, dim))

    prob = cls(z, x)
    assert prob.shape == (n, 100)

