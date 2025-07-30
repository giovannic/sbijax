import pytest
from jax import numpy as jnp, random as jr, tree
import flax.nnx as nnx
from sfmpe.c2stnf import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    train_c2st_nf_main_classifier,
    precompute_c2st_nf_null_classifiers,
    evaluate_c2st_nf
)

@pytest.mark.parametrize(
    "dim,n_layers,activation,batch_size,latent_dim",
    [
        (8, 1, nnx.relu, None, 16),      # unbatched input
        (8, 1, nnx.relu, 4, 16),         # small batch
        (16, 2, nnx.tanh, 8, 16),        # larger model and batch
    ],
)
def test_binary_mlp_z_only_shape(dim, n_layers, activation, batch_size, latent_dim):
    """
    Test that BinaryMLPClassifier __call__ accepts z-only inputs
    and returns an array of shape matching the input batch dimensions.
    """
    # Initialize RNG and model - note dim instead of dim*2 for z-only
    key = jr.PRNGKey(0)
    model = BinaryMLPClassifier(
        dim=dim,  # Only z dimension, no x
        n_layers=n_layers,
        activation=activation,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(key),
    )

    # Generate test inputs (z only)
    if batch_size is None:
        shape = (dim,)
    else:
        shape = (batch_size, dim)
    z = jr.normal(key, shape)

    # Forward pass with z only (u = z)
    out = model(z)

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
    "dim,d_size,batch_size,latent_dim",
    [
        (8, 100, 10, 16),
        (16, 200, 20, 16)
    ],
)
def test_train_main_classifier_updates_params(dim, d_size, batch_size, latent_dim):
    """
    Test that train_c2st_nf_main_classifier runs for 1 epoch and updates classifier parameters.
    Uses z samples to differentiate from normally distributed data.
    """
    # Setup
    key = jr.PRNGKey(42)
    # Instantiate classifier for z-only input
    classifier = BinaryMLPClassifier(
        dim=dim,  # Only z dimension
        n_layers=2,
        activation=nnx.relu,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(key),
    )
    # Generate z samples directly (no theta, x pairs)
    key_z = jr.split(key, 2)[0]
    z_samples = jr.normal(key_z, (d_size, dim))

    # Snapshot initial parameters
    initial_params = nnx.state(classifier)

    # Train for 1 epoch with z samples directly
    train_c2st_nf_main_classifier(
        rng_key=key,
        classifier=classifier,
        z_samples=z_samples,
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
    "dim,d_size,batch_size,num_classifiers,latent_dim",
    [
        (8, 100, 10, 3, 16),  # 3 null classifiers
        (16, 200, 10, 5, 16)  # 5 null classifiers
    ],
)
def test_precompute_null_distribution_c2stnf(dim, d_size, batch_size, num_classifiers, latent_dim):
    """
    Test that precompute_c2st_nf_null_classifiers trains multiple null classifiers
    with fresh RNGs and results differ (z-only version).
    """
    key = jr.PRNGKey(0)

    # Initialize null classifier for z-only input
    null_classifier = MultiBinaryMLPClassifier(
        dim=dim,  # Only z dimension, no x
        n_layers=2,
        activation=nnx.relu,
        n=num_classifiers,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0)
    )

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
    precompute_c2st_nf_null_classifiers(
        rng_key=key,
        classifiers=null_classifier,
        latent_dim=dim,
        N_cal=d_size,
        num_epochs=1,
        batch_size=batch_size
    )

    post_train_params = nnx.state(null_classifier)

    # Check parameters are same shape
    assert all(
        tree.leaves(
            tree.map(
                lambda leaf1, leaf2: leaf1.shape == leaf2.shape,
                pre_train_params,
                post_train_params
            )
        )
    )

    # Check parameters have updated
    changed = tree.map(
        lambda leaf1, leaf2: jnp.any(jnp.abs(leaf1 - leaf2) > delta),
        pre_train_params,
        post_train_params
    )
    assert all(tree.leaves(changed)), "Expected all classifiers to be updated"

    # And in different ways
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
    "dim,n,latent_dim",
    [
        (8, 3, 16),
        (16, 5, 16),
    ],
)
def test_multi_classifier_z_only_shapes(dim, n, latent_dim):
    """
    Test that the multi classifier initialises parameters with the correct shapes
    and when called outputs probabilities with the correct shape for z-only inputs.
    """
    # Initialize null classifier for z-only input
    cls = MultiBinaryMLPClassifier(
        dim=dim,  # Only z dimension, no x
        n_layers=2,
        activation=nnx.relu,
        n=n,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0)
    )

    state = nnx.state(cls)

    assert all(
        tree.leaves(
            tree.map(lambda leaf: leaf.shape[0] == n, state)
        )
    )

    # Create z-only input data
    z = jnp.zeros((100, dim))

    prob = cls(z)
    assert prob.shape == (n, 100)

@pytest.mark.parametrize(
    "dim,Nv,num_null,latent_dim",
    [
        (8, 10, 5, 16),
        (16, 20, 10, 16),
    ],
)
def test_evaluate_c2st_nf_output(dim, Nv, num_null, latent_dim):
    """
    Test evaluate_c2st_nf returns proper-shaped, positive statistics for untrained classifiers.
    Uses z samples from estimated posterior and null classifiers.
    """
    key = jr.PRNGKey(123)
    
    # Main classifier for z-only input
    key_main = jr.PRNGKey(1)
    main_clf = BinaryMLPClassifier(
        dim=dim,  # Only z dimension
        n_layers=2,
        activation=nnx.relu,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(key_main),
    )
    
    # Null classifiers for z-only input
    null_clf = MultiBinaryMLPClassifier(
        dim=dim,  # Only z dimension
        n_layers=2,
        activation=nnx.relu,
        n=num_null,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0),
    )
    
    # Generate z samples from estimated posterior
    z_posterior_samples = jr.normal(key, (100, dim))
    
    # Evaluate
    null_stats, t_stat, p_val = evaluate_c2st_nf(
        rng_key=key,
        z_posterior_samples=z_posterior_samples,
        main_classifier=main_clf,
        null_classifier=null_clf,
        latent_dim=dim,
        Nv=Nv,
    )

    # Assertions
    assert isinstance(null_stats, jnp.ndarray), "Expected JAX array for null_stats"
    assert null_stats.shape == (num_null,), f"Expected shape ({num_null},), got {null_stats.shape}"
    
    for arr in (t_stat, p_val):
        assert isinstance(arr, jnp.ndarray), "Expected JAX array"
        assert arr.shape == (), (
            f"Expected shape (), got {arr.shape}"
        )
        assert arr >= 0, "Expected non-negative values"

