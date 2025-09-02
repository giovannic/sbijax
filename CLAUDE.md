# CLAUDE.md

This is a project for Flow Matching for Posterior Estimation, written using JAX.

# Development Commands

- `python -m pytest test/` - Run test suite
- `pyright` - Type checking

The virtualenv is in the `env` directory

## Testing

Tests use pytest and are located in `test/`. The `conftest.py` defines common fixtures for prior/simulator functions using Tensorflow Probability distributions.

Tests use custom markers to categorize different types of tests:

- **Default tests**: Core functionality tests that run quickly and should always pass
- **Slow tests**: Tests that take longer to run (marked with `@pytest.mark.slow`)
- **Flow diagnostic tests**: Tests for validating flow-based models (marked with `@pytest.mark.flow_diagnostics`)

By default, slow and flow diagnostic tests are excluded to keep the development feedback loop fast. Use these commands to run specific test categories:

```bash
# Run all tests including slow ones
python -m pytest test/ -m "slow or not slow"

# Run only slow tests
python -m pytest test/ -m "slow"

# Run only flow diagnostic tests
python -m pytest test/ -m "flow_diagnostics"

# Run all tests (including slow and diagnostic)
python -m pytest test/ -m ""
```

## Experimentation

I will ask you to make debugging code to log outputs or make plots. Experiments are throwaway code, do not save code to disk, just run the code and have the outputs saved to disk.

## Coding style

 * Please add type annotations to your function signatures. You can use jax types from `jaxtyping`.
