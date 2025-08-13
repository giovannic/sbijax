# CLAUDE.md

This is a project for Flow Matching for Posterior Estimation, written using JAX.

# Development Commands

- `python -m pytest test/` - Run test suite
- `pyright` - Type checking

The virtualenv is in the `env` directory

## Testing

Tests use pytest and are located in `test/`. The `conftest.py` defines common fixtures for prior/simulator functions using Tensorflow Probability distributions.

At the moment the e2e tests and the MMD test take too long and do not pass. Please only check them if they are related to the task at hand and check them one by one. i.e. `--ignore=test/test_cnf_e2e.py --ignore=test/test_lc2stnf_e2e.py --ignore=test/test_lc2st_e2e.py --ignore=test/test_mmd.py`

## Experimentation

I will ask you to make debugging code to log outputs or make plots. Experiments are throwaway code, do not save code to disk, just run the code and have the outputs saved to disk.

## Coding style

 * Please add type annotations to your function signatures. You can use jax types from `jaxtyping`.
