#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/env/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected virtualenv python at $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p /tmp/matplotlib

JAX_ENABLE_X64=1 \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
MPLCONFIGDIR=/tmp/matplotlib \
PYTHONPATH="$ROOT_DIR" \
"$PYTHON_BIN" "$ROOT_DIR/examples/seir_mcmc.py" \
  n_simulations=200 \
  n_post_samples=100 \
  mcmc.n_chains=4 \
  mcmc.sampler=nuts_tfp \
  n_obs=10 \
  n_timesteps=730 \
  n_warmup=730 \
  n_sites=10 \
  mcmc.step_size=1e-1 \
  mcmc.use_numpyro_model=true \
  mcmc.init_to_truth=true
