# CLAUDE.md

## Development Commands

- `python -m pytest test/` - Run test suite
- `pyright` - Type checking

## Core Architecture

### Main Components

- **`sfmpe/fmpe.py`** - Core flow matching posterior estimation implementation
- **`sfmpe/train.py`** - Training utilities and fit_model function
- **`sfmpe/nn/`** - Neural network architectures:
  - `make_continuous_flow.py` - Continuous normalizing flows
  - `mlp.py` - Multi-layer perceptrons
  - `transformer/` - Transformer-based architectures with attention mechanisms
- **`sfmpe/util/`** - Utilities for data handling, early stopping, and type definitions
- **`sfmpe/_ne_base.py`** and **`sfmpe/_sbi_base.py`** - Base classes for neural estimation and simulation-based inference

### Testing

Tests use pytest and are located in `test/`. The `conftest.py` defines common fixtures for prior/simulator functions using TensorFlow Probability distributions.

Please look at my global preferences for how we should work @~/.claude/CLAUDE.md
