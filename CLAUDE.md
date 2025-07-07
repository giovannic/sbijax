# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SFMPE (Structured Flow Matching for Posterior Estimation) is a Python machine learning library for Bayesian inference using neural flow matching. The project implements flow-based posterior estimation methods built on JAX/Flax with TensorFlow Probability.

## Development Commands

- `python -m pytest test/` - Run test suite
- `pyright` - Type checking (dev dependency)

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

### Key Dependencies

- **JAX/Flax** - Primary ML framework for neural networks
- **TensorFlow Probability** - Probabilistic programming and distributions
- **sbijax** - Simulation-based inference utilities
- **Optax** - Optimization library

### Testing

Tests use pytest and are located in `test/`. The `conftest.py` defines common fixtures for prior/simulator functions using TensorFlow Probability distributions.

## Development Notes

The project implements structured flow matching with cross-attention mechanisms for handling hierarchical parameter structures. Key patterns include:
- Flow matching with time-dependent sampling (`_sample_theta_t`, `_ut` functions)
- Cross-attention masking for structured data (`cross_2d_masks`, `make_cross_attention_mask`)
- Integration with sbijax for simulation-based inference workflows