# SFMPE

Structured flow matching for posterior estimation.

Under development. More to come...

## Running Experiments

### Hierarchical Gaussian Example

The hierarchical Gaussian example uses Hydra for configuration management:

#### Basic run with default parameters:
```bash
python examples/hierarchical_gaussian.py
```

#### Override specific parameters:
```bash
python examples/hierarchical_gaussian.py n_simulations=5000 n_epochs=500
```

#### Parameter sweep (equivalent to the old hg_run.sh):
```bash
python examples/hierarchical_gaussian.py -m n_simulations=1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 n_rounds=1 n_epochs=1000
```

#### Plot results:
```bash
python examples/plot_metrics.py --output-dir outputs
```

Configuration files are located in `examples/conf/` and can be customized as needed.
