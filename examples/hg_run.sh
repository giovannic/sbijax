#!/bin/bash

#run for n_simulations in 1000..10000 in steps of 1000

for n_simulations in $(seq 1000 1000 10000)
do
  python examples/hierarchical_gaussian.py --n_simulations $n_simulations --n_rounds 1 --n_epochs 1000
done
