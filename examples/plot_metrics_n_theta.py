"""
Plot LC2ST metrics from hierarchical Gaussian runs with different n_theta values.

This script scans Hydra output directories for results from different numbers
of theta parameters and creates a comparison plot showing how the LC2ST statistic 
varies with n_theta for different estimation methods.
"""

import argparse
from pathlib import Path
from typing import Optional
from plotting_utils import collect_metrics, plot_metrics




def main() -> None:
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot LC2ST metrics vs n_theta from hierarchical Gaussian Hydra runs"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("outputs"),
        help="Base output directory containing Hydra runs (default: outputs)"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("examples/outputs/lc2st_n_theta_comparison.png"),
        help="Path to save the plot (default: examples/outputs/lc2st_n_theta_comparison.png)"
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=5000,
        help="Filter results by number of simulations (default: 5000)"
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        help="Filter results by number of rounds"
    )
    parser.add_argument(
        "--n-epochs",
        type=int, 
        help="Filter results by number of epochs"
    )
    
    args = parser.parse_args()
    
    # Collect metrics
    print("Collecting metrics from Hydra runs...")
    filters = {}
    if args.n_simulations is not None:
        filters['n_simulations'] = args.n_simulations
    if args.n_rounds is not None:
        filters['n_rounds'] = args.n_rounds
    if args.n_epochs is not None:
        filters['n_epochs'] = args.n_epochs
    
    metrics = collect_metrics(
        args.output_dir, 
        target_variable='n_theta',
        filters=filters
    )
    
    if not metrics:
        print("No metrics found. Make sure to run hierarchical_gaussian.py first.")
        print(f"Looking for runs with n_simulations={args.n_simulations}")
        return
    
    # Print summary
    print("\nFound metrics for:")
    for method_name, data in metrics.items():
        n_thetas = [x[0] for x in data]
        print(f"  {method_name}: {len(data)} runs, "
              f"n_theta values: {min(n_thetas)}-{max(n_thetas)}")
    
    # Create plot
    print(f"\nCreating plot...")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(
        metrics, 
        args.save_path,
        x_label='Number of Theta Parameters (n_theta)',
        title='LC2ST Statistic vs n_theta by Method'
    )


if __name__ == "__main__":
    main()