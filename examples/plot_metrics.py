"""
Plot LC2ST metrics from hierarchical Gaussian runs with different sample sizes.

This script scans Hydra output directories for results from different simulation 
budgets and creates a comparison plot showing how the LC2ST statistic varies 
with the number of simulations for different estimation methods.
"""

import argparse
from pathlib import Path
from typing import Optional
from plotting_utils import collect_metrics, plot_metrics




def main() -> None:
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot LC2ST metrics from hierarchical Gaussian Hydra runs"
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
        default=Path("examples/outputs/lc2st_comparison.png"),
        help="Path to save the plot (default: examples/outputs/lc2st_comparison.png)"
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
    parser.add_argument(
        "--task-filter",
        type=str,
        help="Filter results by task type (Gaussian, Brownian, SEIR)"
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        help="Filter results by number of theta parameters (for Gaussian/Brownian tasks)"
    )
    parser.add_argument(
        "--n-sites",
        type=int,
        help="Filter results by number of sites (for SEIR tasks)"
    )
    
    args = parser.parse_args()
    
    # Collect metrics
    print("Collecting metrics from Hydra runs...")
    filters = {}
    if args.n_rounds is not None:
        filters['n_rounds'] = args.n_rounds
    if args.n_epochs is not None:
        filters['n_epochs'] = args.n_epochs
    if args.n_theta is not None:
        filters['n_theta'] = args.n_theta
    if args.n_sites is not None:
        filters['n_sites'] = args.n_sites
    
    metrics = collect_metrics(
        args.output_dir, 
        target_variable='n_simulations',
        filters=filters
    )
    
    if not metrics:
        print("No metrics found.")
        if args.task_filter:
            print(f"No data found for task filter: {args.task_filter}")
        return
    
    # Apply task filter if specified
    if args.task_filter:
        metrics = {k: v for k, v in metrics.items() if k[0].lower() == args.task_filter.lower()}
    
    # Print summary
    print("\nFound metrics for:")
    for (task, method), data in metrics.items():
        n_sims = [x[0] for x in data]
        total_runs = sum(len(seed_stats) for _, seed_stats, _ in data)
        print(f"  {task} {method}: {len(data)} parameter sets, "
              f"{total_runs} total runs, simulations: {min(n_sims)}-{max(n_sims)}")
    
    # Create plot
    print(f"\nCreating plot...")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(
        metrics, 
        args.save_path,
        x_label='Number of Simulations',
        title='LC2ST Statistic vs Simulation Budget'
    )


if __name__ == "__main__":
    main()
