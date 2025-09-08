"""
Plot LC2ST metrics from runs with different parameter values.

This script scans Hydra output directories for results from different parameter
values (n_theta, n_sites, etc.) and creates a comparison plot showing how the 
LC2ST statistic varies with the target parameter for different estimation methods.
"""

import argparse
from pathlib import Path
from typing import Optional
from plotting_utils import collect_metrics, plot_metrics




def main() -> None:
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot LC2ST metrics vs target parameter from Hydra runs"
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
        help="Path to save the plot (auto-generated if not specified)"
    )
    parser.add_argument(
        "--target-variable",
        type=str,
        default="n_theta",
        help="Variable to plot on x-axis (default: n_theta)"
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        help="Filter results by number of simulations"
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
    
    # Parameter configuration
    PARAM_CONFIG = {
        'n_theta': {
            'x_label': 'Number of Theta Parameters (n_theta)',
            'title': 'LC2ST Statistic vs n_theta',
            'default_filters': {'n_simulations': 5000}
        },
        'n_sites': {
            'x_label': 'Number of Sites (n_sites)',
            'title': 'LC2ST Statistic vs n_sites', 
            'default_filters': {'n_simulations': 5000}
        }
    }
    
    # Get configuration for target variable
    param_config = PARAM_CONFIG.get(args.target_variable, {
        'x_label': f'{args.target_variable}',
        'title': f'LC2ST Statistic vs {args.target_variable}',
        'default_filters': {}
    })
    
    # Set default save path if not provided
    if args.save_path is None:
        args.save_path = Path(f"examples/outputs/lc2st_{args.target_variable}_comparison.png")
    
    # Collect metrics
    print(f"Collecting metrics from Hydra runs for {args.target_variable}...")
    filters = param_config['default_filters'].copy()
    if args.n_simulations is not None:
        filters['n_simulations'] = args.n_simulations
    if args.n_rounds is not None:
        filters['n_rounds'] = args.n_rounds
    if args.n_epochs is not None:
        filters['n_epochs'] = args.n_epochs
    
    metrics = collect_metrics(
        args.output_dir, 
        target_variable=args.target_variable,
        filters=filters
    )
    
    if not metrics:
        print(f"No metrics found for {args.target_variable}.")
        if filters:
            print(f"Applied filters: {filters}")
        return
    
    # Print summary
    print("\nFound metrics for:")
    for (task, method), data in metrics.items():
        param_values = [x[0] for x in data]
        total_runs = sum(len(seed_stats) for _, seed_stats, _ in data)
        print(f"  {task} {method}: {len(data)} parameter sets, "
              f"{total_runs} total runs, {args.target_variable} values: {min(param_values)}-{max(param_values)}")
    
    # Create plot
    print(f"\nCreating plot...")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(
        metrics, 
        args.save_path,
        x_label=param_config['x_label'],
        title=param_config['title']
    )


if __name__ == "__main__":
    main()