"""
Plot LC2ST metrics from hierarchical Gaussian runs with different sample sizes.

This script scans the outputs directory for results from different simulation 
budgets and creates a comparison plot showing how the LC2ST statistic varies 
with the number of simulations for different estimation methods.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


def parse_directory_name(dir_name: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse directory name to extract n_simulations, n_rounds, n_epochs.
    
    Args:
        dir_name: Directory name like "1000sims_2rounds_1000epochs"
        
    Returns:
        Tuple of (n_simulations, n_rounds, n_epochs) or None if parse fails
    """
    pattern = r'(\d+)sims_(\d+)rounds_(\d+)epochs'
    match = re.match(pattern, dir_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def collect_metrics(
    base_dir: Path, 
    n_rounds_filter: Optional[int] = None,
    n_epochs_filter: Optional[int] = None
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Collect LC2ST metrics from all runs.
    
    Args:
        base_dir: Base directory containing outputs
        n_rounds_filter: Only include runs with this number of rounds
        n_epochs_filter: Only include runs with this number of epochs
        
    Returns:
        Dictionary mapping method names to list of (n_simulations, main_stat)
    """
    metrics = {}
    
    # Scan for hierarchical_gaussian results
    hg_dir = base_dir / 'hierarchical_gaussian'
    if not hg_dir.exists():
        print(f"Warning: {hg_dir} does not exist")
        return metrics
    
    for run_dir in hg_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        # Parse directory name
        parsed = parse_directory_name(run_dir.name)
        if parsed is None:
            continue
            
        n_simulations, n_rounds, n_epochs = parsed
        
        # Apply filters if specified
        if n_rounds_filter is not None and n_rounds != n_rounds_filter:
            continue
        if n_epochs_filter is not None and n_epochs != n_epochs_filter:
            continue
        
        # Check for method subdirectories
        for method_dir in run_dir.iterdir():
            if not method_dir.is_dir():
                continue
                
            method_name = method_dir.name.upper()  # SFMPE or FMPE
            stats_file = method_dir / 'stats.json'
            
            if not stats_file.exists():
                continue
                
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    
                main_stat = stats.get('main_stat')
                if main_stat is not None:
                    if method_name not in metrics:
                        metrics[method_name] = []
                    metrics[method_name].append((n_simulations, main_stat))
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse {stats_file}: {e}")
                continue
    
    # Sort by n_simulations
    for method_name in metrics:
        metrics[method_name].sort(key=lambda x: x[0])
    
    return metrics


def plot_metrics(
    metrics: Dict[str, List[Tuple[int, float]]], 
    output_path: Path
) -> None:
    """
    Create a plot comparing LC2ST statistics across simulation budgets.
    
    Args:
        metrics: Dictionary mapping method names to (n_simulations, main_stat) 
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method_name, data) in enumerate(metrics.items()):
        if not data:
            continue
            
        n_sims, stats = zip(*data)
        color = colors[i % len(colors)]
        
        plt.plot(n_sims, stats, 'o-', label=method_name, color=color, 
                linewidth=2, markersize=6)
    
    plt.xlabel('Number of Simulations')
    plt.ylabel('LC2ST Statistic')
    plt.title('LC2ST Statistic vs Simulation Budget by Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    if metrics:
        all_stats = [stat for data in metrics.values() for _, stat in data]
        if all_stats:
            plt.ylim(0, max(all_stats) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


def main() -> None:
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot LC2ST metrics from hierarchical Gaussian runs"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("examples/outputs"),
        help="Base output directory (default: examples/outputs)"
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
    
    args = parser.parse_args()
    
    # Collect metrics
    print("Collecting metrics...")
    metrics = collect_metrics(
        args.output_dir, 
        n_rounds_filter=args.n_rounds,
        n_epochs_filter=args.n_epochs
    )
    
    if not metrics:
        print("No metrics found. Make sure to run hierarchical_gaussian.py first.")
        return
    
    # Print summary
    print("\nFound metrics for:")
    for method_name, data in metrics.items():
        n_sims = [x[0] for x in data]
        print(f"  {method_name}: {len(data)} runs, "
              f"simulations: {min(n_sims)}-{max(n_sims)}")
    
    # Create plot
    print(f"\nCreating plot...")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(metrics, args.save_path)


if __name__ == "__main__":
    main()