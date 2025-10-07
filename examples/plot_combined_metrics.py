"""
Plot combined metrics from multiple Hydra multirun directories.

This script scans a base directory containing timestamped multirun directories,
automatically detects whether each run sweeps over sample budget or dimension,
and creates a single figure with subplots organized by task (rows) and sweep
type (columns).
"""

import argparse
from pathlib import Path
from plotting_utils import (
    scan_multirun_directories,
    plot_combined_metrics
)


def main() -> None:
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot combined LC2ST metrics from multiple Hydra multiruns"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Base directory containing timestamped multirun directories"
        )
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("examples/outputs/lc2st_combined.png"),
        help=(
            "Path to save the combined plot "
            "(default: examples/outputs/lc2st_combined.png)"
        )
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
        "--metric",
        type=str,
        choices=['lc2st', 'cnf_log_prob', 'kl_divergence'],
        default='lc2st',
        help=(
            "Metric to plot (default: lc2st). "
            "Options: lc2st, cnf_log_prob, kl_divergence"
        )
    )

    args = parser.parse_args()

    # Scan and categorize multirun directories
    print(f"Scanning {args.input_dir} for multirun directories...")
    categorized = scan_multirun_directories(args.input_dir)

    if not categorized:
        print("No valid multirun directories found.")
        return

    # Print summary
    print("\nFound multirun directories:")
    for (sweep_type, task), dirs in categorized.items():
        print(f"  {task} ({sweep_type}): {len(dirs)} multirun(s)")
        for d in dirs:
            print(f"    - {d.name}")

    # Build filters
    filters = {}
    if args.n_rounds is not None:
        filters['n_rounds'] = args.n_rounds
    if args.n_epochs is not None:
        filters['n_epochs'] = args.n_epochs

    if filters:
        print(f"\nApplying filters: {filters}")

    # Create combined plot
    print("\nCreating combined plot...")
    plot_combined_metrics(categorized, args.save_path, filters, args.metric)


if __name__ == "__main__":
    main()
