"""
Shared utilities for plotting metrics from Hydra runs.

This module contains common functions used by plotting scripts to load
configurations, collect metrics, and create plots from Hydra output directories.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Literal
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import re


def load_config_from_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load configuration from Hydra's .hydra directory.
    
    Args:
        run_dir: Directory containing a Hydra run output
        
    Returns:
        Configuration dictionary or None if not found
    """
    config_path = run_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        try:
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    return None


def collect_metrics(
    base_dir: Path, 
    target_variable: str,
    filters: Optional[Dict[str, int]] = None
) -> Dict[Tuple[str, str], List[Tuple[int, List[float], List[List[float]]]]]:
    """
    Collect LC2ST metrics from all Hydra runs, grouped by task and method.
    
    Args:
        base_dir: Base directory containing Hydra outputs
        target_variable: Variable to use as x-axis (e.g., 'n_simulations', 'n_theta')
        filters: Dictionary of config filters to apply (e.g., {'n_rounds': 10})
        
    Returns:
        Dictionary mapping (task, method) tuples to list of (target_value, [seed_stats], [null_stats_per_seed])
        where task is 'Gaussian', 'Brownian', or 'SEIR' and method is 'FMPE', 'TFMPE (prior)', or 'TFMPE (observed)'
    """
    if filters is None:
        filters = {}
        
    # Store data grouped by (task, method, target_value)
    raw_data = {}  # (task, method, target_value) -> [(main_stat, null_stats), ...]
    # Final metrics grouped by (task, method)
    metrics = {}
    
    # Scan for Hydra output directories (recursively look for .hydra subdirectories)
    def find_hydra_runs(directory: Path):
        hydra_runs = []
        if directory.is_dir():
            # Check if this directory contains .hydra
            if (directory / ".hydra").exists():
                hydra_runs.append(directory)
            else:
                # Recursively search subdirectories (up to 2 levels deep to avoid infinite recursion)
                for subdir in directory.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        hydra_runs.extend(find_hydra_runs(subdir))
        return hydra_runs
    
    all_run_dirs = find_hydra_runs(base_dir)
    
    for run_dir in all_run_dirs:
            
        # Load configuration
        config = load_config_from_run(run_dir)
        if config is None:
            continue
            
        # Get target variable value
        target_value = config.get(target_variable)
        if target_value is None:
            continue
        
        # Get seed value
        seed = config.get('seed', 0)
        
        # Detect task type from run directory or config
        task = detect_task_type(run_dir, config)
        if task is None:
            continue
            
        # Apply filters
        skip_run = False
        for filter_key, filter_value in filters.items():
            if config.get(filter_key) != filter_value:
                skip_run = True
                break
        if skip_run:
            continue
        
        # Check for method subdirectories (sfmpe, fmpe)
        for method_dir in run_dir.iterdir():
            if not method_dir.is_dir() or method_dir.name.startswith('.'):
                continue
                
            # Enhanced method naming with f_in sampling distinction
            method_name = method_dir.name.upper()
            if method_name == 'SFMPE':
                # Check f_in_sample parameter to distinguish TFMPE variants
                f_in_sample = config.get('f_in_sample', 'prior')
                if f_in_sample == 'observed':
                    method_name = 'TFMPE (observed)'
                else:
                    method_name = 'TFMPE (prior)'
            elif method_name == 'FMPE':
                method_name = 'FMPE'
                
            stats_file = method_dir / 'stats.json'
            
            if not stats_file.exists():
                continue
                
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    
                main_stat = stats.get('main_stat')
                null_stats = stats.get('null_stats', [])
                if main_stat is not None:
                    # Store in raw_data grouped by (task, method, target_value)
                    key = (task, method_name, target_value)
                    if key not in raw_data:
                        raw_data[key] = []
                    raw_data[key].append((main_stat, null_stats))
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse {stats_file}: {e}")
                continue
    
    # Convert raw_data to final metrics format
    for (task, method, target_value), seed_data in raw_data.items():
        task_method_key = (task, method)
        if task_method_key not in metrics:
            metrics[task_method_key] = []
        
        # Separate main stats and null stats
        main_stats = [data[0] for data in seed_data]
        null_stats_per_seed = [data[1] for data in seed_data]
        
        metrics[task_method_key].append((target_value, main_stats, null_stats_per_seed))
    
    # Sort by target variable value
    for task_method_key in metrics:
        metrics[task_method_key].sort(key=lambda x: x[0])
    
    return metrics


def detect_task_type(run_dir: Path, config: Dict[str, Any]) -> Optional[str]:
    """
    Detect whether this is a Gaussian or Brownian task.

    Args:
        run_dir: Run directory path
        config: Configuration dictionary

    Returns:
        'Gaussian' or 'Brownian' or None if cannot detect
    """
    # Check for task indicators in the config or directory structure
    # Look for 'obs_rate' which is specific to Brownian motion
    if 'obs_rate' in config:
        return 'Brownian'
    # Look for 'var_obs' which is specific to Gaussian
    elif 'var_obs' in config:
        return 'Gaussian'
    elif 'n_sites' in config:
        return 'SEIR'
    else:
        # Fallback: check parent directory names
        dir_path = str(run_dir)
        if 'brownian' in dir_path.lower():
            return 'Brownian'
        elif 'gaussian' in dir_path.lower():
            return 'Gaussian'
    return None


SweepType = Literal['samples', 'dimension']


def detect_sweep_type(multirun_dir: Path) -> Optional[SweepType]:
    """
    Detect whether a multirun sweeps over samples or dimension.

    Args:
        multirun_dir: Directory containing a Hydra multirun

    Returns:
        'samples' if sweeping n_simulations, 'dimension' if sweeping
        n_theta/n_sites, None if cannot detect
    """
    multirun_yaml = multirun_dir / "multirun.yaml"
    if not multirun_yaml.exists():
        return None

    try:
        config = OmegaConf.load(multirun_yaml)
        overrides = config.get('hydra', {}).get('overrides', {}).get(
            'task', []
        )

        for override in overrides:
            # Look for comma-separated values indicating a sweep
            if 'n_simulations=' in override and ',' in override:
                return 'samples'
            if ('n_theta=' in override or 'n_sites=' in override) and \
                ',' in override:
                return 'dimension'

    except Exception as e:
        print(f"Warning: Failed to parse {multirun_yaml}: {e}")

    return None


def scan_multirun_directories(
    base_dir: Path
) -> Dict[Tuple[SweepType, str], List[Path]]:
    """
    Scan base directory and categorize multirun directories by sweep type
    and task.

    Args:
        base_dir: Base directory containing timestamped multirun directories

    Returns:
        Dictionary mapping (sweep_type, task) to list of multirun directories
    """
    categorized = {}

    # Find all potential multirun directories
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        sweep_type = detect_sweep_type(item)
        if sweep_type is None:
            continue

        # Look at first run config to determine task type
        first_run = None
        for subdir in item.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                first_run = subdir
                break

        if first_run is None:
            continue

        config = load_config_from_run(first_run)
        if config is None:
            continue

        task = detect_task_type(first_run, config)
        if task is None:
            continue

        key = (sweep_type, task)
        if key not in categorized:
            categorized[key] = []
        categorized[key].append(item)

    return categorized


def collect_metrics_from_multirun(
    multirun_dir: Path,
    target_variable: str,
    filters: Optional[Dict[str, int]] = None
) -> Dict[Tuple[str, str], List[Tuple[int, List[float],
                                      List[List[float]]]]]:
    """
    Collect metrics from a single multirun directory.

    Args:
        multirun_dir: Directory containing a Hydra multirun
        target_variable: Variable to use as x-axis
        filters: Dictionary of config filters to apply

    Returns:
        Dictionary mapping (task, method) tuples to metrics data
    """
    # Use existing collect_metrics but point it at the multirun dir
    return collect_metrics(multirun_dir, target_variable, filters)


def plot_metrics(
    metrics: Dict[Tuple[str, str], List[Tuple[int, List[float], List[List[float]]]]], 
    output_path: Path,
    x_label: str,
    title: str
) -> None:
    """
    Create separate plots for each task comparing LC2ST statistics with confidence intervals.
    
    Args:
        metrics: Dictionary mapping (task, method) to (x_value, [seed_stats], [null_stats_per_seed]) tuples
        output_path: Base path to save the plots (will be modified for each task)
        x_label: Label for the x-axis
        title: Base title for the plots
    """
    # Color scheme for methods
    method_colors = {
        'FMPE': '#1f77b4',           # Blue
        'TFMPE (prior)': '#ff7f0e',  # Orange
        'TFMPE (observed)': '#2ca02c' # Green
    }
    
    # Group metrics by task
    tasks_data = {}
    for (task, method), data in metrics.items():
        if task not in tasks_data:
            tasks_data[task] = {}
        tasks_data[task][method] = data
    
    # Create separate plot for each task
    for task, task_metrics in tasks_data.items():
        plt.figure(figsize=(10, 6))
        
        # Get all x_values for this task to set consistent axis
        all_x_values = set()
        all_means = []
        all_stds = []
        
        for method, data in task_metrics.items():
                
            x_values = []
            y_means = []
            y_stds = []
            
            for x_value, seed_stats, null_stats_per_seed in data:
                x_values.append(x_value)
                y_means.append(np.mean(seed_stats))
                y_stds.append(np.std(seed_stats))
                all_x_values.add(x_value)
            
            x_values = np.array(x_values)
            y_means = np.array(y_means)
            y_stds = np.array(y_stds)
            
            all_means.extend(y_means)
            all_stds.extend(y_stds)
            
            color = method_colors.get(method, '#666666')
            
            # Plot mean line with solid line style
            plt.plot(x_values, y_means, '-', color=color, 
                    label=method, linewidth=2, markersize=6, marker='o')
            
            # Plot confidence interval (mean ± std)
            plt.fill_between(x_values, y_means - y_stds, y_means + y_stds, 
                            color=color, alpha=0.2)
            
            # Plot 95% quantile of null statistics as dotted line
            null_95_quantiles = []
            for x_value, seed_stats, null_stats_per_seed in data:
                # Flatten all null stats for this parameter value across all seeds
                all_null_stats = [stat for seed_null_stats in null_stats_per_seed 
                                for stat in seed_null_stats]
                null_95_quantiles.append(np.percentile(all_null_stats, 95))
            
            plt.plot(x_values, null_95_quantiles, '--', color=color, 
                    linewidth=1.5, alpha=0.7, label=f'{method} (95% null)')
        
        plt.xlabel(x_label)
        plt.ylabel('LC2ST Statistic')
        plt.title(f'{title}: {task} Task')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set discrete x-axis ticks
        sorted_x_values = sorted(all_x_values)
        plt.xticks(sorted_x_values)
        
        # Set reasonable axis limits
        max_val = max(np.array(all_means) + np.array(all_stds))
        plt.ylim(0, max_val * 1.1)
        
        plt.tight_layout()
        
        # Create task-specific output path
        output_dir = output_path.parent
        output_stem = output_path.stem
        output_suffix = output_path.suffix
        task_output_path = output_dir / f"{output_stem}_{task.lower()}{output_suffix}"
        
        plt.savefig(task_output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {task_output_path}")


def plot_combined_metrics(
    categorized_multiruns: Dict[Tuple[SweepType, str], List[Path]],
    output_path: Path,
    filters: Optional[Dict[str, int]] = None
) -> None:
    """
    Create a combined figure with subplots for each task and sweep type.

    Args:
        categorized_multiruns: Dictionary mapping (sweep_type, task) to
                               list of multirun directories
        output_path: Path to save the combined figure
        filters: Dictionary of config filters to apply
    """
    # Color scheme for methods
    method_colors = {
        'FMPE': '#1f77b4',           # Blue
        'TFMPE (prior)': '#ff7f0e',  # Orange
        'TFMPE (observed)': '#2ca02c' # Green
    }

    # Determine which tasks are present
    tasks = sorted(set(task for (_, task) in categorized_multiruns.keys()))
    n_tasks = len(tasks)

    if n_tasks == 0:
        print("No tasks found to plot")
        return

    # Create figure with subplots: rows = tasks, cols = 2 (samples, dim)
    # Share x-axes within columns
    fig, axes = plt.subplots(
        n_tasks, 2, figsize=(16, 5 * n_tasks), squeeze=False,
        sharex='col'
    )

    # Column labels and target variables
    col_config = {
        0: {
            'sweep': 'samples',
            'var': 'n_simulations',
            'label': 'Number of Simulations'
        },
        1: {
            'sweep': 'dimension',
            'var': None,  # Will be determined per task
            'label': 'Number of Parameters'
        }
    }

    # Track all methods seen for legend
    all_methods = set()

    for row_idx, task in enumerate(tasks):
        for col_idx in [0, 1]:
            ax = axes[row_idx, col_idx]
            sweep_type = col_config[col_idx]['sweep']
            key = (sweep_type, task)

            if key not in categorized_multiruns:
                # No data for this combination
                ax.text(
                    0.5, 0.5, 'No data',
                    ha='center', va='center',
                    transform=ax.transAxes
                )
                ax.set_xlabel(col_config[col_idx]['label'])
                ax.set_ylabel('LC2ST Statistic')
                continue

            # Determine target variable for this task/sweep
            if sweep_type == 'samples':
                target_var = 'n_simulations'
            else:
                # Use n_sites for SEIR, n_theta for others
                target_var = 'n_sites' if task == 'SEIR' else 'n_theta'

            # Collect metrics from all multirun directories for this combo
            combined_metrics = {}
            for multirun_dir in categorized_multiruns[key]:
                metrics = collect_metrics_from_multirun(
                    multirun_dir, target_var, filters
                )
                # Merge metrics
                for (mtask, method), data in metrics.items():
                    mkey = (mtask, method)
                    if mkey not in combined_metrics:
                        combined_metrics[mkey] = []
                    combined_metrics[mkey].extend(data)

            # Sort by parameter value and group by method
            for mkey in combined_metrics:
                combined_metrics[mkey].sort(key=lambda x: x[0])

            # Plot each method
            all_x_values = set()
            all_means = []
            all_stds = []

            for (mtask, method), data in combined_metrics.items():
                if mtask != task:
                    continue

                all_methods.add(method)

                x_values = []
                y_means = []
                y_stds = []

                for x_value, seed_stats, null_stats_per_seed in data:
                    x_values.append(x_value)
                    y_means.append(np.mean(seed_stats))
                    y_stds.append(np.std(seed_stats))
                    all_x_values.add(x_value)

                x_values = np.array(x_values)
                y_means = np.array(y_means)
                y_stds = np.array(y_stds)

                all_means.extend(y_means)
                all_stds.extend(y_stds)

                color = method_colors.get(method, '#666666')

                # Plot mean line
                ax.plot(
                    x_values, y_means, '-o',
                    color=color, linewidth=2,
                    markersize=6
                )

                # Plot confidence interval
                ax.fill_between(
                    x_values, y_means - y_stds,
                    y_means + y_stds,
                    color=color, alpha=0.2
                )

                # Plot 95% quantile of null statistics
                null_95_quantiles = []
                for x_value, seed_stats, null_stats_per_seed in data:
                    all_null_stats = [
                        stat for seed_null_stats in null_stats_per_seed
                        for stat in seed_null_stats
                    ]
                    null_95_quantiles.append(
                        np.percentile(all_null_stats, 95)
                    )

                ax.plot(
                    x_values, null_95_quantiles, '--',
                    color=color, linewidth=1.5, alpha=0.7
                )

            # Set x-axis ticks
            if all_x_values:
                sorted_x_values = sorted(all_x_values)
                ax.set_xticks(sorted_x_values)

            # Set y-axis limits
            if all_means and all_stds:
                max_val = max(
                    np.array(all_means) + np.array(all_stds)
                )
                ax.set_ylim(0, max_val * 1.1)

            # Labels - only show x-label on bottom row
            if row_idx == n_tasks - 1:
                ax.set_xlabel(col_config[col_idx]['label'])
            if col_idx == 0:
                ax.set_ylabel('LC2ST Statistic')

            # Title only on top row
            if row_idx == 0:
                col_title = (
                    'Varying Sample Budget' if col_idx == 0
                    else 'Varying Dimension'
                )
                ax.set_title(col_title, fontsize=12, fontweight='bold')

            # Task label on left side
            if col_idx == 0:
                ax.text(
                    -0.15, 0.5, task,
                    transform=ax.transAxes,
                    fontsize=12, fontweight='bold',
                    ha='right', va='center',
                    rotation=90
                )

            ax.grid(True, alpha=0.3)

    # Create shared legend
    handles = []
    labels = []
    for method in ['FMPE', 'TFMPE (prior)', 'TFMPE (observed)']:
        if method in all_methods:
            # Solid line for main stat
            handles.append(
                plt.Line2D(
                    [0], [0], color=method_colors[method],
                    linewidth=2, marker='o', markersize=6
                )
            )
            labels.append(method)
            # Dashed line for null
            handles.append(
                plt.Line2D(
                    [0], [0], color=method_colors[method],
                    linewidth=1.5, linestyle='--', alpha=0.7
                )
            )
            labels.append(f'{method} (95% null)')

    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        frameon=True,
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined plot saved to {output_path}")
