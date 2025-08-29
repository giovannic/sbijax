"""
Shared utilities for plotting metrics from Hydra runs.

This module contains common functions used by plotting scripts to load 
configurations, collect metrics, and create plots from Hydra output directories.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


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
) -> Dict[Tuple[str, str], List[Tuple[int, List[float]]]]:
    """
    Collect LC2ST metrics from all Hydra runs, grouped by task and method.
    
    Args:
        base_dir: Base directory containing Hydra outputs
        target_variable: Variable to use as x-axis (e.g., 'n_simulations', 'n_theta')
        filters: Dictionary of config filters to apply (e.g., {'n_rounds': 10})
        
    Returns:
        Dictionary mapping (task, method) tuples to list of (target_value, [seed_stats])
        where task is 'Gaussian' or 'Brownian' and method is 'FMPE' or 'TFMPE'
    """
    if filters is None:
        filters = {}
        
    # Store data grouped by (task, method, target_value, seed)
    raw_data = {}
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
                
            # Rename SFMPE to TFMPE (Tokenised Flow Matching for Posterior Estimation)
            method_name = method_dir.name.upper()
            if method_name == 'SFMPE':
                method_name = 'TFMPE'
                
            stats_file = method_dir / 'stats.json'
            
            if not stats_file.exists():
                continue
                
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    
                main_stat = stats.get('main_stat')
                if main_stat is not None:
                    # Store in raw_data grouped by (task, method, target_value)
                    key = (task, method_name, target_value)
                    if key not in raw_data:
                        raw_data[key] = []
                    raw_data[key].append(main_stat)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse {stats_file}: {e}")
                continue
    
    # Convert raw_data to final metrics format
    for (task, method, target_value), seed_stats in raw_data.items():
        task_method_key = (task, method)
        if task_method_key not in metrics:
            metrics[task_method_key] = []
        metrics[task_method_key].append((target_value, seed_stats))
    
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
    else:
        # Fallback: check parent directory names
        dir_path = str(run_dir)
        if 'brownian' in dir_path.lower():
            return 'Brownian'
        elif 'gaussian' in dir_path.lower():
            return 'Gaussian'
    return None


def plot_metrics(
    metrics: Dict[Tuple[str, str], List[Tuple[int, List[float]]]], 
    output_path: Path,
    x_label: str,
    title: str
) -> None:
    """
    Create a plot comparing LC2ST statistics with confidence intervals.
    
    Args:
        metrics: Dictionary mapping (task, method) to (x_value, [seed_stats]) pairs
        output_path: Path to save the plot
        x_label: Label for the x-axis
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Color scheme: Gaussian (blues), Brownian (oranges/reds)
    task_colors = {
        'Gaussian': '#1f77b4',  # Blue
        'Brownian': '#ff7f0e'   # Orange
    }
    
    # Line styles: FMPE (solid), TFMPE (dashed)
    method_styles = {
        'FMPE': '-',
        'TFMPE': '--'
    }
    
    for (task, method), data in metrics.items():
        if not data:
            continue
            
        x_values = []
        y_means = []
        y_stds = []
        
        for x_value, seed_stats in data:
            x_values.append(x_value)
            y_means.append(np.mean(seed_stats))
            y_stds.append(np.std(seed_stats))
        
        x_values = np.array(x_values)
        y_means = np.array(y_means)
        y_stds = np.array(y_stds)
        
        color = task_colors.get(task, '#666666')
        linestyle = method_styles.get(method, '-')
        label = f"{task} {method}"
        
        # Plot mean line
        plt.plot(x_values, y_means, linestyle=linestyle, color=color, 
                label=label, linewidth=2, markersize=6, marker='o')
        
        # Plot confidence interval (mean ± std)
        plt.fill_between(x_values, y_means - y_stds, y_means + y_stds, 
                        color=color, alpha=0.2)
    
    plt.xlabel(x_label)
    plt.ylabel('LC2ST Statistic')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set discrete x-axis ticks
    if metrics:
        all_x_values = set()
        for data in metrics.values():
            for x_value, _ in data:
                all_x_values.add(x_value)
        if all_x_values:
            sorted_x_values = sorted(all_x_values)
            plt.xticks(sorted_x_values)
    
    # Set reasonable axis limits
    if metrics:
        all_means = []
        all_stds = []
        for data in metrics.values():
            for _, seed_stats in data:
                all_means.append(np.mean(seed_stats))
                all_stds.append(np.std(seed_stats))
        if all_means:
            max_val = max(np.array(all_means) + np.array(all_stds))
            plt.ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")