"""
Shared utilities for plotting metrics from Hydra runs.

This module contains common functions used by plotting scripts to load 
configurations, collect metrics, and create plots from Hydra output directories.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
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
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Collect LC2ST metrics from all Hydra runs.
    
    Args:
        base_dir: Base directory containing Hydra outputs
        target_variable: Variable to use as x-axis (e.g., 'n_simulations', 'n_theta')
        filters: Dictionary of config filters to apply (e.g., {'n_rounds': 10})
        
    Returns:
        Dictionary mapping method names to list of (target_value, main_stat)
    """
    if filters is None:
        filters = {}
        
    metrics = {}
    
    # Scan for Hydra output directories (typically contain .hydra subdirectory)
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        # Check if this is a Hydra run directory
        if not (run_dir / ".hydra").exists():
            continue
            
        # Load configuration
        config = load_config_from_run(run_dir)
        if config is None:
            continue
            
        # Get target variable value
        target_value = config.get(target_variable)
        if target_value is None:
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
                    metrics[method_name].append((target_value, main_stat))
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse {stats_file}: {e}")
                continue
    
    # Sort by target variable value
    for method_name in metrics:
        metrics[method_name].sort(key=lambda x: x[0])
    
    return metrics


def plot_metrics(
    metrics: Dict[str, List[Tuple[int, float]]], 
    output_path: Path,
    x_label: str,
    title: str
) -> None:
    """
    Create a plot comparing LC2ST statistics.
    
    Args:
        metrics: Dictionary mapping method names to (x_value, main_stat) pairs
        output_path: Path to save the plot
        x_label: Label for the x-axis
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method_name, data) in enumerate(metrics.items()):
        if not data:
            continue
            
        x_values, stats = zip(*data)
        color = colors[i % len(colors)]
        
        plt.plot(x_values, stats, 'o-', label=method_name, color=color, 
                linewidth=2, markersize=6)
    
    plt.xlabel(x_label)
    plt.ylabel('LC2ST Statistic')
    plt.title(title)
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