#!/usr/bin/env python3
"""Run summarization script for PyTorch training runs."""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import StructuredLogger


def load_run_data(run_folder: Path) -> Dict[str, Any]:
    """Load data from a training run.
    
    Args:
        run_folder: Path to run folder
        
    Returns:
        Dictionary containing run data
    """
    run_data = {}
    
    # Load config
    config_path = run_folder / 'config.yaml'
    if config_path.exists():
        from utils.config import load_config
        run_data['config'] = load_config(config_path)
    
    # Load training history
    history_path = run_folder / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            run_data['history'] = json.load(f)
    
    # Load metrics
    metrics_path = run_folder / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            run_data['metrics'] = json.load(f)
    
    # Load scalars
    scalars_path = run_folder / 'scalars.csv'
    if scalars_path.exists():
        run_data['scalars'] = pd.read_csv(scalars_path)
    
    return run_data


def print_run_summary(run_folder: Path, run_data: Dict[str, Any]):
    """Print run summary.
    
    Args:
        run_folder: Path to run folder
        run_data: Run data dictionary
    """
    print(f"Run Folder: {run_folder}")
    print("=" * 80)
    
    # Print config summary
    if 'config' in run_data:
        config = run_data['config']
        print("\nConfiguration:")
        print("-" * 40)
        
        # Experiment info
        experiment = config.get('experiment', {})
        print(f"Experiment: {experiment.get('name', 'Unknown')}")
        print(f"Seed: {experiment.get('seed', 'Unknown')}")
        
        # Model info
        model = config.get('model', {})
        print(f"Model: {model.get('backbone', 'Unknown')}")
        print(f"Pretrained: {model.get('pretrained', 'Unknown')}")
        print(f"Freeze Backbone: {model.get('freeze_backbone', 'Unknown')}")
        
        # Data info
        data = config.get('data', {})
        print(f"Dataset Type: {data.get('dataset_type', 'Unknown')}")
        print(f"Num Classes: {data.get('num_classes', 'Unknown')}")
        
        # Training info
        training = config.get('training', {})
        print(f"Epochs: {training.get('epochs', 'Unknown')}")
        print(f"Batch Size: {config.get('dataloader', {}).get('batch_size', 'Unknown')}")
        print(f"Learning Rate: {config.get('optimizer', {}).get('lr', 'Unknown')}")
        print(f"Optimizer: {config.get('optimizer', {}).get('name', 'Unknown')}")
        print(f"Scheduler: {config.get('scheduler', {}).get('name', 'Unknown')}")
    
    # Print training history
    if 'history' in run_data:
        history = run_data['history']
        print("\nTraining History:")
        print("-" * 40)
        
        if 'train_loss' in history and history['train_loss']:
            final_train_loss = history['train_loss'][-1]
            print(f"Final Train Loss: {final_train_loss:.4f}")
        
        if 'val_loss' in history and history['val_loss']:
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])
            print(f"Final Val Loss: {final_val_loss:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
        
        if 'learning_rates' in history and history['learning_rates']:
            final_lr = history['learning_rates'][-1]
            print(f"Final Learning Rate: {final_lr:.2e}")
    
    # Print metrics
    if 'metrics' in run_data:
        metrics = run_data['metrics']
        print("\nFinal Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
    
    # Print file paths
    print("\nFiles:")
    print("-" * 40)
    
    # Checkpoints
    checkpoint_dir = run_folder / 'checkpoints'
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        if checkpoints:
            print(f"Checkpoints: {len(checkpoints)} files")
            for checkpoint in sorted(checkpoints):
                print(f"  - {checkpoint.name}")
    
    # Plots
    plot_dir = run_folder / 'plots'
    if plot_dir.exists():
        plots = list(plot_dir.glob('*.png'))
        if plots:
            print(f"Plots: {len(plots)} files")
            for plot in sorted(plots):
                print(f"  - {plot.name}")
    
    # Samples
    sample_dir = run_folder / 'samples'
    if sample_dir.exists():
        samples = list(sample_dir.glob('*.png'))
        if samples:
            print(f"Sample Visualizations: {len(samples)} files")
    
    # Logs
    log_files = list(run_folder.glob('*.jsonl'))
    if log_files:
        print(f"Log Files: {len(log_files)} files")
        for log_file in sorted(log_files):
            print(f"  - {log_file.name}")


def compare_runs(run_folders: List[Path]):
    """Compare multiple runs.
    
    Args:
        run_folders: List of run folder paths
    """
    print("Run Comparison:")
    print("=" * 80)
    
    # Load data for all runs
    runs_data = []
    for run_folder in run_folders:
        run_data = load_run_data(run_folder)
        run_data['folder'] = run_folder
        runs_data.append(run_data)
    
    # Create comparison table
    comparison_data = []
    for run_data in runs_data:
        row = {'Run': run_data['folder'].name}
        
        # Add config info
        if 'config' in run_data:
            config = run_data['config']
            row['Model'] = config.get('model', {}).get('backbone', 'Unknown')
            row['Epochs'] = config.get('training', {}).get('epochs', 'Unknown')
            row['Batch Size'] = config.get('dataloader', {}).get('batch_size', 'Unknown')
            row['Learning Rate'] = config.get('optimizer', {}).get('lr', 'Unknown')
        
        # Add final metrics
        if 'history' in run_data:
            history = run_data['history']
            if 'val_loss' in history and history['val_loss']:
                row['Final Val Loss'] = f"{history['val_loss'][-1]:.4f}"
                row['Best Val Loss'] = f"{min(history['val_loss']):.4f}"
        
        if 'metrics' in run_data:
            metrics = run_data['metrics']
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[metric] = f"{value:.4f}"
        
        comparison_data.append(row)
    
    # Print comparison table
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))


def main():
    """Main summarization function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Summarize training runs')
    parser.add_argument('--run-folder', type=str, help='Path to single run folder')
    parser.add_argument('--runs-dir', type=str, help='Path to directory containing multiple runs')
    parser.add_argument('--compare', action='store_true', help='Compare multiple runs')
    parser.add_argument('--latest', action='store_true', help='Show latest run')
    args = parser.parse_args()
    
    if args.run_folder:
        # Single run
        run_folder = Path(args.run_folder)
        if not run_folder.exists():
            print(f"Run folder not found: {run_folder}")
            return
        
        run_data = load_run_data(run_folder)
        print_run_summary(run_folder, run_data)
    
    elif args.runs_dir:
        # Multiple runs
        runs_dir = Path(args.runs_dir)
        if not runs_dir.exists():
            print(f"Runs directory not found: {runs_dir}")
            return
        
        # Find run folders
        run_folders = []
        for item in runs_dir.iterdir():
            if item.is_dir():
                # Check if it's a run folder (contains config.yaml)
                if (item / 'config.yaml').exists():
                    run_folders.append(item)
        
        if not run_folders:
            print(f"No run folders found in {runs_dir}")
            return
        
        if args.latest:
            # Show latest run
            latest_run = max(run_folders, key=lambda x: x.stat().st_mtime)
            run_data = load_run_data(latest_run)
            print_run_summary(latest_run, run_data)
        elif args.compare:
            # Compare runs
            compare_runs(run_folders)
        else:
            # List all runs
            print(f"Found {len(run_folders)} runs in {runs_dir}:")
            print("-" * 40)
            for run_folder in sorted(run_folders):
                print(f"  - {run_folder.name}")
    
    else:
        print("Please provide either --run-folder or --runs-dir")
        return


if __name__ == '__main__':
    main()
