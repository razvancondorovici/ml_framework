#!/usr/bin/env python3
"""Evaluation script for PyTorch models."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config, get_config_parser
from utils.device import get_device_info
from utils.logger import StructuredLogger
from utils.checkpoint import load_checkpoint
from datasets.classification import create_classification_dataset
from datasets.segmentation import create_segmentation_dataset
from transforms.augmentations import get_default_classification_transforms, get_segmentation_transforms_from_config
from models.registry import build_classifier, build_segmentation_model
from engine.evaluator import Evaluator


def create_dataset(config: Dict[str, Any], split: str = 'val') -> Any:
    """Create evaluation dataset.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('val', 'test')
        
    Returns:
        Dataset instance
    """
    data_config = config['data']
    dataset_type = data_config.get('dataset_type', 'classification')
    
    # Get transforms
    if dataset_type == 'segmentation':
        num_classes = data_config.get('num_classes', 2)
        transform = get_segmentation_transforms_from_config(config, split=split, num_classes=num_classes)
    else:
        transform = get_default_classification_transforms(split=split)
    
    # Create dataset
    if dataset_type == 'segmentation':
        dataset = create_segmentation_dataset({
            **data_config,
            'transform': transform
        }, split=split)
    else:
        dataset = create_classification_dataset({
            **data_config,
            'transform': transform
        }, split=split)
    
    return dataset


def create_model(config: Dict[str, Any]) -> Any:
    """Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config['model']
    data_config = config['data']
    
    # Get model parameters
    backbone = model_config.get('backbone', 'resnet50')
    num_classes = data_config.get('num_classes', 10)
    pretrained = model_config.get('pretrained', True)
    freeze_backbone = model_config.get('freeze_backbone', False)
    dropout = model_config.get('dropout', 0.0)
    
    # Create model
    if data_config.get('dataset_type') == 'segmentation':
        model = build_segmentation_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        model = build_classifier(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    return model


def setup_evaluation_folder(config: Dict[str, Any], 
                           checkpoint_path: str,
                           split: str,
                           output: Optional[str] = None) -> Path:
    """Setup evaluation folder structure similar to training.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file
        split: Dataset split ('val', 'test')
        output: Optional output path override
        
    Returns:
        Path to evaluation folder
    """
    checkpoint_path = Path(checkpoint_path)
    experiment_name = config.get('experiment', {}).get('name', 'unnamed_experiment')
    
    # If output is explicitly provided, use it
    if output:
        eval_folder = Path(output)
        eval_folder.mkdir(parents=True, exist_ok=True)
        return eval_folder
    
    # Try to infer run folder from checkpoint path
    # Check if checkpoint is in runs/{experiment_name}/... structure
    # Expected structure: runs/{experiment_name}/{timestamp}/checkpoints/...
    run_folder = None
    if 'runs' in checkpoint_path.parts:
        # Find the runs directory
        runs_idx = checkpoint_path.parts.index('runs')
        if len(checkpoint_path.parts) > runs_idx + 1:
            # Check if experiment name matches
            checkpoint_exp_name = checkpoint_path.parts[runs_idx + 1]
            if checkpoint_exp_name == experiment_name:
                # Navigate up from checkpoint to find the run folder (timestamp folder)
                # The run folder should be the parent of 'checkpoints' folder
                current = checkpoint_path.parent
                # If we're in a 'checkpoints' folder, go up one more level
                if current.name == 'checkpoints':
                    run_folder = current.parent
                    # Verify it's under the experiment name
                    if run_folder.parent.name == experiment_name:
                        # Create evaluation subfolder in the same run
                        eval_folder = run_folder / f'eval_{split}'
                        eval_folder.mkdir(parents=True, exist_ok=True)
                        return eval_folder
                else:
                    # Checkpoint might be directly in the timestamp folder
                    # Look for the timestamp folder (parent of current)
                    if current.parent.name == experiment_name:
                        run_folder = current
                        eval_folder = run_folder / f'eval_{split}'
                        eval_folder.mkdir(parents=True, exist_ok=True)
                        return eval_folder
    
    # If we couldn't infer the run folder, create a new evaluation folder
    # Structure: runs/{experiment_name}/eval_{split}_{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_folder = Path('runs') / experiment_name / f'eval_{split}_{timestamp}'
    eval_folder.mkdir(parents=True, exist_ok=True)
    
    return eval_folder


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = get_config_parser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, help='Path to save evaluation results (overrides auto-detection)')
    parser.add_argument('--device', type=str, help='Device to evaluate on (cuda, cpu)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.overrides)
    
    # Setup evaluation folder (similar to training structure)
    eval_folder = setup_evaluation_folder(config, args.checkpoint, args.split, args.output)
    print(f"Evaluation folder: {eval_folder}")
    
    # Save config to evaluation folder
    from utils.config import save_config
    config_path = eval_folder / 'config.yaml'
    save_config(config, config_path)
    
    # Create subdirectories similar to training
    plots_dir = eval_folder / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Create logger
    logger = StructuredLogger(eval_folder, 'evaluation')
    logger.info("Starting evaluation", checkpoint=args.checkpoint, split=args.split, eval_folder=str(eval_folder))
    
    try:
        # Create dataset
        print(f"Creating {args.split} dataset...")
        dataset = create_dataset(config, args.split)
        print(f"Dataset: {len(dataset)} samples")
        
        # Create data loader
        from torch.utils.data import DataLoader
        
        dataloader_config = config.get('dataloader', {})
        dataloader = DataLoader(
            dataset,
            batch_size=dataloader_config.get('batch_size', 32),
            shuffle=False,
            num_workers=dataloader_config.get('num_workers', 4),
            pin_memory=dataloader_config.get('pin_memory', True),
            drop_last=dataloader_config.get('drop_last', False)
        )
        
        # Create model
        print("Creating model...")
        model = create_model(config)
        print(f"Model: {type(model).__name__}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        epoch, best_metric, checkpoint_config = load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            strict=False
        )
        print(f"Loaded checkpoint from epoch {epoch}, best metric: {best_metric:.4f}")
        
        # Create evaluator
        print("Creating evaluator...")
        evaluator = Evaluator(
            model=model,
            config=config,
            device=args.device
        )
        
        # Evaluate model
        print("Starting evaluation...")
        class_names = config.get('data', {}).get('class_names')
        
        # Save plots to plots directory
        if config.get('data', {}).get('dataset_type') == 'segmentation':
            results = evaluator.evaluate_segmentation(
                dataloader=dataloader,
                class_names=class_names,
                save_plots=True,
                save_dir=plots_dir
            )
        else:
            results = evaluator.evaluate_classification(
                dataloader=dataloader,
                class_names=class_names,
                save_plots=True,
                save_dir=plots_dir
            )
        
        # Print results
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric, value in results['metrics'].items():
            if isinstance(value, (list, np.ndarray)):
                print(f"{metric}: {value}")
            elif isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Save results
        import json
        
        # Save metrics
        metrics_path = eval_folder / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save predictions
        predictions_path = eval_folder / 'predictions.npy'
        np.save(predictions_path, results['predictions'])
        
        # Save targets
        targets_path = eval_folder / 'targets.npy'
        np.save(targets_path, results['targets'])
        
        # Save probabilities if available
        if results['probabilities'] is not None:
            probabilities_path = eval_folder / 'probabilities.npy'
            np.save(probabilities_path, results['probabilities'])
        
        # Save evaluation summary
        summary = {
            'checkpoint': str(args.checkpoint),
            'split': args.split,
            'epoch': epoch,
            'best_metric_from_checkpoint': best_metric,
            'num_samples': len(dataset),
            'metrics': results['metrics']
        }
        
        # Add per-class metrics if available
        if 'per_class_accuracy' in results:
            summary['per_class_accuracy'] = results['per_class_accuracy']
        if 'top_k_accuracy' in results:
            summary['top_k_accuracy'] = results['top_k_accuracy']
        if 'per_class_iou' in results:
            summary['per_class_iou'] = results['per_class_iou']
            summary['mean_iou'] = results.get('mean_iou')
        if 'per_class_dice' in results:
            summary['per_class_dice'] = results['per_class_dice']
            summary['mean_dice'] = results.get('mean_dice')
        
        summary_path = eval_folder / 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {eval_folder}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Plots: {plots_dir}")
        print(f"  - Predictions: {predictions_path}")
        print(f"  - Summary: {summary_path}")
        
        logger.info("Evaluation completed successfully", **results['metrics'])
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
