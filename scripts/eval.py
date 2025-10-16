#!/usr/bin/env python3
"""Evaluation script for PyTorch models."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config, get_config_parser
from utils.device import get_device_info
from utils.logger import StructuredLogger
from utils.checkpoint import load_checkpoint
from datasets.classification import create_classification_dataset
from datasets.segmentation import create_segmentation_dataset
from transforms.augmentations import get_default_classification_transforms, get_default_segmentation_transforms
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
        transform = get_default_segmentation_transforms(split=split)
    else:
        transform = get_default_classification_transforms(split=split)
    
    # Create dataset
    if dataset_type == 'segmentation':
        dataset = create_segmentation_dataset({
            **data_config,
            'transform': transform
        })
    else:
        dataset = create_classification_dataset({
            **data_config,
            'transform': transform
        })
    
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


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = get_config_parser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, help='Path to save evaluation results')
    parser.add_argument('--device', type=str, help='Device to evaluate on (cuda, cpu)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.overrides)
    
    # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Create logger
    log_dir = args.output or 'logs'
    logger = StructuredLogger(log_dir, 'evaluation')
    logger.info("Starting evaluation", checkpoint=args.checkpoint, split=args.split)
    
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
        load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            strict=False
        )
        
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
        save_dir = Path(args.output) if args.output else None
        
        if config.get('data', {}).get('dataset_type') == 'segmentation':
            results = evaluator.evaluate_segmentation(
                dataloader=dataloader,
                class_names=class_names,
                save_plots=True,
                save_dir=save_dir
            )
        else:
            results = evaluator.evaluate_classification(
                dataloader=dataloader,
                class_names=class_names,
                save_plots=True,
                save_dir=save_dir
            )
        
        # Print results
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            import json
            metrics_path = output_path / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(results['metrics'], f, indent=2)
            
            # Save predictions
            import numpy as np
            predictions_path = output_path / 'predictions.npy'
            np.save(predictions_path, results['predictions'])
            
            print(f"\nResults saved to {output_path}")
        
        logger.info("Evaluation completed successfully", **results['metrics'])
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
