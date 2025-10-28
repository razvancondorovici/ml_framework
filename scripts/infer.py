#!/usr/bin/env python3
"""Inference script for PyTorch models."""

import argparse
import os
import sys
from pathlib import Path
import torch
from typing import Dict, Any

from torch.utils.checkpoint import checkpoint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config, get_config_parser
from utils.device import get_device_info
from utils.logger import StructuredLogger
from models.registry import build_classifier, build_segmentation_model
from engine.inferencer import Inferencer
from metrics.wrappers import MetricsWrapper


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
    """Main inference function."""
    # Parse arguments
    parser = get_config_parser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    # Load configuration
    config = load_config(args.config, args.overrides)
    checkpoint_path = args.checkpoint
    # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")

    config['output'] = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "test")
    # Create logger
    logger = StructuredLogger(Path(config['output']), 'inference')
    logger.info("Starting inference", checkpoint=checkpoint_path, input=config['input'])
    
    try:
        # Create model
        print("Creating model...")
        model = create_model(config)
        print(f"Model: {type(model).__name__}")
        
        # Create inferencer
        print("Creating inferencer...")
        inferencer = Inferencer(
            model=model, config=config,
            device="cuda" if device_info["cuda_available"] else "cpu"
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        inferencer.load_checkpoint(checkpoint_path)
        
        # Run inference
        print("Starting inference...")
        class_names = config.get('data', {}).get('class_names')
        input_type = "folder" if os.path.isdir(config['input']) else "csv"

        if input_type == 'folder':
            results = inferencer.predict_folder(
                folder_path=config['input'],
                output_path=config['output'],
                class_names=class_names,
                batch_size=config['dataloader']['batch_size'],
                num_workers=config['dataloader']['num_workers']
            )
        else:  # CSV - this functionality remained unchanged with "args"
            results = inferencer.predict_csv(
                csv_path=args.input,
                image_column=args.image_column,
                output_path=args.output,
                class_names=class_names,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        
        # Print results summary
        print(f"\nInference completed!")
        print(f"Processed {len(results['results'])} images")
        print(f"Results saved to {config['output']}")

        metrics_instance = MetricsWrapper(config['data'].get('num_classes', 10), config['metrics'].task,
            average=config.get('average', 'macro'),
            threshold=config.get('threshold', 0.5))
        metrics_instance.update(torch.tensor(results['probabilities']), torch.tensor(results['results']['GT']))
        test_metrics = metrics_instance.compute()
        test_metrics = {k: v.float().mean().item() if hasattr(v, 'item') and v.numel() > 1 else (v.item() if hasattr(v, 'item') else v) for k, v in test_metrics.items()}

        # Print sample predictions; actually the first 5 of them
        if len(results['results']) > 0:
            print("\nSample predictions:")
            print("=" * 50)
            sample_results = results['results'].head(5)
            for _, row in sample_results.iterrows():
                if 'class_name' in row:
                    logger.info(f"{row['image_path']}: {row['class_name']} (confidence: {row['confidence']:.3f})")
                else:
                    logger.info(f"{row['image_path']}: class {row['prediction']} (confidence: {row['confidence']:.3f})")
        
        logger.info("Inference completed successfully", num_images=len(results['results']))
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

    print(test_metrics)
    logger.info(test_metrics)


if __name__ == '__main__':
    print("RUNNING INFERENCE", "+="*34)
    main()
