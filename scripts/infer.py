#!/usr/bin/env python3
"""Inference script for PyTorch models."""

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
from models.registry import build_classifier, build_segmentation_model
from engine.inferencer import Inferencer


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
    parser.add_argument('--input', type=str, required=True, help='Path to input images (folder or CSV)')
    parser.add_argument('--output', type=str, required=True, help='Path to save results')
    parser.add_argument('--input-type', type=str, choices=['folder', 'csv'], default='folder', help='Type of input')
    parser.add_argument('--image-column', type=str, default='image_path', help='Name of image column in CSV')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--device', type=str, help='Device to run inference on (cuda, cpu)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.overrides)
    
    # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Create logger
    logger = StructuredLogger(Path(args.output).parent, 'inference')
    logger.info("Starting inference", checkpoint=args.checkpoint, input=args.input)
    
    try:
        # Create model
        print("Creating model...")
        model = create_model(config)
        print(f"Model: {type(model).__name__}")
        
        # Create inferencer
        print("Creating inferencer...")
        inferencer = Inferencer(
            model=model,
            config=config,
            device=args.device
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        inferencer.load_checkpoint(args.checkpoint)
        
        # Run inference
        print("Starting inference...")
        class_names = config.get('data', {}).get('class_names')
        
        if args.input_type == 'folder':
            results = inferencer.predict_folder(
                folder_path=args.input,
                output_path=args.output,
                class_names=class_names,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        else:  # CSV
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
        print(f"Results saved to {args.output}")
        
        # Print sample predictions
        if len(results['results']) > 0:
            print("\nSample predictions:")
            print("=" * 50)
            sample_results = results['results'].head(5)
            for _, row in sample_results.iterrows():
                if 'class_name' in row:
                    print(f"{row['image_path']}: {row['class_name']} (confidence: {row['confidence']:.3f})")
                else:
                    print(f"{row['image_path']}: class {row['prediction']} (confidence: {row['confidence']:.3f})")
        
        logger.info("Inference completed successfully", num_images=len(results['results']))
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == '__main__':
    main()
