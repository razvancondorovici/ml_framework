#!/usr/bin/env python3
"""Model export script for PyTorch models."""

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
    """Main export function."""
    # Parse arguments
    parser = get_config_parser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Path to save exported model')
    parser.add_argument('--format', type=str, choices=['torchscript', 'onnx'], default='torchscript', help='Export format')
    parser.add_argument('--input-shape', type=int, nargs=3, default=[3, 224, 224], help='Input shape (C, H, W)')
    parser.add_argument('--device', type=str, help='Device to run export on (cuda, cpu)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.overrides)
    
    # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Create logger
    logger = StructuredLogger(Path(args.output).parent, 'export')
    logger.info("Starting model export", checkpoint=args.checkpoint, format=args.format)
    
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
        
        # Export model
        print(f"Exporting model to {args.format}...")
        input_shape = tuple(args.input_shape)
        inferencer.export_model(
            export_path=args.output,
            export_format=args.format,
            input_shape=input_shape
        )
        
        print(f"Model exported successfully to {args.output}")
        logger.info("Model export completed successfully")
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise


if __name__ == '__main__':
    main()
