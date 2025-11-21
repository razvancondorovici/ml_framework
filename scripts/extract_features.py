#!/usr/bin/env python3
"""Feature extraction script for PyTorch models."""

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
    """Main feature extraction function."""
    # Parse arguments
    parser = get_config_parser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input images folder')
    parser.add_argument('--output', type=str, required=True, help='Path to save features')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, help='Device to run on (cuda, cpu)')
    parser.add_argument('--save-format', type=str, choices=['pickle', 'npz', 'both'], default='pickle',
                       help='Format to save features')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.overrides)
    
    # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Create logger
    logger = StructuredLogger(Path(args.output).parent, 'feature_extraction')
    logger.info("Starting feature extraction", checkpoint=args.checkpoint, input=args.input)
    
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
        
        # Extract features
        print("Starting feature extraction...")
        features_dict = inferencer.extract_features_from_folder(
            folder_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_format=args.save_format
        )
        
        # Print summary
        print(f"\nFeature extraction completed successfully!")
        print(f"Extracted features from {len(features_dict)} images")
        print(f"Features saved to {args.output}")
        
        # Print sample
        sample_keys = list(features_dict.keys())[:3]
        print("\nSample features:")
        print("=" * 50)
        for key in sample_keys:
            feature = features_dict[key]
            print(f"{key}: shape={feature.shape}, mean={feature.mean():.4f}, std={feature.std():.4f}")
        
        logger.info("Feature extraction completed successfully", num_images=len(features_dict))
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise


if __name__ == '__main__':
    main()





