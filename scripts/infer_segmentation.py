#!/usr/bin/env python3
"""
Standalone Segmentation Inference Script

This script loads a segmentation model from a checkpoint, processes images from a folder,
and saves output segmentation masks at original resolution as PNG files.

Usage:
    python scripts/infer_segmentation.py \
        --checkpoint path/to/checkpoint.pt \
        --config path/to/config.yaml \
        --input path/to/images \
        --output path/to/output
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core dependencies
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import yaml

# Computer vision libraries
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Colormap for visualization
from matplotlib import cm


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Build segmentation model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch model
    """
    model_config = config['model']
    data_config = config['data']
    
    # Extract model parameters
    backbone = model_config['backbone']
    num_classes = data_config['num_classes']
    pretrained = model_config.get('pretrained', True)
    in_channels = model_config.get('in_channels', 3)
    
    # Handle SMP DeepLabV3Plus models
    if backbone.startswith('smp_deeplabv3plus_'):
        encoder_name = backbone.replace('smp_deeplabv3plus_', '')
        encoder_weights = "imagenet" if pretrained else None
        
        # Extract additional SMP parameters from config
        encoder_output_stride = model_config.get('encoder_output_stride', 8)
        encoder_depth = model_config.get('encoder_depth', 5)
        decoder_channels = model_config.get('decoder_channels', 256)
        upsampling = model_config.get('upsampling', 4)
        auxiliary_loss = model_config.get('auxiliary_loss', True)
        
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            encoder_output_stride=encoder_output_stride,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            upsampling=upsampling,
            auxiliary_loss=auxiliary_loss
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Only SMP DeepLabV3Plus models are supported.")
    
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> Tuple[int, float, Dict[str, Any]]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (epoch, best_metric, config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    # Use weights_only=False to handle OmegaConf objects and other complex types
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model state with non-strict matching
    model_state = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    # Filter out missing keys and handle shape mismatches
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state_dict.keys():
        if key not in model_state:
            missing_keys.append(key)
    
    for key in model_state.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    
    # Load matching keys
    filtered_state = {k: v for k, v in model_state.items() 
                     if k in model_state_dict and v.shape == model_state_dict[k].shape}
    
    model_state_dict.update(filtered_state)
    model.load_state_dict(model_state_dict)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    config = checkpoint.get('config', {})
    
    print(f"Loaded checkpoint from epoch {epoch}, best metric: {best_metric:.4f}")
    return epoch, best_metric, config


def get_inference_transforms(config: Dict[str, Any]) -> A.Compose:
    """Get inference transforms from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Albumentations transform pipeline
    """
    transforms_config = config.get('transforms', {})
    
    # Base transforms for inference
    transforms = []
    
    # Resize
    if 'resize' in transforms_config:
        size = transforms_config['resize']
        if isinstance(size, int):
            size = (size, size)
        transforms.append(A.Resize(size[0], size[1]))
    
    # Normalization
    if 'normalize' in transforms_config:
        mean = transforms_config['normalize'].get('mean', [0.485, 0.456, 0.406])
        std = transforms_config['normalize'].get('std', [0.229, 0.224, 0.225])
        transforms.append(A.Normalize(mean=mean, std=std))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_color_map(num_classes: int) -> np.ndarray:
    """Get color map for visualization.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        Color map array
    """
    if num_classes == 2:
        # Binary segmentation: background=black, foreground=white
        return np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    else:
        # Multi-class: use matplotlib colormap
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_classes))
        return (colors[:, :3] * 255).astype(np.uint8)


def get_overlay_color_map(num_classes: int) -> np.ndarray:
    """Get color map for overlay visualization with green objects.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        Color map array with RGBA channels
    """
    # Create RGBA color map
    color_map = np.zeros((num_classes, 4), dtype=np.uint8)
    
    # Background class (0) - transparent
    color_map[0] = [0, 0, 0, 0]
    
    # Object classes - bright green with transparency
    if num_classes == 2:
        # Binary: only one object class
        color_map[1] = [0, 255, 0, 180]  # Bright green with ~70% opacity
    else:
        # Multi-class: use different shades of green for different classes
        for i in range(1, num_classes):
            # Vary the green intensity slightly for different classes
            green_intensity = int(255 * (0.7 + 0.3 * (i - 1) / max(1, num_classes - 2)))
            color_map[i] = [0, green_intensity, 0, 180]  # Green with transparency
    
    return color_map


def create_overlay_image(original_image: np.ndarray, mask: np.ndarray, 
                        color_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Create overlay image by blending original image with colored mask.
    
    Args:
        original_image: Original input image (RGB)
        mask: Segmentation mask (class indices)
        color_map: Color mapping for classes (RGBA)
        alpha: Blending factor for mask overlay
        
    Returns:
        Overlay image (RGB)
    """
    # Ensure original image is RGB
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        overlay = original_image.copy()
    else:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create colored mask
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 4), dtype=np.uint8)
    
    for class_id in range(len(color_map)):
        class_mask = (mask == class_id)
        colored_mask[class_mask] = color_map[class_id]
    
    # Separate RGB and alpha channels
    mask_rgb = colored_mask[:, :, :3]
    mask_alpha = colored_mask[:, :, 3:4] / 255.0
    
    # Blend the mask with the original image
    # Only apply overlay where mask_alpha > 0 (non-transparent pixels)
    valid_mask = mask_alpha > 0
    
    for c in range(3):  # RGB channels
        overlay_channel = overlay[:, :, c].astype(np.float32)
        mask_channel = mask_rgb[:, :, c].astype(np.float32)
        alpha_channel = mask_alpha[:, :, 0]
        
        # Blend: overlay = (1 - alpha) * original + alpha * mask
        blended = (1 - alpha_channel * alpha) * overlay_channel + alpha_channel * alpha * mask_channel
        overlay[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return overlay


def process_image(image_path: str, model: torch.nn.Module, 
                 transforms: A.Compose, device: torch.device,
                 original_size: Tuple[int, int], color_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a single image and return segmentation mask and original image.
    
    Args:
        image_path: Path to input image
        model: Segmentation model
        transforms: Image transforms
        device: Device to run inference on
        original_size: Original image size (width, height)
        color_map: Color mapping for visualization
        
    Returns:
        Tuple of (colored_mask, raw_predictions, original_image)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transformed = transforms(image=original_image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        
        # Handle auxiliary loss outputs
        if isinstance(output, tuple):
            output = output[0]
        
        # Get predictions
        predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize back to original size
    predictions_resized = cv2.resize(
        predictions.astype(np.uint8), 
        original_size, 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Apply color mapping
    colored_mask = color_map[predictions_resized]
    
    return colored_mask, predictions_resized, original_image


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Standalone Segmentation Inference')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input images folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output folder for masks')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 1 for memory efficiency)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        sys.exit(1)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model.to(device)
    print(f"Model: {type(model).__name__}")
    
    # Load checkpoint
    epoch, best_metric, checkpoint_config = load_checkpoint(model, args.checkpoint)
    
    # Get inference transforms
    transforms = get_inference_transforms(config)
    
    # Get color maps
    num_classes = config['data']['num_classes']
    color_map = get_color_map(num_classes)
    overlay_color_map = get_overlay_color_map(num_classes)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    input_dir = Path(args.input)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Error: No image files found in {args.input}")
        print(f"Supported extensions: {image_extensions}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    print("Will generate both segmentation masks and overlay images with green object highlighting")
    
    # Process images
    processed_count = 0
    failed_count = 0
    
    for image_file in image_files:
        try:
            print(f"Processing: {image_file.name}")
            
            # Get original image size
            with Image.open(image_file) as img:
                original_size = img.size  # (width, height)
            
            # Process image
            colored_mask, raw_predictions, original_image = process_image(
                str(image_file), model, transforms, device, 
                original_size, color_map
            )
            
            # Save mask
            output_filename = f"{image_file.stem}_mask.png"
            output_path = output_dir / output_filename
            
            # Convert BGR to RGB for PIL
            colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
            mask_image = Image.fromarray(colored_mask_rgb)
            mask_image.save(output_path)
            
            # Create and save overlay image
            overlay_image = create_overlay_image(original_image, raw_predictions, overlay_color_map)
            overlay_filename = f"{image_file.stem}_overlay.png"
            overlay_path = output_dir / overlay_filename
            
            # Save overlay image
            overlay_pil = Image.fromarray(overlay_image)
            overlay_pil.save(overlay_path)
            
            processed_count += 1
            print(f"  -> Saved mask: {output_path}")
            print(f"  -> Saved overlay: {overlay_path}")
            
        except Exception as e:
            print(f"  -> Error processing {image_file.name}: {e}")
            failed_count += 1
            continue
    
    # Summary
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed: {failed_count} images")
    print(f"Generated {processed_count * 2} files total:")
    print(f"  - {processed_count} segmentation masks (*_mask.png)")
    print(f"  - {processed_count} overlay images (*_overlay.png)")
    print(f"Output saved to: {output_dir}")
    
    if failed_count > 0:
        print(f"\nWarning: {failed_count} images failed to process. Check error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()


#python scripts/infer_segmentation.py --checkpoint c:/Facultate/MelanoDet/MarkerSegmentation/vis_epoch_099_0.0164.pt --config c:/Facultate/MelanoDet/MarkerSegmentation/config_vis.yaml --input c:/Facultate/MelanoDet/OriginalDataset/VIS --output c:/Facultate/MelanoDet/OriginalDataset/VIS_Masks
#python scripts/infer_segmentation.py --checkpoint c:/Facultate/MelanoDet/MarkerSegmentation/th_full_epoch_085_0.0501.pt --config c:/Facultate/MelanoDet/MarkerSegmentation/config_th_full.yaml --input c:/Facultate/MelanoDet/OriginalDataset/TH --output c:/Facultate/MelanoDet/OriginalDataset/TH_Masks_Full