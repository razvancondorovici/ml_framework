"""Data augmentation pipelines using Albumentations and torchvision."""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from PIL import Image


class AlbumentationsTransform:
    """Wrapper for Albumentations transforms to work with PIL Images."""
    
    def __init__(self, transform: A.Compose):
        self.transform = transform
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply transform to PIL Image.
        
        Args:
            image: PIL Image
            
        Returns:
            Transformed image as tensor
        """
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Apply Albumentations transform
        transformed = self.transform(image=image_np)
        
        # Convert back to tensor
        return torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0


class AlbumentationsSegmentationTransform:
    """Wrapper for Albumentations transforms for segmentation tasks."""
    
    def __init__(self, transform: A.Compose):
        self.transform = transform
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transform to PIL Image and mask.
        
        Args:
            image: PIL Image
            mask: PIL Mask
            
        Returns:
            Tuple of (transformed_image, transformed_mask) as tensors
        """
        # Convert PIL to numpy
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Apply Albumentations transform
        transformed = self.transform(image=image_np, mask=mask_np)
        
        # Convert back to tensors
        if isinstance(transformed['image'], torch.Tensor):
            # Already converted to tensor by ToTensorV2
            image_tensor = transformed['image']
        else:
            # Convert from numpy
            image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
        
        if isinstance(transformed['mask'], torch.Tensor):
            # Already converted to tensor
            mask_tensor = transformed['mask'].long()
        else:
            # Convert from numpy
            mask_tensor = torch.from_numpy(transformed['mask']).long()
        
        return image_tensor, mask_tensor


def get_classification_transforms(config: Dict[str, Any], 
                                split: str = 'train',
                                deterministic: bool = False) -> T.Compose:
    """Get classification transforms for a specific split.
    
    Args:
        config: Transform configuration
        split: Data split ('train', 'val', 'test')
        deterministic: Whether to use deterministic transforms
        
    Returns:
        Transform pipeline
    """
    if config.get('use_albumentations', True):
        return get_albumentations_classification_transforms(config, split, deterministic)
    else:
        return get_torchvision_classification_transforms(config, split)


def get_albumentations_classification_transforms(config: Dict[str, Any],
                                               split: str = 'train',
                                               deterministic: bool = False) -> AlbumentationsTransform:
    """Get Albumentations transforms for classification.
    
    Args:
        config: Transform configuration
        split: Data split ('train', 'val', 'test')
        deterministic: Whether to use deterministic transforms
        
    Returns:
        Albumentations transform pipeline
    """
    # Base transforms
    transforms = []
    
    # Resize
    if 'resize' in config:
        size = config['resize']
        if isinstance(size, int):
            size = (size, size)
        transforms.append(A.Resize(size[0], size[1]))
    
    # Training augmentations
    if split == 'train' and not deterministic:
        # Geometric transforms
        if config.get('horizontal_flip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if config.get('rotation', 0) > 0:
            transforms.append(A.Rotate(limit=config['rotation'], p=0.5))
        
        if config.get('shift_scale_rotate', False):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ))
        
        # Color transforms
        if config.get('color_jitter', False):
            transforms.append(A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ))
        
        if config.get('hue_saturation_value', False):
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ))
        
        # Noise and blur
        if config.get('gaussian_noise', False):
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
        
        if config.get('gaussian_blur', False):
            transforms.append(A.GaussianBlur(blur_limit=3, p=0.3))
        
        # Advanced augmentations
        if config.get('cutout', False):
            transforms.append(A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.3
            ))
        
        if config.get('mixup', False):
            transforms.append(A.MixUp(p=0.2))
    
    # Normalization
    if 'normalize' in config:
        mean = config['normalize'].get('mean', [0.485, 0.456, 0.406])
        std = config['normalize'].get('std', [0.229, 0.224, 0.225])
        transforms.append(A.Normalize(mean=mean, std=std))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    # Create transform pipeline
    transform = A.Compose(transforms)
    
    return AlbumentationsTransform(transform)


def get_torchvision_classification_transforms(config: Dict[str, Any],
                                            split: str = 'train') -> T.Compose:
    """Get torchvision transforms for classification.
    
    Args:
        config: Transform configuration
        split: Data split ('train', 'val', 'test')
        
    Returns:
        Torchvision transform pipeline
    """
    transforms = []
    
    # Resize
    if 'resize' in config:
        size = config['resize']
        if isinstance(size, int):
            size = (size, size)
        transforms.append(T.Resize(size))
    
    # Training augmentations
    if split == 'train':
        if config.get('horizontal_flip', True):
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        
        if config.get('vertical_flip', False):
            transforms.append(T.RandomVerticalFlip(p=0.5))
        
        if config.get('rotation', 0) > 0:
            transforms.append(T.RandomRotation(degrees=config['rotation']))
        
        if config.get('color_jitter', False):
            transforms.append(T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ))
    
    # Convert to tensor
    transforms.append(T.ToTensor())
    
    # Normalization
    if 'normalize' in config:
        mean = config['normalize'].get('mean', [0.485, 0.456, 0.406])
        std = config['normalize'].get('std', [0.229, 0.224, 0.225])
        transforms.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transforms)


def get_segmentation_transforms(config: Dict[str, Any],
                              split: str = 'train',
                              deterministic: bool = False) -> AlbumentationsSegmentationTransform:
    """Get segmentation transforms for a specific split.
    
    Args:
        config: Transform configuration
        split: Data split ('train', 'val', 'test')
        deterministic: Whether to use deterministic transforms
        
    Returns:
        Segmentation transform pipeline
    """
    # Base transforms
    transforms = []
    
    # Resize
    if 'resize' in config:
        size = config['resize']
        if isinstance(size, int):
            size = (size, size)
        transforms.append(A.Resize(size[0], size[1]))
    
    # Training augmentations
    if split == 'train' and not deterministic:
        # Geometric transforms (same for image and mask)
        if config.get('horizontal_flip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if config.get('rotation', 0) > 0:
            transforms.append(A.Rotate(limit=config['rotation'], p=0.5))
        
        if config.get('shift_scale_rotate', False):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ))
        
        # Elastic transform
        if config.get('elastic_transform', False):
            transforms.append(A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ))
        
        # Grid distortion
        if config.get('grid_distortion', False):
            transforms.append(A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3
            ))
        
        # Color transforms (only for image)
        if config.get('color_jitter', False):
            transforms.append(A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ))
        
        if config.get('hue_saturation_value', False):
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ))
        
        # Noise and blur (only for image)
        if config.get('gaussian_noise', False):
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
        
        if config.get('gaussian_blur', False):
            transforms.append(A.GaussianBlur(blur_limit=3, p=0.3))
    
    # Normalization (only for image)
    if 'normalize' in config:
        mean = config['normalize'].get('mean', [0.485, 0.456, 0.406])
        std = config['normalize'].get('std', [0.229, 0.224, 0.225])
        transforms.append(A.Normalize(mean=mean, std=std))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    # Create transform pipeline
    transform = A.Compose(transforms)
    
    return AlbumentationsSegmentationTransform(transform)


def get_default_classification_transforms(image_size: int = 224,
                                        split: str = 'train',
                                        use_albumentations: bool = True) -> Union[T.Compose, AlbumentationsTransform]:
    """Get default classification transforms.
    
    Args:
        image_size: Target image size
        split: Data split ('train', 'val', 'test')
        use_albumentations: Whether to use Albumentations
        
    Returns:
        Transform pipeline
    """
    config = {
        'resize': image_size,
        'horizontal_flip': True,
        'color_jitter': split == 'train',
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'use_albumentations': use_albumentations
    }
    
    return get_classification_transforms(config, split)


def get_default_segmentation_transforms(image_size: int = 512,
                                      split: str = 'train') -> AlbumentationsSegmentationTransform:
    """Get default segmentation transforms.
    
    Args:
        image_size: Target image size
        split: Data split ('train', 'val', 'test')
        
    Returns:
        Segmentation transform pipeline
    """
    config = {
        'resize': image_size,
        'horizontal_flip': True,
        'rotation': 15,
        'shift_scale_rotate': True,
        'elastic_transform': split == 'train',
        'grid_distortion': split == 'train',
        'color_jitter': split == 'train',
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    return get_segmentation_transforms(config, split)
