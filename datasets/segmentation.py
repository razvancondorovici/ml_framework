"""Segmentation dataset implementations."""

import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Callable, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

# Import Kaggle utilities if available
try:
    from utils.kaggle_utils import is_kaggle_environment, get_kaggle_paths
except ImportError:
    is_kaggle_environment = lambda: False
    get_kaggle_paths = lambda: {}


class SegmentationDataset(Dataset):
    """Image segmentation dataset.
    
    Supports both indexed masks (0, 1, 2, ...) and RGB masks with color mapping.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 annotations_file: Optional[Union[str, Path]] = None,
                 class_names: Optional[List[str]] = None,
                 color_mapping: Optional[Dict[int, Tuple[int, int, int]]] = None,
                 mask_format: str = 'indexed',  # 'indexed' or 'rgb'
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            mask_dir: Directory containing masks
            annotations_file: CSV file with image and mask paths (optional)
            class_names: List of class names
            color_mapping: Mapping from class ID to RGB color for RGB masks
            mask_format: 'indexed' for class indices, 'rgb' for RGB masks
            transform: Image transformations
            target_transform: Target transformations
        """
        # Handle Kaggle environment paths
        if is_kaggle_environment():
            data_dir = str(data_dir).replace('c:\\', '/kaggle/input/').replace('\\', '/')
            mask_dir = str(mask_dir).replace('c:\\', '/kaggle/input/').replace('\\', '/')
        
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_format = mask_format
        self.color_mapping = color_mapping or {}
        self.transform = transform
        self.target_transform = target_transform
        
        if annotations_file is not None:
            # Load from CSV
            self.samples = self._load_from_csv(annotations_file)
        else:
            # Load from folder structure
            self.samples = self._load_from_folders()
        
        # Set up class mapping
        if class_names is not None:
            self.class_names = class_names
        else:
            # Extract class names from samples or use default
            unique_classes = self._get_unique_classes()
            self.class_names = [f"class_{i}" for i in unique_classes]
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
    
    def _load_from_csv(self, annotations_file: Union[str, Path]) -> List[tuple]:
        """Load samples from CSV file.
        
        Expected CSV format:
        - image_path, mask_path
        """
        df = pd.read_csv(annotations_file)
        
        image_col = df.columns[0]
        mask_col = df.columns[1]
        
        samples = []
        for _, row in df.iterrows():
            image_path = self.data_dir / row[image_col]
            mask_path = self.mask_dir / row[mask_col]
            
            if image_path.exists() and mask_path.exists():
                samples.append((str(image_path), str(mask_path)))
        
        return samples
    
    def _load_from_folders(self) -> List[tuple]:
        """Load samples from folder structure.
        
        Expected structure:
        data_dir/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
        mask_dir/
        ├── image1.png
        ├── image2.png
        └── ...
        """
        samples = []
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend(self.data_dir.glob(f'*{ext}'))
        
        for image_file in image_files:
            # Find corresponding mask file
            mask_file = self.mask_dir / f"{image_file.stem}.png"
            if mask_file.exists():
                samples.append((str(image_file), str(mask_file)))
        
        return samples
    
    def _get_unique_classes(self) -> List[int]:
        """Get unique class indices from masks."""
        unique_classes = set()
        
        for image_path, mask_path in self.samples[:10]:  # Sample first 10 masks
            try:
                mask = self._load_mask(mask_path)
                unique_classes.update(np.unique(mask))
            except Exception:
                continue
        
        return sorted(list(unique_classes))
    
    def _load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """Load and process mask.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Processed mask as numpy array
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.mask_format == 'rgb':
            # Convert RGB mask to indexed mask
            mask = self._rgb_to_indexed(mask)
        
        return mask
    
    def _rgb_to_indexed(self, mask: np.ndarray) -> np.ndarray:
        """Convert RGB mask to indexed mask.
        
        Args:
            mask: RGB mask (H, W, 3) or grayscale mask (H, W)
            
        Returns:
            Indexed mask (H, W)
        """
        if len(mask.shape) == 2:
            # Already grayscale, treat as indexed
            return mask
        
        # Convert RGB to indexed
        indexed_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        
        for class_id, color in self.color_mapping.items():
            # Find pixels matching this color
            color_mask = np.all(mask == color, axis=2)
            indexed_mask[color_mask] = class_id
        
        return indexed_mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get item by index.
        
        Returns:
            Tuple of (image, mask)
        """
        image_path, mask_path = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Load mask
        try:
            mask = self._load_mask(mask_path)
            mask = Image.fromarray(mask)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            mask = Image.new('L', (224, 224), 0)
        
        # Apply transforms
        if self.transform is not None:
            # Check if this is an AlbumentationsSegmentationTransform
            if hasattr(self.transform, '__class__') and 'AlbumentationsSegmentationTransform' in self.transform.__class__.__name__:
                # Albumentations segmentation transform expects (image, mask)
                image_tensor, mask_tensor = self.transform(image, mask)
                return image_tensor, mask_tensor
            elif hasattr(self.transform, 'transforms'):
                # Standard Albumentations transform
                transformed = self.transform(image=np.array(image), mask=np.array(mask))
                image = Image.fromarray(transformed['image'])
                mask = Image.fromarray(transformed['mask'])
            else:
                # torchvision-style transform (apply to image only)
                image = self.transform(image)
                # Resize mask to match image if needed
                if hasattr(image, 'size'):
                    mask = mask.resize(image.size, Image.NEAREST)
        
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        return image, mask


def create_segmentation_dataset(config: Dict[str, Any]) -> Dataset:
    """Create segmentation dataset from config.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Dataset instance
    """
    data_dir = config['data_dir']
    mask_dir = config['mask_dir']
    
    # Handle Kaggle environment paths
    if is_kaggle_environment():
        # Convert Windows paths to Kaggle paths
        data_dir = str(data_dir).replace('c:\\', '/kaggle/input/').replace('\\', '/')
        mask_dir = str(mask_dir).replace('c:\\', '/kaggle/input/').replace('\\', '/')
        
        # Handle validation paths if they exist
        if 'val_data_dir' in config:
            config['val_data_dir'] = str(config['val_data_dir']).replace('c:\\', '/kaggle/input/').replace('\\', '/')
        if 'val_mask_dir' in config:
            config['val_mask_dir'] = str(config['val_mask_dir']).replace('c:\\', '/kaggle/input/').replace('\\', '/')
    
    annotations_file = config.get('annotations_file')
    class_names = config.get('class_names')
    color_mapping = config.get('color_mapping')
    mask_format = config.get('mask_format', 'indexed')
    transform = config.get('transform')
    target_transform = config.get('target_transform')
    
    return SegmentationDataset(
        data_dir=data_dir,
        mask_dir=mask_dir,
        annotations_file=annotations_file,
        class_names=class_names,
        color_mapping=color_mapping,
        mask_format=mask_format,
        transform=transform,
        target_transform=target_transform
    )
