"""Classification dataset implementations."""

import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ImageClassificationDataset(Dataset):
    """Single-label image classification dataset.
    
    Supports both folder-based structure and CSV-based annotations.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 annotations_file: Optional[Union[str, Path]] = None,
                 class_names: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            annotations_file: CSV file with image paths and labels (optional)
            class_names: List of class names (optional)
            transform: Image transformations
            target_transform: Target transformations
        """
        self.data_dir = Path(data_dir)
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
            # Extract class names from samples
            unique_classes = sorted(set(label for _, label in self.samples))
            self.class_names = [f"class_{i}" for i in unique_classes]
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
    
    def _load_from_csv(self, annotations_file: Union[str, Path]) -> List[tuple]:
        """Load samples from CSV file.
        
        Expected CSV format:
        - image_path, label
        - or image_path, label1, label2, ... (for multi-label)
        """
        df = pd.read_csv(annotations_file)
        
        # Assume first column is image path, rest are labels
        image_col = df.columns[0]
        label_cols = df.columns[1:]
        
        samples = []
        for _, row in df.iterrows():
            image_path = self.data_dir / row[image_col]
            if image_path.exists():
                if len(label_cols) == 1:
                    # Single label
                    label = int(row[label_cols[0]])
                else:
                    # Multi-label (convert to list)
                    label = [int(row[col]) for col in label_cols]
                samples.append((str(image_path), label))
        
        return samples
    
    def _load_from_folders(self) -> List[tuple]:
        """Load samples from folder structure.
        
        Expected structure:
        data_dir/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image3.jpg
            └── image4.jpg
        """
        samples = []
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            for image_file in class_dir.iterdir():
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    samples.append((str(image_file), class_name))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get item by index.
        
        Returns:
            Tuple of (image, label)
        """
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Convert label to index if it's a string
        if isinstance(label, str):
            label = self.class_to_idx[label]
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label


class MultiLabelClassificationDataset(Dataset):
    """Multi-label image classification dataset.
    
    Loads from CSV with multiple label columns.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 annotations_file: Union[str, Path],
                 class_names: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            annotations_file: CSV file with image paths and labels
            class_names: List of class names
            transform: Image transformations
            target_transform: Target transformations
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Load from CSV
        self.samples = self._load_from_csv(annotations_file)
        
        # Set up class mapping
        if class_names is not None:
            self.class_names = class_names
        else:
            # Extract class names from CSV columns
            df = pd.read_csv(annotations_file)
            self.class_names = list(df.columns[1:])  # Skip first column (image path)
        
        self.num_classes = len(self.class_names)
    
    def _load_from_csv(self, annotations_file: Union[str, Path]) -> List[tuple]:
        """Load samples from CSV file."""
        df = pd.read_csv(annotations_file)
        
        image_col = df.columns[0]
        label_cols = df.columns[1:]
        
        samples = []
        for _, row in df.iterrows():
            image_path = self.data_dir / row[image_col]
            if image_path.exists():
                # Multi-label: convert to binary vector
                labels = [int(row[col]) for col in label_cols]
                samples.append((str(image_path), labels))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get item by index.
        
        Returns:
            Tuple of (image, labels) where labels is a binary vector
        """
        image_path, labels = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        
        return image, labels


def create_classification_dataset(config: Dict[str, Any]) -> Dataset:
    """Create classification dataset from config.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Dataset instance
    """
    dataset_type = config.get('dataset_type', 'classification')
    data_dir = config['data_dir']
    annotations_file = config.get('annotations_file')
    class_names = config.get('class_names')
    transform = config.get('transform')
    target_transform = config.get('target_transform')
    
    if dataset_type == 'multi_label':
        return MultiLabelClassificationDataset(
            data_dir=data_dir,
            annotations_file=annotations_file,
            class_names=class_names,
            transform=transform,
            target_transform=target_transform
        )
    else:
        return ImageClassificationDataset(
            data_dir=data_dir,
            annotations_file=annotations_file,
            class_names=class_names,
            transform=transform,
            target_transform=target_transform
        )
