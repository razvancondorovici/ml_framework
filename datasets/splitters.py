"""Dataset splitting utilities."""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import numpy as np


def random_split(samples: List[Any], 
                train_ratio: float = 0.8,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1,
                random_state: int = 42) -> Tuple[List[Any], List[Any], List[Any]]:
    """Random split of samples into train/val/test sets.
    
    Args:
        samples: List of samples to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_samples, temp_samples = train_test_split(
        samples, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=val_ratio_adjusted,
        random_state=random_state
    )
    
    return train_samples, val_samples, test_samples


def stratified_split(samples: List[Any], 
                    labels: List[Any],
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    test_ratio: float = 0.1,
                    random_state: int = 42) -> Tuple[List[Any], List[Any], List[Any]]:
    """Stratified split of samples into train/val/test sets.
    
    Args:
        samples: List of samples to split
        labels: List of labels for stratification
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert len(samples) == len(labels), "Samples and labels must have same length"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # First split: train vs (val + test)
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, encoded_labels,
        train_size=train_ratio,
        stratify=encoded_labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples, val_labels, test_labels = train_test_split(
        temp_samples, temp_labels,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=random_state
    )
    
    return train_samples, val_samples, test_samples


def file_list_split(file_list_path: Union[str, Path],
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   random_state: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Split based on file list.
    
    Args:
        file_list_path: Path to file containing list of files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    return random_split(files, train_ratio, val_ratio, test_ratio, random_state)


def csv_split(csv_path: Union[str, Path],
             image_col: str = 'image_path',
             label_col: str = 'label',
             train_ratio: float = 0.8,
             val_ratio: float = 0.1,
             test_ratio: float = 0.1,
             random_state: int = 42,
             stratified: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split CSV-based dataset.
    
    Args:
        csv_path: Path to CSV file
        image_col: Name of image path column
        label_col: Name of label column
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        stratified: Whether to use stratified splitting
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)
    
    if stratified:
        train_df, val_df, test_df = stratified_split(
            df[image_col].tolist(),
            df[label_col].tolist(),
            train_ratio, val_ratio, test_ratio, random_state
        )
    else:
        train_df, val_df, test_df = random_split(
            df[image_col].tolist(),
            train_ratio, val_ratio, test_ratio, random_state
        )
    
    # Convert back to DataFrames
    train_df = df[df[image_col].isin(train_df)].reset_index(drop=True)
    val_df = df[df[image_col].isin(val_df)].reset_index(drop=True)
    test_df = df[df[image_col].isin(test_df)].reset_index(drop=True)
    
    return train_df, val_df, test_df


def external_split(train_dir: Union[str, Path],
                  val_dir: Union[str, Path],
                  test_dir: Optional[Union[str, Path]] = None) -> Tuple[List[str], List[str], List[str]]:
    """Split based on external directories.
    
    Args:
        train_dir: Directory containing training files
        val_dir: Directory containing validation files
        test_dir: Directory containing test files (optional)
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    
    # Get all files from train directory
    train_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        train_files.extend([str(f) for f in train_dir.glob(f'**/*{ext}')])
    
    # Get all files from val directory
    val_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        val_files.extend([str(f) for f in val_dir.glob(f'**/*{ext}')])
    
    # Get test files if directory provided
    test_files = []
    if test_dir is not None:
        test_dir = Path(test_dir)
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            test_files.extend([str(f) for f in test_dir.glob(f'**/*{ext}')])
    
    return train_files, val_files, test_files


def create_data_splits(config: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """Create data splits based on configuration.
    
    Args:
        config: Data splitting configuration
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    split_method = config.get('split_method', 'random')
    
    if split_method == 'random':
        return random_split(
            config['samples'],
            config.get('train_ratio', 0.8),
            config.get('val_ratio', 0.1),
            config.get('test_ratio', 0.1),
            config.get('random_state', 42)
        )
    
    elif split_method == 'stratified':
        return stratified_split(
            config['samples'],
            config['labels'],
            config.get('train_ratio', 0.8),
            config.get('val_ratio', 0.1),
            config.get('test_ratio', 0.1),
            config.get('random_state', 42)
        )
    
    elif split_method == 'file_list':
        return file_list_split(
            config['file_list_path'],
            config.get('train_ratio', 0.8),
            config.get('val_ratio', 0.1),
            config.get('test_ratio', 0.1),
            config.get('random_state', 42)
        )
    
    elif split_method == 'external':
        return external_split(
            config['train_dir'],
            config['val_dir'],
            config.get('test_dir')
        )
    
    else:
        raise ValueError(f"Unknown split method: {split_method}")
