"""Kaggle-specific utilities for ML Framework."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def is_kaggle_environment() -> bool:
    """Check if running in Kaggle environment.
    
    Returns:
        True if running in Kaggle, False otherwise
    """
    return os.path.exists('/kaggle') and 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def get_kaggle_paths() -> Dict[str, Path]:
    """Get Kaggle-specific paths.
    
    Returns:
        Dictionary with Kaggle path mappings
    """
    return {
        'input_dir': Path('/kaggle/input'),
        'working_dir': Path('/kaggle/working'),
        'temp_dir': Path('/kaggle/temp')
    }


def setup_kaggle_environment() -> Path:
    """Setup Kaggle environment and create necessary directories.
    
    Returns:
        Path to working directory
    """
    if not is_kaggle_environment():
        raise RuntimeError("Not running in Kaggle environment")
    
    working_dir = Path('/kaggle/working')
    
    # Create necessary directories
    directories = [
        'outputs',
        'checkpoints', 
        'logs',
        'plots',
        'samples',
        'results'
    ]
    
    for directory in directories:
        (working_dir / directory).mkdir(exist_ok=True)
    
    return working_dir


def convert_paths_to_kaggle(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert local paths in config to Kaggle paths.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with converted paths
    """
    if not is_kaggle_environment():
        return config
    
    # Path mappings for common local to Kaggle conversions
    path_mappings = {
        'data_dir': '/kaggle/input/your-dataset/images',
        'mask_dir': '/kaggle/input/your-dataset/masks', 
        'val_data_dir': '/kaggle/input/your-dataset/val/images',
        'val_mask_dir': '/kaggle/input/your-dataset/val/masks'
    }
    
    # Apply path conversions
    if 'data' in config:
        data_config = config['data']
        for key, default_path in path_mappings.items():
            if key in data_config:
                # If path contains local paths, replace with Kaggle paths
                current_path = str(data_config[key])
                if 'c:\\' in current_path.lower() or '/home/' in current_path:
                    data_config[key] = default_path
    
    return config


def get_kaggle_optimized_dataloader_config() -> Dict[str, Any]:
    """Get Kaggle-optimized dataloader configuration.
    
    Returns:
        Optimized dataloader configuration for Kaggle
    """
    return {
        'batch_size': 16,  # Increased for GPU
        'num_workers': 2,  # Limited CPU cores on Kaggle
        'pin_memory': True,  # Enable for GPU
        'persistent_workers': True,
        'prefetch_factor': 2,
        'drop_last': True
    }


def get_kaggle_optimized_training_config() -> Dict[str, Any]:
    """Get Kaggle-optimized training configuration.
    
    Returns:
        Optimized training configuration for Kaggle
    """
    return {
        'epochs': 30,  # Reduced for time limits
        'amp': True,  # Enable mixed precision for GPU
        'gradient_clip_norm': 1.0,
        'gradient_accumulation_steps': 1  # Reduce accumulation
    }


def get_kaggle_optimized_callbacks_config() -> Dict[str, Any]:
    """Get Kaggle-optimized callbacks configuration.
    
    Returns:
        Optimized callbacks configuration for Kaggle
    """
    return {
        'checkpoint': {
            'monitor': 'val_loss',
            'mode': 'min',
            'save_top_k': 2,  # Reduce to save space
            'enabled': True,
            'save_last': True,
            'save_best': True
        },
        'early_stopping': {
            'monitor': 'val_loss',
            'mode': 'min',
            'patience': 8,  # Reduce patience for faster convergence
            'enabled': True
        },
        'sample_visualizer': {
            'num_samples': 4,
            'save_every_n_epochs': 10,  # Less frequent to save time
            'enabled': True
        },
        'confusion_matrix': {
            'save_every_n_epochs': 15,
            'enabled': True
        },
        'learning_rate': {
            'save_every_n_epochs': 10,
            'enabled': True
        }
    }


def get_kaggle_optimized_transforms_config() -> Dict[str, Any]:
    """Get Kaggle-optimized transforms configuration.
    
    Returns:
        Optimized transforms configuration for Kaggle
    """
    return {
        'resize': 512,
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation': 15,
        'shift_scale_rotate': True,
        'elastic_transform': False,  # Disable heavy augmentations
        'grid_distortion': False,
        'color_jitter': True,
        'gaussian_noise': False,  # Disable to save time
        'gaussian_blur': False,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'use_albumentations': True
    }


def cleanup_kaggle_memory():
    """Clean up GPU memory in Kaggle environment."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()


def print_kaggle_info():
    """Print Kaggle environment information."""
    if not is_kaggle_environment():
        print("Not running in Kaggle environment")
        return
    
    print("=== Kaggle Environment Info ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Print available input datasets
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        print(f"\nAvailable datasets:")
        for dataset_dir in input_dir.iterdir():
            if dataset_dir.is_dir():
                print(f"  - {dataset_dir.name}")
    
    # Print working directory contents
    working_dir = Path('/kaggle/working')
    if working_dir.exists():
        print(f"\nWorking directory contents:")
        for item in working_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}/")
            else:
                print(f"  - {item.name}")


def setup_kaggle_notebook():
    """Complete setup for Kaggle notebook."""
    if not is_kaggle_environment():
        print("Warning: Not running in Kaggle environment")
        return
    
    print("Setting up Kaggle environment...")
    
    # Setup directories
    working_dir = setup_kaggle_environment()
    print(f"Created working directory: {working_dir}")
    
    # Print environment info
    print_kaggle_info()
    
    # Install additional packages if needed
    try:
        import timm
        import segmentation_models_pytorch as smp
        print("Required packages already installed")
    except ImportError:
        print("Installing required packages...")
        os.system("pip install timm segmentation-models-pytorch albumentations torchmetrics omegaconf")
    
    print("Kaggle setup complete!")
    return working_dir
