"""Google Colab-specific utilities for ML Framework."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def is_colab_environment() -> bool:
    """Check if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_colab_paths() -> Dict[str, Path]:
    """Get Colab-specific paths.
    
    Returns:
        Dictionary with Colab path mappings
    """
    return {
        'content_dir': Path('/content'),
        'drive_dir': Path('/content/drive/MyDrive'),
        'working_dir': Path('/content')
    }


def setup_colab_environment() -> Path:
    """Setup Colab environment and create necessary directories.
    
    Returns:
        Path to working directory
    """
    if not is_colab_environment():
        raise RuntimeError("Not running in Colab environment")
    
    working_dir = Path('/content')
    
    # Create necessary directories
    directories = [
        'outputs',
        'checkpoints', 
        'logs',
        'plots',
        'samples',
        'results',
        'datasets'
    ]
    
    for directory in directories:
        (working_dir / directory).mkdir(exist_ok=True)
    
    return working_dir


def mount_google_drive():
    """Mount Google Drive in Colab.
    
    Returns:
        Path to mounted drive if successful, None otherwise
    """
    if not is_colab_environment():
        return None
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
        return Path('/content/drive/MyDrive')
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return None


def convert_paths_to_colab(config: Dict[str, Any], drive_path: Optional[str] = None) -> Dict[str, Any]:
    """Convert local paths in config to Colab paths.
    
    Args:
        config: Configuration dictionary
        drive_path: Optional path to Google Drive dataset location
        
    Returns:
        Configuration with converted paths
    """
    if not is_colab_environment():
        return config
    
    # Default dataset path (can be overridden)
    default_drive_path = drive_path or '/content/drive/MyDrive/datasets'
    default_content_path = '/content/datasets'
    
    # Apply path conversions
    if 'data' in config:
        data_config = config['data']
        
        # Convert Windows paths to Colab paths
        path_keys = ['train_data_dir', 'val_data_dir', 'test_data_dir', 
                    'data_dir', 'mask_dir', 'val_data_dir', 'val_mask_dir',
                    'test_data_dir', 'test_mask_dir']
        
        for key in path_keys:
            if key in data_config:
                current_path = str(data_config[key])
                
                # If path contains Windows paths, convert to Colab paths
                if 'c:\\' in current_path.lower() or 'C:\\' in current_path:
                    # Extract dataset name from path
                    # Example: c:\Facultate\...\Dataset\2025_11_03\all\herlev_grouped\train
                    # Try to find dataset folder structure
                    if 'datasets' in current_path.lower() or 'dataset' in current_path.lower():
                        # Try to use Google Drive path first, then content path
                        if Path(default_drive_path).exists():
                            # Extract relative path after dataset folder
                            parts = Path(current_path).parts
                            if 'dataset' in parts or 'Dataset' in parts:
                                idx = next((i for i, p in enumerate(parts) if 'dataset' in p.lower()), -1)
                                if idx >= 0:
                                    relative_path = Path(*parts[idx+1:])
                                    new_path = str(Path(default_drive_path) / relative_path)
                                    data_config[key] = new_path
                                    print(f"Converted {key}: {current_path} -> {new_path}")
                        else:
                            # Use content path
                            relative_path = Path(current_path).name
                            new_path = str(Path(default_content_path) / relative_path)
                            data_config[key] = new_path
                            print(f"Converted {key}: {current_path} -> {new_path}")
    
    return config


def get_colab_optimized_dataloader_config() -> Dict[str, Any]:
    """Get Colab-optimized dataloader configuration.
    
    Returns:
        Optimized dataloader configuration for Colab GPU
    """
    return {
        'batch_size': 32,  # Good for T4 GPU
        'num_workers': 2,  # Colab has limited CPU cores
        'pin_memory': True,  # Enable for GPU
        'persistent_workers': True,
        'prefetch_factor': 2,
        'drop_last': False
    }


def get_colab_optimized_training_config() -> Dict[str, Any]:
    """Get Colab-optimized training configuration.
    
    Returns:
        Optimized training configuration for Colab
    """
    return {
        'epochs': 50,  # Reasonable for Colab sessions
        'amp': True,  # Enable mixed precision for GPU
        'gradient_clip_norm': 1.0,
        'gradient_accumulation_steps': 1
    }


def get_colab_optimized_callbacks_config() -> Dict[str, Any]:
    """Get Colab-optimized callbacks configuration.
    
    Returns:
        Optimized callbacks configuration for Colab
    """
    return {
        'checkpoint': {
            'monitor': 'val_loss',
            'mode': 'min',
            'save_top_k': 3,
            'enabled': True,
            'save_last': True,
            'save_best': True
        },
        'early_stopping': {
            'monitor': 'val_loss',
            'mode': 'min',
            'patience': 10,
            'enabled': True
        },
        'sample_visualizer': {
            'num_samples': 8,
            'save_every_n_epochs': 5,
            'enabled': True
        },
        'confusion_matrix': {
            'save_every_n_epochs': 10,
            'enabled': True
        },
        'learning_rate': {
            'save_every_n_epochs': 5,
            'enabled': True
        }
    }


def cleanup_colab_memory():
    """Clean up GPU memory in Colab environment."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()


def print_colab_info():
    """Print Colab environment information."""
    if not is_colab_environment():
        print("Not running in Colab environment")
        return
    
    print("=== Google Colab Environment Info ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check Google Drive
    drive_path = Path('/content/drive/MyDrive')
    if drive_path.exists():
        print(f"\nGoogle Drive mounted: {drive_path}")
        
        # Check for datasets
        datasets_path = drive_path / 'datasets'
        if datasets_path.exists():
            print(f"Datasets directory found: {datasets_path}")
            # List dataset folders
            dataset_folders = [d for d in datasets_path.iterdir() if d.is_dir()]
            if dataset_folders:
                print("Available dataset folders:")
                for folder in dataset_folders[:10]:  # Show first 10
                    print(f"  - {folder.name}")
                if len(dataset_folders) > 10:
                    print(f"  ... and {len(dataset_folders) - 10} more")
    else:
        print("\nGoogle Drive not mounted. Run mount_google_drive() to mount it.")
    
    # Print working directory contents
    working_dir = Path('/content')
    if working_dir.exists():
        print(f"\nWorking directory contents:")
        for item in sorted(working_dir.iterdir())[:10]:
            if item.is_dir():
                print(f"  - {item.name}/")
            else:
                print(f"  - {item.name}")
        if len(list(working_dir.iterdir())) > 10:
            print(f"  ... and {len(list(working_dir.iterdir())) - 10} more items")


def setup_colab_notebook():
    """Complete setup for Colab notebook."""
    if not is_colab_environment():
        print("Warning: Not running in Colab environment")
        return None
    
    print("Setting up Colab environment...")
    
    # Setup directories
    working_dir = setup_colab_environment()
    print(f"Created working directory: {working_dir}")
    
    # Print environment info
    print_colab_info()
    
    # Install additional packages if needed
    try:
        import timm
        import segmentation_models_pytorch as smp
        print("Required packages already installed")
    except ImportError:
        print("Installing required packages...")
        os.system("pip install timm segmentation-models-pytorch albumentations torchmetrics omegaconf")
    
    print("Colab setup complete!")
    return working_dir

