#!/usr/bin/env python3
"""Kaggle setup script for ML Framework."""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.kaggle_utils import setup_kaggle_environment, print_kaggle_info, is_kaggle_environment


def install_dependencies():
    """Install required dependencies for Kaggle."""
    print("Installing dependencies...")
    
    packages = [
        "timm",
        "segmentation-models-pytorch", 
        "albumentations",
        "torchmetrics",
        "omegaconf"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            # Continue with other packages


def clone_repository(repo_url: Optional[str] = None):
    """Clone the repository if not already present."""
    if not is_kaggle_environment():
        print("Not running in Kaggle environment")
        return
    
    working_dir = Path('/kaggle/working')
    framework_dir = working_dir / 'ML_Framework'
    
    if framework_dir.exists():
        print(f"Framework already exists at {framework_dir}")
        return framework_dir
    
    if repo_url:
        print(f"Cloning repository from {repo_url}...")
        try:
            subprocess.run(['git', 'clone', repo_url, str(framework_dir)], check=True)
            print(f"✓ Repository cloned to {framework_dir}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone repository: {e}")
            return None
    else:
        print("No repository URL provided. Please upload your framework manually.")
        return None
    
    return framework_dir


def setup_framework_environment(framework_dir: Path):
    """Setup the framework environment."""
    if not framework_dir or not framework_dir.exists():
        print("Framework directory not found")
        return False
    
    # Change to framework directory
    os.chdir(framework_dir)
    
    # Add framework to Python path
    sys.path.insert(0, str(framework_dir))
    
    print(f"✓ Framework setup complete at {framework_dir}")
    return True


def create_kaggle_config(config_template: str = "segmentation_kaggle_gpu.yaml", 
                        dataset_name: str = "your-dataset"):
    """Create a Kaggle-specific configuration file."""
    config_path = Path('/kaggle/working/kaggle_config.yaml')
    
    # Read the template config
    template_path = Path('/kaggle/working/ML_Framework/configs') / config_template
    if not template_path.exists():
        print(f"Template config not found: {template_path}")
        return None
    
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Replace dataset name placeholder
    config_content = config_content.replace('your-dataset', dataset_name)
    
    # Save the config
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Created Kaggle config at {config_path}")
    return config_path


def run_training(config_path: Path):
    """Run the training script."""
    if not config_path or not config_path.exists():
        print("Config file not found")
        return False
    
    print("Starting training...")
    try:
        # Change to framework directory
        framework_dir = Path('/kaggle/working/ML_Framework')
        os.chdir(framework_dir)
        
        # Run training
        result = subprocess.run([
            sys.executable, 'scripts/train.py',
            '--config', str(config_path),
            '--device', 'cuda'
        ], capture_output=True, text=True)
        
        print("Training output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def save_results():
    """Save training results to working directory."""
    working_dir = Path('/kaggle/working')
    results_dir = working_dir / 'final_results'
    
    # Create results directory
    results_dir.mkdir(exist_ok=True)
    
    # Copy important files
    important_patterns = ['*.pt', '*.png', '*.json', '*.csv', '*.log']
    
    for pattern in important_patterns:
        for file_path in working_dir.rglob(pattern):
            if file_path.is_file():
                relative_path = file_path.relative_to(working_dir)
                dest_path = results_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
    
    print(f"✓ Results saved to {results_dir}")
    
    # List saved files
    print("Saved files:")
    for item in results_dir.rglob('*'):
        if item.is_file():
            print(f"  - {item.relative_to(results_dir)}")


def main():
    """Main setup and training function."""
    print("=== Kaggle ML Framework Setup ===")
    
    # Check if running in Kaggle
    if not is_kaggle_environment():
        print("Warning: Not running in Kaggle environment")
        print("This script is designed for Kaggle notebooks")
        return
    
    # Print Kaggle info
    print_kaggle_info()
    
    # Setup Kaggle environment
    working_dir = setup_kaggle_environment()
    print(f"Working directory: {working_dir}")
    
    # Install dependencies
    install_dependencies()
    
    # Clone repository (you'll need to provide your repo URL)
    repo_url = None  # Replace with your GitHub repository URL
    framework_dir = clone_repository(repo_url)
    
    # Setup framework environment
    if not setup_framework_environment(framework_dir):
        print("Failed to setup framework environment")
        return
    
    # Create Kaggle config
    dataset_name = "your-dataset"  # Replace with your actual dataset name
    config_path = create_kaggle_config(dataset_name=dataset_name)
    
    if not config_path:
        print("Failed to create config file")
        return
    
    # Run training
    success = run_training(config_path)
    
    if success:
        print("Training completed successfully!")
        save_results()
    else:
        print("Training failed!")
    
    print("=== Setup Complete ===")


if __name__ == '__main__':
    main()
