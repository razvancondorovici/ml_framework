import torch
import sys
import os
from pathlib import Path

# =============================================================================
# CELL 1: Check Environment
# =============================================================================

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# CELL 2: Clone Repository and Setup Framework
# =============================================================================
# This cell can be run multiple times safely:
# - First run: Installs packages and clones repository
# - Subsequent runs: Only re-clones repository (skips package installation)
# - Use this when you want to pull latest changes from your repository

# Function to safely re-clone repository
def safe_reclone_repository():
    """Safely remove and re-clone the repository, handling directory changes."""
    # Ensure we're in a safe directory before removing
    original_cwd = os.getcwd()
    safe_dir = '/kaggle/working'
    
    try:
        # Change to safe directory
        os.chdir(safe_dir)
        
        # Remove existing framework directory
        import shutil
        framework_path = '/kaggle/working/ML_Framework'
        if os.path.exists(framework_path):
            shutil.rmtree(framework_path)
            print(f"Removed existing framework directory: {framework_path}")
        
        # Clone repository
        print("Cloning repository...")
        os.system('git clone https://github.com/razvancondorovici/ml_framework.git /kaggle/working/ML_Framework')
        
        # Change to framework directory
        os.chdir(framework_path)
        print(f"Changed to framework directory: {os.getcwd()}")
        
        # Add framework to Python path (remove old path if exists)
        framework_path_str = '/kaggle/working/ML_Framework'
        if framework_path_str in sys.path:
            sys.path.remove(framework_path_str)
        sys.path.append(framework_path_str)
        
        return True
        
    except Exception as e:
        print(f"Error during repository re-clone: {e}")
        # Try to restore original directory
        try:
            os.chdir(original_cwd)
        except:
            os.chdir('/kaggle/working')
        return False

# Check if packages are already installed to avoid reinstalling
def check_packages_installed():
    """Check if required packages are already installed."""
    try:
        import timm
        import segmentation_models_pytorch
        import albumentations
        import omegaconf
        return True
    except ImportError:
        return False

# Only install packages if they're not already installed
if not check_packages_installed():
    print("Installing required packages...")
    # Step 1: Force install NumPy 1.x and ignore other requirements
    !pip install "numpy==1.26.4" --force-reinstall --no-deps

    # Step 2: Install packages WITHOUT letting them upgrade NumPy
    !pip install timm segmentation-models-pytorch albumentations omegaconf --no-deps

    # Step 3: Install only the missing dependencies (not NumPy)
    !pip install pillow tqdm scipy pydantic antlr4-python3-runtime filelock fsspec packaging httpx typer-slim typing-extensions
else:
    print("Required packages already installed, skipping installation.")

# Re-clone repository
success = safe_reclone_repository()

if success:
    # Import framework components
    from utils.kaggle_utils import setup_kaggle_environment, print_kaggle_info, is_kaggle_environment
    from utils.config import load_config
    print("Framework setup complete!")
else:
    print("Failed to setup framework. Please restart the notebook.")

# =============================================================================
# CELL 3: Setup Kaggle Environment
# =============================================================================

# Setup Kaggle environment
if is_kaggle_environment():
    working_dir = setup_kaggle_environment()
    print_kaggle_info()
    #print(f"Working directory: {working_dir}")

# =============================================================================
# CELL 5: Run Training
# =============================================================================

import subprocess

# Run training
print("Starting training...")
result = subprocess.run([
    'python', 'scripts/train.py',
    '--config', '/kaggle/working/ML_Framework/configs/segmentation_kaggle_gpu.yaml',
    '--device', 'cuda'
], capture_output=True, text=True)

print("Training output:")
print(result.stdout)

if result.stderr:
    print("Errors:")
    print(result.stderr)

# =============================================================================
# CELL 6: Save and Display Results
# =============================================================================

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import json
from PIL import Image

# Copy results to a permanent location
output_dir = Path('/kaggle/working/outputs')
results_dir = Path('/kaggle/working/final_results')

if output_dir.exists():
    shutil.copytree(output_dir, results_dir, dirs_exist_ok=True)

# List all important files
print("Training completed! Available outputs:")
important_extensions = ['.pt', '.png', '.json', '.csv', '.log']

for item in Path('/kaggle/working').rglob('*'):
    if item.is_file() and item.suffix in important_extensions:
        print(f"  {item}")

# Display training history plots
history_files = list(Path('/kaggle/working').rglob('training_history.json'))
if history_files:
    with open(history_files[0], 'r') as f:
        history = json.load(f)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    if 'val_mean_iou' in history:
        ax2.plot(history['val_mean_iou'], label='Val mIoU')
    if 'val_dice' in history:
        ax2.plot(history['val_dice'], label='Val Dice')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Display latest validation sample
print("\n" + "="*50)
print("LATEST VALIDATION SAMPLE")
print("="*50)

# Find the most recent run folder
runs_dir = Path('/kaggle/working/ML_Framework/runs')
if runs_dir.exists():
    run_folders = [f for f in runs_dir.iterdir() if f.is_dir()]
    if run_folders:
        # Get the most recent run
        latest_run = max(run_folders, key=lambda x: x.stat().st_mtime)
        print(f"Latest run: {latest_run.name}")
        
        # Look for experiment folders within the run
        experiment_folders = [f for f in latest_run.iterdir() if f.is_dir()]
        if experiment_folders:
            # Get the most recent experiment
            latest_experiment = max(experiment_folders, key=lambda x: x.stat().st_mtime)
            print(f"Latest experiment: {latest_experiment.name}")
            
            # Look for samples in the experiment folder
            samples_dir = latest_experiment / 'samples'
            if samples_dir.exists():
                # Find validation sample files (files ending with _val.png)
                val_sample_files = list(samples_dir.glob('*_val.png'))
                
                if val_sample_files:
                    # Sort by modification time and get the latest
                    latest_val_sample = max(val_sample_files, key=lambda x: x.stat().st_mtime)
                    print(f"Latest validation sample: {latest_val_sample.name}")
                    
                    # Display the validation sample
                    try:
                        img = Image.open(latest_val_sample)
                        plt.figure(figsize=(12, 8))
                        plt.imshow(img)
                        plt.title(f'Latest Validation Sample\n{latest_val_sample.name}')
                        plt.axis('off')
                        plt.show()
                    except Exception as e:
                        print(f"Error displaying image: {e}")
                else:
                    print("No validation sample files (*_val.png) found")
                    
                    # Debug: show what files are actually in the samples directory
                    all_files = list(samples_dir.glob('*'))
                    if all_files:
                        print(f"Files found in samples directory ({len(all_files)}):")
                        for file in all_files:
                            print(f"  - {file.name}")
                    else:
                        print("No files found in samples directory")
            else:
                print("No samples directory found in latest experiment")
                
                # Debug: show what directories exist in the experiment
                experiment_contents = list(latest_experiment.iterdir())
                if experiment_contents:
                    print(f"Contents of latest experiment:")
                    for item in experiment_contents:
                        print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        else:
            print("No experiment folders found in latest run")
    else:
        print("No run folders found")
else:
    print("No runs directory found")

print("\nNotebook execution complete!")