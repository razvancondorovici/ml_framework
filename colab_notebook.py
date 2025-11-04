"""
Google Colab Notebook Template for ML Framework
Copy the cells below into separate Colab notebook cells.
"""

import torch
import sys
import os
from pathlib import Path

# =============================================================================
# CELL 1: Enable GPU and Check Environment
# =============================================================================
# Make sure to enable GPU in Colab: Runtime -> Change runtime type -> GPU

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ GPU not available! Please enable GPU: Runtime -> Change runtime type -> GPU")

# =============================================================================
# CELL 2: Install Dependencies
# =============================================================================

# Install required packages
!pip install timm segmentation-models-pytorch albumentations torchmetrics omegaconf PyYAML matplotlib seaborn scikit-learn pandas Pillow opencv-python tqdm

# Verify installation
try:
    import timm
    import torchmetrics
    print("✅ All packages installed successfully!")
except ImportError as e:
    print(f"❌ Package installation failed: {e}")

# =============================================================================
# CELL 3: Clone Repository and Setup Framework
# =============================================================================
# Replace 'YOUR_GITHUB_USERNAME' and 'YOUR_REPO_NAME' with your actual values
# Or use the direct repository URL

# Function to safely re-clone repository
def safe_reclone_repository():
    """Safely remove and re-clone the repository."""
    original_cwd = os.getcwd()
    safe_dir = '/content'
    
    try:
        os.chdir(safe_dir)
        
        # Remove existing framework directory
        import shutil
        framework_path = '/content/ml_framework'
        if os.path.exists(framework_path):
            shutil.rmtree(framework_path)
            print(f"Removed existing framework directory: {framework_path}")
        
        # Clone repository
        print("Cloning repository...")
        # UPDATE THIS URL with your repository
        repo_url = "https://github.com/razvancondorovici/ml_framework.git"  # Change this!
        os.system(f'git clone {repo_url} /content/ml_framework')
        
        # Change to framework directory
        os.chdir(framework_path)
        print(f"Changed to framework directory: {os.getcwd()}")
        
        # Add framework to Python path
        framework_path_str = '/content/ml_framework'
        if framework_path_str in sys.path:
            sys.path.remove(framework_path_str)
        sys.path.append(framework_path_str)
        
        return True
        
    except Exception as e:
        print(f"Error during repository clone: {e}")
        try:
            os.chdir(original_cwd)
        except:
            os.chdir('/content')
        return False

# Clone repository
success = safe_reclone_repository()

if success:
    # Import framework components
    from utils.colab_utils import setup_colab_environment, print_colab_info, is_colab_environment, mount_google_drive
    from utils.config import load_config
    print("✅ Framework setup complete!")
else:
    print("❌ Failed to setup framework. Please check the repository URL.")

# =============================================================================
# CELL 4: Mount Google Drive (Optional - if your dataset is on Drive)
# =============================================================================
# If your dataset is stored on Google Drive, uncomment and run this cell

# Mount Google Drive
# drive_path = mount_google_drive()
# if drive_path:
#     print(f"✅ Google Drive mounted at: {drive_path}")
# else:
#     print("⚠️ Google Drive mounting failed or not needed")

# =============================================================================
# CELL 5: Setup Colab Environment
# =============================================================================

# Setup Colab environment
if is_colab_environment():
    working_dir = setup_colab_environment()
    print_colab_info()
    print(f"✅ Colab environment ready!")
else:
    print("⚠️ Not running in Colab environment")

# =============================================================================
# CELL 6: Upload Dataset (Alternative - if not using Google Drive)
# =============================================================================
# If your dataset is not on Google Drive, you can upload it directly
# Uncomment the code below and run this cell

# from google.colab import files
# import zipfile
# 
# # Upload dataset zip file
# uploaded = files.upload()
# 
# # Extract dataset
# for filename in uploaded.keys():
#     if filename.endswith('.zip'):
#         with zipfile.ZipFile(filename, 'r') as zip_ref:
#             zip_ref.extractall('/content/datasets')
#         print(f"✅ Extracted {filename} to /content/datasets")
#     else:
#         print(f"⚠️ Skipped {filename} (not a zip file)")

# =============================================================================
# CELL 7: Create/Update Configuration for Colab
# =============================================================================
# Update the paths in your config file to match your Colab setup
# If using Google Drive: /content/drive/MyDrive/datasets/your_dataset/train
# If using uploaded data: /content/datasets/your_dataset/train

import yaml
from pathlib import Path

# Example: Create a Colab-specific config
# You can also manually edit the config file after cloning
config_path = Path('/content/ml_framework/configs/classification_cells_herlev.yaml')

if config_path.exists():
    print(f"✅ Config file found: {config_path}")
    print("📝 Update the data paths in the config file to match your Colab setup:")
    print("   - Google Drive: /content/drive/MyDrive/datasets/...")
    print("   - Uploaded: /content/datasets/...")
else:
    print("⚠️ Config file not found. Please create one or update the path above.")

# =============================================================================
# CELL 8: Run Training
# =============================================================================

import subprocess

# Update this path to your config file
config_file = '/content/ml_framework/configs/classification_cells_herlev.yaml'

# Optionally, convert paths automatically
from utils.colab_utils import convert_paths_to_colab, load_config
try:
    # Load and convert config
    config = load_config(config_file)
    config = convert_paths_to_colab(config)
    
    # Save converted config temporarily
    temp_config = '/content/ml_framework/configs/temp_colab_config.yaml'
    import yaml
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    config_file = temp_config
    print("✅ Paths converted for Colab")
except Exception as e:
    print(f"⚠️ Could not auto-convert paths: {e}")
    print("Please manually update the config file paths")

# Run training
print("🚀 Starting training...")
print(f"Using config: {config_file}")

result = subprocess.run([
    'python', '/content/ml_framework/scripts/train.py',
    '--config', config_file,
    '--device', 'cuda'
], capture_output=True, text=True)

print("=" * 60)
print("TRAINING OUTPUT:")
print("=" * 60)
print(result.stdout)

if result.stderr:
    print("=" * 60)
    print("ERRORS:")
    print("=" * 60)
    print(result.stderr)

if result.returncode == 0:
    print("✅ Training completed successfully!")
else:
    print(f"❌ Training failed with return code: {result.returncode}")

# =============================================================================
# CELL 9: Display Results and Download Checkpoints
# =============================================================================

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import json
from PIL import Image

# Find the latest run
runs_dir = Path('/content/ml_framework/runs')
if runs_dir.exists():
    run_folders = [f for f in runs_dir.iterdir() if f.is_dir()]
    if run_folders:
        # Get the most recent run
        latest_run = max(run_folders, key=lambda x: x.stat().st_mtime)
        print(f"📁 Latest run: {latest_run.name}")
        
        # Look for experiment folders
        experiment_folders = [f for f in latest_run.iterdir() if f.is_dir()]
        if experiment_folders:
            latest_experiment = max(experiment_folders, key=lambda x: x.stat().st_mtime)
            print(f"📁 Latest experiment: {latest_experiment.name}")
            
            # Display training history
            history_file = latest_experiment / 'training_history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Plot training curves
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Loss curves
                if 'train_loss' in history:
                    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
                if 'val_loss' in history:
                    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
                
                # Accuracy curves (classification)
                if 'train_accuracy' in history:
                    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy', marker='o')
                if 'val_accuracy' in history:
                    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', marker='s')
                axes[0, 1].set_title('Training Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
                
                # Sample visualizations
                samples_dir = latest_experiment / 'samples'
                if samples_dir.exists():
                    val_samples = list(samples_dir.glob('*_val.png'))
                    if val_samples:
                        latest_sample = max(val_samples, key=lambda x: x.stat().st_mtime)
                        img = Image.open(latest_sample)
                        axes[1, 0].imshow(img)
                        axes[1, 0].set_title(f'Latest Validation Sample')
                        axes[1, 0].axis('off')
                
                # Checkpoints info
                checkpoints_dir = latest_experiment / 'checkpoints'
                if checkpoints_dir.exists():
                    checkpoints = list(checkpoints_dir.glob('*.pt'))
                    axes[1, 1].text(0.1, 0.9, f"Checkpoints: {len(checkpoints)}", 
                                   transform=axes[1, 1].transAxes, fontsize=12)
                    axes[1, 1].text(0.1, 0.8, f"Best: best.pt", 
                                   transform=axes[1, 1].transAxes, fontsize=12)
                    axes[1, 1].text(0.1, 0.7, f"Last: last.pt", 
                                   transform=axes[1, 1].transAxes, fontsize=12)
                    axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.show()
            
            # List checkpoints for download
            checkpoints_dir = latest_experiment / 'checkpoints'
            if checkpoints_dir.exists():
                print("\n📦 Available checkpoints:")
                for ckpt in checkpoints_dir.glob('*.pt'):
                    size_mb = ckpt.stat().st_size / (1024 * 1024)
                    print(f"  - {ckpt.name} ({size_mb:.2f} MB)")
                
                print("\n💾 To download checkpoints, run:")
                print("   from google.colab import files")
                print(f"   files.download('{checkpoints_dir / 'best.pt'}')")
        else:
            print("⚠️ No experiment folders found")
    else:
        print("⚠️ No run folders found")
else:
    print("⚠️ No runs directory found")

print("\n✅ Notebook execution complete!")

