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
def safe_reclone_repository(branch_name=None):
    """Safely remove and re-clone the repository, optionally switch to a branch.
    
    Args:
        branch_name: Name of the branch to switch to (None = use default/main branch)
    """
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
        
        # Switch to branch if specified
        if branch_name:
            print(f"Switching to branch: {branch_name}")
            result = os.system(f'git checkout {branch_name}')
            if result == 0:
                print(f"✅ Successfully switched to branch: {branch_name}")
                # Show current branch
                current_branch = os.popen('git branch --show-current').read().strip()
                print(f"Current branch: {current_branch}")
            else:
                print(f"⚠️ Failed to switch to branch: {branch_name}")
                print("Available branches:")
                os.system('git branch -a')
        else:
            # Show current branch
            current_branch = os.popen('git branch --show-current').read().strip()
            print(f"Current branch: {current_branch}")
        
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

# Clone repository and switch to branch
# Set branch_name to None to use the default branch, or specify a branch name
branch_name = None  # Change this to your branch name, e.g., "dev", "feature/new-model", etc.
success = safe_reclone_repository(branch_name=branch_name)

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
drive_path = mount_google_drive()
if drive_path:
    print(f"✅ Google Drive mounted at: {drive_path}")
else:
    print("⚠️ Google Drive mounting failed or not needed")

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

from pathlib import Path
from google.colab import files
import zipfile
import os

# Find the latest run for the current experiment
runs_dir = Path('/content/ml_framework/runs')
if not runs_dir.exists():
    print("⚠️ No runs directory found")
else:
    run_folders = [f for f in runs_dir.iterdir() if f.is_dir()]
    if not run_folders:
        print("⚠️ No run folders found")
    else:
        # Get the most recent run (by modification time)
        latest_run = max(run_folders, key=lambda x: x.stat().st_mtime)
        experiment_name = latest_run.name
        print(f"📁 Latest run: {experiment_name}")
        
        # Look for experiment folders within the run
        experiment_folders = [f for f in latest_run.iterdir() if f.is_dir()]
        if not experiment_folders:
            print("⚠️ No experiment folders found in latest run")
        else:
            # Get the most recent experiment folder (timestamp folder)
            latest_experiment = max(experiment_folders, key=lambda x: x.stat().st_mtime)
            print(f"📁 Latest experiment: {latest_experiment.name}")
            
            # Collect files to include in zip
            files_to_zip = []
            
            # 1. Best model checkpoint
            best_ckpt = latest_experiment / 'checkpoints' / 'best.pt'
            if best_ckpt.exists():
                files_to_zip.append(('checkpoints/best.pt', best_ckpt))
                print("✅ Found: best.pt")
            else:
                print("⚠️ best.pt not found")
            
            # 2. Latest accuracy_curves_epoch_xxx.png
            plots_dir = latest_experiment / 'plots'
            if plots_dir.exists():
                accuracy_curves = list(plots_dir.glob('accuracy_curves_epoch_*.png'))
                if accuracy_curves:
                    # Sort by epoch number (extract from filename)
                    def get_epoch_num(path):
                        import re
                        match = re.search(r'epoch_(\d+)', path.name)
                        return int(match.group(1)) if match else 0
                    
                    latest_accuracy_curve = max(accuracy_curves, key=get_epoch_num)
                    files_to_zip.append(('plots/' + latest_accuracy_curve.name, latest_accuracy_curve))
                    print(f"✅ Found: {latest_accuracy_curve.name}")
                else:
                    print("⚠️ No accuracy_curves_epoch_*.png files found")
            else:
                print("⚠️ plots directory not found")
            
            # 3. scalars.csv
            scalars_csv = latest_experiment / 'scalars.csv'
            if scalars_csv.exists():
                files_to_zip.append(('scalars.csv', scalars_csv))
                print("✅ Found: scalars.csv")
            else:
                print("⚠️ scalars.csv not found")
            
            # 4. config.yaml
            config_yaml = latest_experiment / 'config.yaml'
            if config_yaml.exists():
                files_to_zip.append(('config.yaml', config_yaml))
                print("✅ Found: config.yaml")
            else:
                print("⚠️ config.yaml not found")
            
            # Create zip file if we have files to include
            if files_to_zip:
                # Create zip filename based on experiment name
                zip_filename = f"{experiment_name}.zip"
                zip_path = Path('/content') / zip_filename
                
                print(f"\n📦 Creating zip file: {zip_filename}")
                
                # Create zip file
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for arcname, file_path in files_to_zip:
                        zipf.write(file_path, arcname)
                        print(f"  Added: {arcname}")
                
                zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
                print(f"\n✅ Zip file created: {zip_filename} ({zip_size_mb:.2f} MB)")
                
                # Download the zip file
                print(f"\n📥 Downloading {zip_filename}...")
                try:
                    files.download(str(zip_path))
                    print(f"✅ Successfully downloaded: {zip_filename}")
                    
                    # Optionally clean up the zip file after download
                    # Uncomment the next line if you want to delete it after downloading
                    # zip_path.unlink()
                    # print(f"🧹 Cleaned up: {zip_filename}")
                except Exception as e:
                    print(f"❌ Failed to download {zip_filename}: {e}")
            else:
                print("\n⚠️ No files found to zip")

print("\n✅ Download complete!")