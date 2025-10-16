# Kaggle Notebook Template for ML Framework
# Copy this code into your Kaggle notebook cells

# =============================================================================
# CELL 1: Environment Setup and Dependencies
# =============================================================================

import torch
import sys
import os
from pathlib import Path

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Install required packages
!pip install timm segmentation-models-pytorch albumentations torchmetrics omegaconf

# =============================================================================
# CELL 2: Clone Repository and Setup Framework
# =============================================================================

# Clone your repository (replace with your GitHub repository URL)
!git clone https://github.com/yourusername/ML_Framework.git /kaggle/working/ML_Framework

# Add framework to Python path
sys.path.append('/kaggle/working/ML_Framework')

# Change to framework directory
os.chdir('/kaggle/working/ML_Framework')

# Import framework components
from utils.kaggle_utils import setup_kaggle_environment, print_kaggle_info, is_kaggle_environment
from scripts.train import main
from utils.config import load_config

print("Framework setup complete!")

# =============================================================================
# CELL 3: Setup Kaggle Environment
# =============================================================================

# Setup Kaggle environment
if is_kaggle_environment():
    working_dir = setup_kaggle_environment()
    print_kaggle_info()
    print(f"Working directory: {working_dir}")

# =============================================================================
# CELL 4: Create Kaggle Configuration
# =============================================================================

# Create Kaggle-optimized configuration
kaggle_config = """
experiment:
  name: segmentation_kaggle_gpu
  seed: 42

data:
  dataset_type: segmentation
  # Replace 'your-dataset' with your actual dataset name
  data_dir: /kaggle/input/your-dataset/dataset/train/images
  mask_dir: /kaggle/input/your-dataset/dataset/train/masks
  val_data_dir: /kaggle/input/your-dataset/dataset/val/images
  val_mask_dir: /kaggle/input/your-dataset/dataset/val/masks
  num_classes: 2
  class_names: ['background', 'markers']
  mask_format: 'rgb'
  train_split: 1.0
  val_split: 1.0
  test_split: 0.0

dataloader:
  batch_size: 16
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  drop_last: true

model:
  backbone: smp_deeplabv3plus_resnet50
  pretrained: true
  freeze_backbone: false
  in_channels: 3
  encoder_depth: 5
  decoder_channels: 256
  upsampling: 4

loss:
  name: combined
  ce_weight: 1.0
  dice_weight: 1.0
  focal_weight: 0.0
  lovasz_weight: 0.0
  ignore_index: 255

optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 1e-4
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  name: cosine_warmup
  warmup_epochs: 5
  total_epochs: 30
  min_lr: 1e-6

training:
  epochs: 30
  amp: true
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1

metrics:
  task: segmentation
  num_classes: 2
  ignore_index: 255

callbacks:
  checkpoint:
    monitor: val_loss
    mode: min
    save_top_k: 2
    enabled: true
    save_last: true
    save_best: true
  
  early_stopping:
    monitor: val_loss
    mode: min
    patience: 8
    enabled: true
  
  sample_visualizer:
    num_samples: 4
    save_every_n_epochs: 10
    enabled: true
  
  confusion_matrix:
    save_every_n_epochs: 15
    enabled: true
  
  learning_rate:
    save_every_n_epochs: 10
    enabled: true

transforms:
  resize: 512
  horizontal_flip: true
  vertical_flip: false
  rotation: 15
  shift_scale_rotate: true
  elastic_transform: false
  grid_distortion: false
  color_jitter: true
  gaussian_noise: false
  gaussian_blur: false
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  use_albumentations: true
"""

# Save configuration
with open('/kaggle/working/kaggle_config.yaml', 'w') as f:
    f.write(kaggle_config)

print("Configuration saved!")

# =============================================================================
# CELL 5: Run Training
# =============================================================================

import subprocess

# Run training
print("Starting training...")
result = subprocess.run([
    'python', 'scripts/train.py',
    '--config', '/kaggle/working/kaggle_config.yaml',
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

# Display some results
import matplotlib.pyplot as plt
import json

# Try to load and display training history
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

print("Notebook execution complete!")
