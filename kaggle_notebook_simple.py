# Simplified Kaggle Notebook Template - Avoids torchmetrics conflicts
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

# Install required packages with specific versions to avoid conflicts
!pip install --upgrade pip
!pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
!pip install timm==0.9.12 segmentation-models-pytorch==0.3.3 albumentations==1.3.1
!pip install omegaconf==2.3.0 PyYAML==6.0.1
# Skip torchmetrics to avoid transformers conflict
# !pip install torchmetrics==1.2.0

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
# CELL 4: Create Kaggle Configuration (No torchmetrics dependency)
# =============================================================================

# Create Kaggle-optimized configuration that doesn't rely on torchmetrics
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

# Simple metrics configuration (no torchmetrics dependency)
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
# CELL 5: Import and Run Training (with fallback metrics)
# =============================================================================

# Import training components with fallback handling
try:
    from scripts.train import main
    print("Using full framework with fallback metrics")
except Exception as e:
    print(f"Error importing full framework: {e}")
    print("Using simplified training approach...")

# Try to run training
import subprocess

print("Starting training...")
try:
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
        
    if result.returncode == 0:
        print("✓ Training completed successfully!")
    else:
        print("✗ Training failed!")

except Exception as e:
    print(f"Training failed with exception: {e}")
    print("This might be due to package conflicts. Try the manual training approach below.")

# =============================================================================
# CELL 6: Alternative Manual Training (if framework fails)
# =============================================================================

# If the framework fails, we can do a simple manual training
print("\\n" + "="*50)
print("ALTERNATIVE: Manual Training Approach")
print("="*50)

try:
    # Import basic components
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from PIL import Image
    import numpy as np
    import cv2
    from pathlib import Path
    import json
    
    print("✓ Basic imports successful")
    
    # Create a simple dataset class
    class SimpleSegmentationDataset:
        def __init__(self, image_dir, mask_dir, transform=None):
            self.image_dir = Path(image_dir)
            self.mask_dir = Path(mask_dir)
            self.transform = transform
            
            # Get image files
            self.image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))
            print(f"Found {len(self.image_files)} images")
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            mask_path = self.mask_dir / (img_path.stem + '.png')
            
            # Load image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            return image, mask
    
    # Create transforms
    transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = SimpleSegmentationDataset(
        '/kaggle/input/your-dataset/dataset/train/images',
        '/kaggle/input/your-dataset/dataset/train/masks',
        transform=transform
    )
    
    val_dataset = SimpleSegmentationDataset(
        '/kaggle/input/your-dataset/dataset/val/images',
        '/kaggle/input/your-dataset/dataset/val/masks',
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✓ Model created and moved to {device}")
    print(f"✓ Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Simple training loop
    num_epochs = 10  # Reduced for demo
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save model
        torch.save(model.state_dict(), f'/kaggle/working/model_epoch_{epoch}.pt')
    
    print("✓ Manual training completed!")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    with open('/kaggle/working/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
except Exception as e:
    print(f"Manual training also failed: {e}")
    print("Please check your dataset paths and structure.")

# =============================================================================
# CELL 7: Save and Display Results
# =============================================================================

import shutil
from pathlib import Path

# Copy results to a permanent location
output_dir = Path('/kaggle/working/outputs')
results_dir = Path('/kaggle/working/final_results')

if output_dir.exists():
    shutil.copytree(output_dir, results_dir, dirs_exist_ok=True)

# List all important files
print("\\nTraining completed! Available outputs:")
important_extensions = ['.pt', '.png', '.json', '.csv', '.log']

for item in Path('/kaggle/working').rglob('*'):
    if item.is_file() and item.suffix in important_extensions:
        print(f"  {item}")

# Display some results
import matplotlib.pyplot as plt

# Try to load and display training history
history_files = list(Path('/kaggle/working').rglob('training_history.json'))
if history_files:
    with open(history_files[0], 'r') as f:
        history = json.load(f)
    
    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss')
    ax.set_title('Training Progress')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

print("\\nNotebook execution complete!")
print("\\nTo use this notebook:")
print("1. Replace 'your-dataset' with your actual Kaggle dataset name")
print("2. Replace the GitHub URL with your repository URL")
print("3. Run all cells")
print("4. Check the final_results folder for your trained models")
