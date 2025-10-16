# Kaggle Setup Guide for ML Framework

This guide will help you run your ML Framework on Kaggle with GPU acceleration.

## Prerequisites

1. **GitHub Repository**: Your framework should be available on GitHub
2. **Kaggle Account**: Free account at [kaggle.com](https://kaggle.com)
3. **Dataset**: Your dataset uploaded to Kaggle Datasets

## Step 1: Upload Your Dataset to Kaggle

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload your data with this structure:
   ```
   your-dataset/
   ├── dataset/
   │   ├── train/
   │   │   ├── images/
   │   │   └── masks/
   │   └── val/
   │       ├── images/
   │       └── masks/
   ```
4. Make the dataset public or private (your choice)
5. Note your dataset name (e.g., `melanodet-markerssegmentation`)

## Step 2: Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU:
   - Click "Settings" (gear icon)
   - Under "Accelerator", select "GPU"
   - Click "Save"

## Step 3: Setup Notebook

### Option A: Use the Template (Recommended)

Copy the code from `kaggle_notebook_template.py` into your notebook cells:

1. **Cell 1**: Environment setup and dependencies
2. **Cell 2**: Clone repository and setup framework  
3. **Cell 3**: Setup Kaggle environment
4. **Cell 4**: Create Kaggle configuration
5. **Cell 5**: Run training
6. **Cell 6**: Save and display results

### Option B: Manual Setup

#### Cell 1: Basic Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
!pip install timm segmentation-models-pytorch albumentations torchmetrics omegaconf
```

#### Cell 2: Clone Framework
```python
!git clone https://github.com/yourusername/ML_Framework.git /kaggle/working/ML_Framework
import sys
sys.path.append('/kaggle/working/ML_Framework')
import os
os.chdir('/kaggle/working/ML_Framework')
```

#### Cell 3: Run Training
```python
import subprocess
result = subprocess.run([
    'python', 'scripts/train.py',
    '--config', 'configs/segmentation_kaggle_gpu.yaml',
    '--device', 'cuda'
])
```

## Step 4: Configure Your Dataset

Before running, update the dataset paths in your configuration:

1. **In the notebook**: Replace `your-dataset` with your actual dataset name
2. **Or modify the config file**: Edit `configs/segmentation_kaggle_gpu.yaml`

Example:
```yaml
data:
  data_dir: /kaggle/input/melanodet-markerssegmentation/dataset/train/images
  mask_dir: /kaggle/input/melanodet-markerssegmentation/dataset/train/masks
  val_data_dir: /kaggle/input/melanodet-markerssegmentation/dataset/val/images
  val_mask_dir: /kaggle/input/melanodet-markerssegmentation/dataset/val/masks
```

## Step 5: Run Training

1. **Save your notebook** (Ctrl+S)
2. **Run all cells** (Shift+Enter for each cell)
3. **Monitor progress** in the output
4. **Check results** in the final cell

## Key Optimizations for Kaggle

### Configuration Changes from CPU to GPU:

| Setting | CPU Version | Kaggle GPU Version | Reason |
|---------|-------------|-------------------|---------|
| `batch_size` | 2 | 16 | GPU can handle larger batches |
| `num_workers` | 0 | 2 | Limited CPU cores on Kaggle |
| `pin_memory` | false | true | Enable for GPU efficiency |
| `amp` | false | true | Mixed precision for speed |
| `epochs` | 50 | 30 | Time limit considerations |
| `patience` | 10 | 8 | Faster convergence |
| `save_top_k` | 3 | 2 | Save disk space |

### Path Handling:

The framework automatically converts Windows paths to Kaggle paths:
- `c:\Facultate\...` → `/kaggle/input/your-dataset/...`
- Backslashes are converted to forward slashes
- Paths are validated for existence

## Troubleshooting

### Common Issues:

1. **Dataset not found**:
   - Check dataset name in configuration
   - Verify dataset is public or you have access
   - Check folder structure matches expected format

2. **GPU not detected**:
   - Ensure GPU is enabled in notebook settings
   - Restart notebook after enabling GPU
   - Check CUDA availability with `torch.cuda.is_available()`

3. **Out of memory**:
   - Reduce batch size in configuration
   - Enable gradient accumulation
   - Use smaller input image size

4. **Training timeout**:
   - Reduce number of epochs
   - Enable early stopping
   - Save checkpoints more frequently

### Memory Management:

```python
# Add to your notebook if you encounter memory issues
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Call periodically during training
cleanup_memory()
```

## Results and Outputs

After training, your results will be saved in `/kaggle/working/`:

- **Model checkpoints**: `*.pt` files
- **Training plots**: `*.png` files  
- **Metrics**: `*.json`, `*.csv` files
- **Logs**: `*.log` files

## Best Practices

1. **Version Control**: Keep your GitHub repository updated
2. **Dataset Organization**: Use consistent folder structures
3. **Resource Monitoring**: Watch GPU memory usage
4. **Regular Saves**: Save intermediate results
5. **Documentation**: Document your experiments

## Kaggle Limits

- **GPU Time**: 30 hours per week (free tier)
- **Session Time**: 9 hours maximum per session
- **Storage**: 20GB working directory
- **Memory**: Varies by GPU type (T4, P100, etc.)

## Example Complete Workflow

```python
# 1. Setup
import torch
!pip install timm segmentation-models-pytorch albumentations torchmetrics omegaconf
!git clone https://github.com/yourusername/ML_Framework.git /kaggle/working/ML_Framework

# 2. Configure
import sys, os
sys.path.append('/kaggle/working/ML_Framework')
os.chdir('/kaggle/working/ML_Framework')

# 3. Run
import subprocess
subprocess.run([
    'python', 'scripts/train.py',
    '--config', 'configs/segmentation_kaggle_gpu.yaml',
    '--device', 'cuda'
])

# 4. Results
from pathlib import Path
for file in Path('/kaggle/working').rglob('*.pt'):
    print(f"Model: {file}")
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Kaggle's documentation
3. Check your dataset structure and permissions
4. Verify all dependencies are installed correctly

Happy training on Kaggle! 🚀
