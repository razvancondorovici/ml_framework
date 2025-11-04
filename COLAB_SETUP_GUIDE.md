# Google Colab Setup Guide for ML Framework

This guide will help you run your ML Framework on Google Colab with GPU acceleration.

## Prerequisites

1. **Google Account**: Free account with access to Google Colab
2. **GitHub Repository**: Your framework should be available on GitHub (or use the repository URL directly)
3. **Dataset**: Your dataset ready to upload or stored on Google Drive

## Step 1: Create a New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "New Notebook"
3. **Enable GPU** (IMPORTANT):
   - Click "Runtime" → "Change runtime type"
   - Under "Hardware accelerator", select "GPU" (T4 is free tier)
   - Click "Save"

## Step 2: Prepare Your Dataset

You have two options for your dataset:

### Option A: Google Drive (Recommended for large datasets)

1. Upload your dataset to Google Drive
2. Organize it like this:
   ```
   MyDrive/
   └── datasets/
       └── your_dataset/
           ├── train/
           │   ├── class1/
           │   └── class2/
           └── val/
               ├── class1/
               └── class2/
   ```
3. The notebook will mount Google Drive automatically

### Option B: Direct Upload (For smaller datasets)

1. Zip your dataset folder
2. Upload it directly in the notebook using the upload cell
3. The dataset will be extracted to `/content/datasets`

## Step 3: Setup Notebook

Copy the cells from `colab_notebook.py` into your Colab notebook. The template is organized into cells:

### Cell 1: Enable GPU and Check Environment
- Verifies GPU is available
- Shows GPU information

### Cell 2: Install Dependencies
- Installs all required packages
- Verifies installation

### Cell 3: Clone Repository
- Clones your ML framework repository
- Sets up Python path

**⚠️ IMPORTANT**: Update the repository URL in this cell:
```python
repo_url = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"  # Change this!
```

### Cell 4: Mount Google Drive (Optional)
- Only needed if using Option A (Google Drive)
- Uncomment and run if your dataset is on Drive

### Cell 5: Setup Colab Environment
- Creates necessary directories
- Shows environment information

### Cell 6: Upload Dataset (Alternative)
- Only needed if using Option B (Direct Upload)
- Uncomment and run if uploading directly

### Cell 7: Create/Update Configuration
- Updates config file paths for Colab
- You may need to manually adjust paths

### Cell 8: Run Training
- Starts the training process
- Automatically converts Windows paths to Colab paths

### Cell 9: Display Results
- Shows training curves
- Lists checkpoints
- Provides download instructions

## Step 4: Configure Your Dataset Paths

Update the paths in your configuration file to match your Colab setup:

### If using Google Drive:
```yaml
data:
  train_data_dir: /content/drive/MyDrive/datasets/your_dataset/train
  val_data_dir: /content/drive/MyDrive/datasets/your_dataset/val
```

### If using uploaded data:
```yaml
data:
  train_data_dir: /content/datasets/your_dataset/train
  val_data_dir: /content/datasets/your_dataset/val
```

### If you have multiple directories (like in your config):
```yaml
data:
  train_data_dir:
    - /content/drive/MyDrive/datasets/herlev_grouped/train
    - /content/drive/MyDrive/datasets/sipakmed_grouped_per_subject/train
  val_data_dir:
    - /content/drive/MyDrive/datasets/herlev_grouped/val
    - /content/drive/MyDrive/datasets/sipakmed_grouped_per_subject/val
```

## Step 5: Run Training

1. **Run all cells** in order (Shift+Enter for each cell)
2. **Monitor progress** in the output
3. **Check results** in the final cell

## Key Optimizations for Colab

### Configuration Changes from CPU to GPU:

| Setting | CPU Version | Colab GPU Version | Reason |
|---------|-------------|-------------------|---------|
| `batch_size` | 8-16 | 32 | GPU can handle larger batches |
| `num_workers` | 4-8 | 2 | Limited CPU cores on Colab |
| `pin_memory` | false | true | Enable for GPU efficiency |
| `amp` | false | true | Mixed precision for speed |
| `epochs` | 100 | 50-100 | Session time considerations |
| `patience` | 10 | 10-15 | Standard convergence |

### Path Handling:

The framework automatically converts Windows paths to Colab paths:
- `c:\Facultate\...` → `/content/drive/MyDrive/datasets/...` (if Drive mounted)
- `c:\Facultate\...` → `/content/datasets/...` (if uploaded)
- Backslashes are converted to forward slashes
- Paths are validated for existence

## Troubleshooting

### Common Issues:

1. **GPU not detected**:
   ```
   CUDA available: False
   ```
   **Solution**: 
   - Go to Runtime → Change runtime type → Select GPU
   - Restart the runtime (Runtime → Restart runtime)
   - Run Cell 1 again to verify

2. **Dataset not found**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   **Solution**:
   - Check dataset path in config file
   - Verify Google Drive is mounted (if using Drive)
   - Check dataset is uploaded correctly
   - Use absolute paths in config

3. **Out of memory (OOM)**:
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**:
   - Reduce `batch_size` in config (try 16 or 8)
   - Enable gradient accumulation
   - Reduce image size in transforms
   - Use smaller model (e.g., efficientnet_b0 instead of b7)

4. **Package conflicts**:
   ```
   ImportError: cannot import name 'X' from 'Y'
   ```
   **Solution**:
   - Restart runtime (Runtime → Restart runtime)
   - Re-run Cell 2 (Install Dependencies)
   - Check for version conflicts in requirements.txt

5. **Training timeout**:
   - Colab free tier has 12-hour session limit
   - Save checkpoints frequently
   - Use early stopping
   - Consider reducing epochs

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

After training, your results will be saved in `/content/ml_framework/runs/`:

- **Model checkpoints**: `*.pt` files in `checkpoints/` folder
- **Training plots**: `*.png` files in `plots/` folder
- **Metrics**: `*.json`, `*.csv` files in experiment folder
- **Logs**: `*.log` files

### Downloading Results:

To download checkpoints or results:

```python
from google.colab import files

# Download best checkpoint
files.download('/content/ml_framework/runs/experiment_name/timestamp/checkpoints/best.pt')

# Download training history
files.download('/content/ml_framework/runs/experiment_name/timestamp/training_history.json')
```

Or use File Browser in Colab sidebar to navigate and download files.

## Best Practices

1. **Save frequently**: Colab sessions can disconnect
   - Enable automatic checkpoint saving
   - Download important checkpoints periodically

2. **Monitor GPU usage**: 
   - Use `nvidia-smi` in a code cell to check GPU utilization
   - Watch for memory leaks

3. **Version control**: 
   - Keep your GitHub repository updated
   - Use git to track changes in Colab if needed

4. **Resource optimization**:
   - Use mixed precision (AMP) - already enabled
   - Adjust batch size based on GPU memory
   - Use gradient accumulation for effective larger batches

5. **Session management**:
   - Keep browser tab open to prevent disconnection
   - Use Colab Pro for longer sessions if needed
   - Save intermediate results

## Colab Limits

- **GPU Time**: ~12 hours per session (free tier)
- **Session Timeout**: ~90 minutes of inactivity
- **Storage**: 80GB temporary storage (deleted on disconnect)
- **Memory**: Varies by GPU type (T4: 16GB, V100: 16GB, A100: 40GB)

## Example Complete Workflow

```python
# 1. Check GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")

# 2. Install packages
!pip install timm segmentation-models-pytorch albumentations torchmetrics omegaconf

# 3. Clone repo
!git clone https://github.com/yourusername/ml_framework.git /content/ml_framework
import sys
sys.path.append('/content/ml_framework')

# 4. Mount Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')

# 5. Setup
from utils.colab_utils import setup_colab_environment, convert_paths_to_colab
setup_colab_environment()

# 6. Run training
import subprocess
subprocess.run([
    'python', '/content/ml_framework/scripts/train.py',
    '--config', '/content/ml_framework/configs/classification_cells_herlev.yaml',
    '--device', 'cuda'
])

# 7. Download results
from google.colab import files
files.download('/content/ml_framework/runs/.../checkpoints/best.pt')
```

## Quick Start Checklist

- [ ] Created Colab notebook
- [ ] Enabled GPU in runtime settings
- [ ] Cloned repository (updated URL)
- [ ] Mounted Google Drive OR uploaded dataset
- [ ] Updated config file paths
- [ ] Verified dataset paths exist
- [ ] Ran all cells successfully
- [ ] Training started without errors
- [ ] Downloaded checkpoints after training

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Colab's documentation
3. Verify all paths and permissions
4. Check GPU availability in Cell 1
5. Restart runtime if issues persist

Happy training on Colab! 🚀

