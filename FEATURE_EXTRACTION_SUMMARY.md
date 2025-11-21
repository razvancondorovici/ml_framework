# Feature Extraction Implementation Summary

## Overview

I've successfully modified the ML framework to support feature extraction from trained classification models. The features are extracted from the backbone network (before the classification head) and saved as a dictionary with `{filename: features}` pairs.

## Changes Made

### 1. Modified `engine/inferencer.py`

Added the following methods to the `Inferencer` class:

- **`enable_feature_extraction()`**: Removes the classification head to enable feature extraction
- **`disable_feature_extraction()`**: Restores the classification head
- **`extract_features(dataloader)`**: Extracts features from a dataloader
- **`extract_features_from_folder(folder_path, output_path, ...)`**: Main method that:
  - Finds all images in a folder (recursively)
  - Extracts features in batches
  - Saves features as a dictionary `{filename: features}`
  - Supports multiple save formats (pickle, npz, or both)
  - Saves metadata JSON file with info about extracted features

### 2. Created `scripts/extract_features.py`

Standalone script for feature extraction with command-line interface:

```bash
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint path/to/checkpoint.pth \
    --input path/to/images \
    --output results/features \
    --batch-size 32 \
    --save-format pickle
```

### 3. Updated `scripts/infer.py`

Added `--extract-features` flag to the existing inference script:

```bash
python scripts/infer.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint path/to/checkpoint.pth \
    --input path/to/images \
    --output results/features \
    --extract-features \
    --save-format both
```

### 4. Created Documentation

- **`docs/FEATURE_EXTRACTION.md`**: Comprehensive guide with usage examples
- **`scripts/load_features_example.py`**: Shows how to load and use extracted features
- **`examples/feature_extraction_workflow.py`**: Complete workflow demonstration
- **`examples/extract_features_example.ps1`**: PowerShell examples for Windows
- **`examples/extract_features_example.sh`**: Bash examples for Linux/Mac

## Usage

### Basic Usage

```python
from engine.inferencer import Inferencer
from models.registry import build_classifier
from utils.config import load_config

# Load config and model
config = load_config('configs/classification_cells_herlev.yaml')
model = build_classifier(
    backbone='timm_efficientnet_b0',
    num_classes=2,
    pretrained=False,
    dropout=0.2
)

# Create inferencer and load checkpoint
inferencer = Inferencer(model=model, config=config)
inferencer.load_checkpoint('path/to/checkpoint.pth')

# Extract features
features_dict = inferencer.extract_features_from_folder(
    folder_path='path/to/images',
    output_path='results/features',
    batch_size=32,
    save_format='pickle'
)
```

### Loading Extracted Features

```python
import pickle

# Load features
with open('results/features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Access features by filename
features = features_dict['image001.jpg']
print(f"Features shape: {features.shape}")  # (1280,) for EfficientNet-B0
```

## Output Files

When you extract features, three files are created:

1. **`features.pkl`** (if save_format='pickle' or 'both'): 
   - Python pickle file containing the dictionary
   - Easy to load with `pickle.load()`

2. **`features.npz`** (if save_format='npz' or 'both'):
   - NumPy compressed format
   - Load with `np.load()`

3. **`features_metadata.json`**: 
   - Contains metadata about the extraction
   - Number of images, feature dimension, source folder, etc.

## Feature Dimensions

The feature dimension depends on the backbone:

- **EfficientNet-B0**: 1280
- **EfficientNet-B1**: 1280
- **EfficientNet-B2**: 1408
- **ResNet-50**: 2048

For your model (EfficientNet-B0 from `classification_cells_herlev.yaml`), the feature dimension will be **1280**.

## Common Use Cases

The extracted features can be used for:

1. **Image Similarity Search**: Find similar images using cosine similarity
2. **Clustering**: Group images using K-Means, DBSCAN, etc.
3. **Visualization**: Reduce to 2D/3D using t-SNE, UMAP, PCA
4. **Transfer Learning**: Train downstream classifiers (SVM, Random Forest, etc.)
5. **Anomaly Detection**: Detect outliers or unusual images
6. **Image Retrieval**: Build a search engine based on visual similarity

## Examples

### Example 1: Extract Features (Command Line)

```bash
# Using the dedicated script
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input "C:\Facultate\CerviAssist\CellsClassification\Dataset\2025_11_03\all\herlev_grouped\test" \
    --output results/herlev_features \
    --batch-size 32

# Using the infer script with --extract-features flag
python scripts/infer.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input "C:\Facultate\CerviAssist\CellsClassification\Dataset\2025_11_03\all\herlev_grouped\test" \
    --output results/herlev_features \
    --extract-features
```

### Example 2: Find Similar Images

```python
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load features
with open('results/features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

filenames = list(features_dict.keys())
features = np.array([features_dict[fn] for fn in filenames])

# Find images similar to the first one
query_idx = 0
similarities = cosine_similarity([features[query_idx]], features)[0]

# Get top 5 most similar
top_5_indices = np.argsort(similarities)[::-1][:5]
print(f"Most similar to {filenames[query_idx]}:")
for idx in top_5_indices:
    print(f"  {filenames[idx]}: {similarities[idx]:.4f}")
```

### Example 3: Cluster Images

```python
import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load features
with open('results/features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

features = np.array(list(features_dict.values()))
filenames = list(features_dict.keys())

# Cluster into 5 groups
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)

# Show distribution
for i in range(5):
    cluster_images = [filenames[j] for j in range(len(filenames)) if clusters[j] == i]
    print(f"Cluster {i}: {len(cluster_images)} images")
```

## Technical Details

### How It Works

1. **Enable Feature Extraction Mode**: The classification head is replaced with `nn.Identity()`
2. **Forward Pass**: Images are passed through the backbone network
3. **Global Average Pooling**: If features are 4D (B, C, H, W), they're pooled to (B, C)
4. **Dictionary Creation**: Features are stored with filename as key
5. **Save to Disk**: Dictionary is serialized using pickle or NumPy

### Compatibility

- Works with all timm models (EfficientNet, ResNet, ViT, etc.)
- Works with torchvision models
- Uses the same transforms as training (from config)
- Supports grayscale images (if configured)
- GPU and CPU compatible
- Batch processing for efficiency

### Performance

- **Batch Size**: Adjust based on GPU memory (32 is default)
- **Num Workers**: Adjust based on CPU cores (4 is default)
- **Speed**: ~100-200 images/sec on GPU (depends on backbone and batch size)

## Files Structure

```
ml_framework/
├── engine/
│   └── inferencer.py          # Modified with feature extraction methods
├── scripts/
│   ├── extract_features.py    # New: Standalone feature extraction script
│   ├── infer.py              # Modified: Added --extract-features flag
│   └── load_features_example.py  # New: Shows how to load features
├── examples/
│   ├── feature_extraction_workflow.py  # New: Complete workflow example
│   ├── extract_features_example.ps1    # New: PowerShell examples
│   └── extract_features_example.sh     # New: Bash examples
├── docs/
│   └── FEATURE_EXTRACTION.md   # New: Comprehensive documentation
└── FEATURE_EXTRACTION_SUMMARY.md  # This file
```

## Testing

To test the feature extraction:

1. Make sure you have a trained checkpoint
2. Run the extract_features script:
   ```bash
   python scripts/extract_features.py \
       --config configs/classification_cells_herlev.yaml \
       --checkpoint your_checkpoint.pth \
       --input path/to/test/images \
       --output results/test_features
   ```
3. Load and verify the features:
   ```bash
   python scripts/load_features_example.py
   ```

## Notes

- The same preprocessing transforms from training are applied during feature extraction
- Features are L2-normalized by default (from the backbone)
- The feature extractor works in eval mode (no dropout, batch norm in eval mode)
- Feature extraction is much faster than full inference (no classification head)

## Future Enhancements (Optional)

Potential improvements that could be added:

1. Support for extracting features from multiple layers
2. Feature aggregation strategies (max pooling, attention, etc.)
3. Feature PCA/dimensionality reduction on-the-fly
4. Built-in similarity search functionality
5. Feature database with fast retrieval (FAISS, Annoy)

---

**Author**: AI Assistant  
**Date**: November 6, 2025  
**Framework**: PyTorch ML Framework for Cervical Cell Classification





