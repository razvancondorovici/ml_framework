# Feature Extraction Guide

This guide explains how to extract features from a trained classification model instead of getting classification predictions.

## Overview

Feature extraction allows you to use your trained model as a feature extractor, removing the final classification layer and extracting the learned representations from the backbone network. This is useful for:

- **Transfer Learning**: Use extracted features as input to another model
- **Similarity Search**: Find similar images based on feature similarity
- **Clustering**: Group images based on their features
- **Visualization**: Reduce features to 2D/3D for visualization (e.g., t-SNE, UMAP)

## Quick Start

### 1. Extract Features from a Folder of Images

```bash
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint path/to/your/checkpoint.pth \
    --input path/to/images/folder \
    --output results/features \
    --batch-size 32 \
    --save-format pickle
```

**Arguments:**
- `--config`: Path to your training configuration file
- `--checkpoint`: Path to your trained model checkpoint
- `--input`: Path to folder containing images (searches recursively)
- `--output`: Path where features will be saved (without extension)
- `--batch-size`: Batch size for processing (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--device`: Device to use ('cuda' or 'cpu', auto-detected if not specified)
- `--save-format`: Format to save features ('pickle', 'npz', or 'both')

### 2. Load Extracted Features in Python

#### Using Pickle (Recommended)

```python
import pickle

# Load features
with open('results/features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Access features by filename
filename = "image001.jpg"
features = features_dict[filename]
print(f"Features shape: {features.shape}")  # e.g., (1280,) for EfficientNet-B0
```

#### Using NumPy NPZ

```python
import numpy as np

# Load features
data = np.load('results/features.npz')
filenames = data['filenames']
features = data['features']

# Convert to dictionary
features_dict = {str(fn): features[i] for i, fn in enumerate(filenames)}
```

#### Load Metadata

```python
import json

with open('results/features_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Number of images: {metadata['num_images']}")
print(f"Feature dimension: {metadata['feature_dim']}")
print(f"Source folder: {metadata['source_folder']}")
```

## Programmatic Usage

You can also use the feature extraction functionality directly in your Python code:

```python
from pathlib import Path
from utils.config import load_config
from models.registry import build_classifier
from engine.inferencer import Inferencer

# Load config and create model
config = load_config('configs/classification_cells_herlev.yaml')
model = build_classifier(
    backbone='timm_efficientnet_b0',
    num_classes=2,
    pretrained=False,
    dropout=0.2
)

# Create inferencer and load checkpoint
inferencer = Inferencer(model=model, config=config, device='cuda')
inferencer.load_checkpoint('path/to/checkpoint.pth')

# Extract features from folder
features_dict = inferencer.extract_features_from_folder(
    folder_path='path/to/images',
    output_path='results/features',
    batch_size=32,
    save_format='pickle'
)

# Use features
for filename, features in features_dict.items():
    print(f"{filename}: {features.shape}")
```

## Feature Dimensions

The feature dimension depends on the backbone architecture:

| Backbone | Feature Dimension |
|----------|------------------|
| EfficientNet-B0 | 1280 |
| EfficientNet-B1 | 1280 |
| EfficientNet-B2 | 1408 |
| EfficientNet-B3 | 1536 |
| ResNet-50 | 2048 |
| ResNet-101 | 2048 |
| ViT-Base | 768 |

## Common Use Cases

### 1. Image Similarity Search

```python
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load features
with open('features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

filenames = list(features_dict.keys())
features = np.array([features_dict[fn] for fn in filenames])

# Find similar images to query image
query_idx = 0
query_features = features[query_idx].reshape(1, -1)

# Compute similarities
similarities = cosine_similarity(query_features, features)[0]

# Get top 5 most similar images
top_indices = np.argsort(similarities)[::-1][:5]
print(f"Most similar to {filenames[query_idx]}:")
for idx in top_indices:
    print(f"  {filenames[idx]}: {similarities[idx]:.4f}")
```

### 2. Clustering

```python
import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load features
with open('features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

filenames = list(features_dict.keys())
features = np.array([features_dict[fn] for fn in filenames])

# Cluster images
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Group images by cluster
for cluster_id in range(n_clusters):
    cluster_images = [filenames[i] for i in range(len(filenames)) if clusters[i] == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_images)} images")
```

### 3. Visualization with t-SNE

```python
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load features
with open('features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

features = np.array(list(features_dict.values()))

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5)
plt.title('t-SNE visualization of image features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('tsne_visualization.png')
```

### 4. Train a Downstream Classifier

```python
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load features
with open('features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Assuming you have labels for your images
filenames = list(features_dict.keys())
features = np.array([features_dict[fn] for fn in filenames])
labels = ...  # Your labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train SVM classifier
clf = SVC(kernel='rbf', gamma='auto')
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Notes

- **Transforms**: The same transforms from your training config are applied during feature extraction
- **Grayscale**: If your config uses `grayscale: true`, the transform will convert images to grayscale but duplicate the channel 3 times for model input
- **Batch Processing**: Features are extracted in batches for efficiency
- **GPU Usage**: Automatically uses GPU if available
- **File Formats**: 
  - Pickle (.pkl): Easy to use in Python, preserves dictionary structure
  - NumPy (.npz): More portable, good for large datasets
  - Both: Saves in both formats for flexibility

## Troubleshooting

### Out of Memory Error
Reduce batch size: `--batch-size 16` or `--batch-size 8`

### Slow Processing
Increase number of workers: `--num-workers 8`

### Model Not Found
Make sure the checkpoint path is correct and contains the trained model weights.

### Feature Dimension Mismatch
Make sure you're using the same model architecture that was used during training.





