# Feature Extraction - Quick Start Guide

## Extract Features (One Command)

```bash
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input path/to/your/images \
    --output results/my_features
```

## Load Features in Python

```python
import pickle

# Load the dictionary
with open('results/my_features.pkl', 'rb') as f:
    features = pickle.load(f)

# Access features by filename
image_features = features['image001.jpg']
print(f"Shape: {image_features.shape}")  # (1280,) for EfficientNet-B0
```

## Find Similar Images

```python
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load features
with open('results/my_features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Convert to arrays
filenames = list(features_dict.keys())
features = np.array([features_dict[fn] for fn in filenames])

# Find similar images
query_idx = 0  # First image
similarities = cosine_similarity([features[query_idx]], features)[0]
top_5 = np.argsort(similarities)[::-1][:5]

print(f"Similar to {filenames[query_idx]}:")
for idx in top_5:
    print(f"  {filenames[idx]}: {similarities[idx]:.4f}")
```

## Cluster Images

```python
import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load features
with open('results/my_features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Cluster
features = np.array(list(features_dict.values()))
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features)

# Show results
filenames = list(features_dict.keys())
for i in range(3):
    imgs = [filenames[j] for j in range(len(filenames)) if clusters[j] == i]
    print(f"Cluster {i}: {len(imgs)} images")
```

## Command-Line Options

```bash
# Basic usage
python scripts/extract_features.py \
    --config <config_file> \
    --checkpoint <checkpoint_path> \
    --input <images_folder> \
    --output <output_path>

# With options
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input path/to/images \
    --output results/features \
    --batch-size 16 \              # Reduce if out of memory
    --num-workers 4 \               # Parallel data loading
    --device cuda \                 # or 'cpu'
    --save-format both              # 'pickle', 'npz', or 'both'
```

## Output Files

After running, you'll get:

- `my_features.pkl` - Dictionary with {filename: features}
- `my_features_metadata.json` - Info about extraction
- `my_features.npz` - NumPy format (if using --save-format both/npz)

## For Your Specific Use Case

Based on your config (`classification_cells_herlev.yaml`):

```bash
# Example with your test set
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input "C:\Facultate\CerviAssist\CellsClassification\Dataset\2025_11_03\all\herlev_grouped\test" \
    --output results/herlev_test_features \
    --batch-size 32 \
    --save-format pickle
```

Feature dimension: **1280** (EfficientNet-B0)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Use `--batch-size 8` or `--batch-size 4` |
| Slow processing | Increase `--num-workers 8` |
| No GPU | Use `--device cpu` |
| File not found | Check checkpoint and input paths |

## More Examples

See `docs/FEATURE_EXTRACTION.md` for detailed documentation and more examples.





