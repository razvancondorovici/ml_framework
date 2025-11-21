#!/bin/bash
# Example: Extract features from images using a trained model

# Example 1: Extract features using the herlev configuration
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input "C:\Facultate\CerviAssist\CellsClassification\Dataset\2025_11_03\all\herlev_grouped\test" \
    --output results/herlev_test_features \
    --batch-size 32 \
    --num-workers 4 \
    --save-format both

# Example 2: Extract features with smaller batch size (if memory is limited)
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input "C:\Facultate\CerviAssist\CellsClassification\Dataset\2025_11_03\all\sante_split\val" \
    --output results/sante_val_features \
    --batch-size 16 \
    --save-format pickle

# Example 3: Extract features on CPU
python scripts/extract_features.py \
    --config configs/classification_cells_herlev.yaml \
    --checkpoint runs/test_grayscale/best_model.pth \
    --input path/to/images \
    --output results/features_cpu \
    --device cpu \
    --batch-size 8 \
    --save-format pickle





