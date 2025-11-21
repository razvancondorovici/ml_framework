#!/usr/bin/env python3
"""Example script showing how to load extracted features."""

import pickle
import numpy as np
from pathlib import Path
import json


def load_features_from_pickle(pickle_path):
    """Load features from pickle file.
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        Dictionary with {filename: features} pairs
    """
    with open(pickle_path, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict


def load_features_from_npz(npz_path):
    """Load features from npz file.
    
    Args:
        npz_path: Path to npz file
        
    Returns:
        Dictionary with {filename: features} pairs
    """
    data = np.load(npz_path)
    filenames = data['filenames']
    features = data['features']
    
    # Convert to dictionary
    features_dict = {}
    for i, filename in enumerate(filenames):
        features_dict[str(filename)] = features[i]
    
    return features_dict


def main():
    """Example usage."""
    # Example 1: Load from pickle
    print("Example 1: Loading from pickle file")
    print("=" * 60)
    
    pickle_path = "path/to/features.pkl"  # Replace with your path
    
    if Path(pickle_path).exists():
        features_dict = load_features_from_pickle(pickle_path)
        
        print(f"Loaded features for {len(features_dict)} images")
        
        # Access features by filename
        for filename, features in list(features_dict.items())[:3]:
            print(f"{filename}: shape={features.shape}")
        
        # Get features for a specific image
        if len(features_dict) > 0:
            filename = list(features_dict.keys())[0]
            features = features_dict[filename]
            print(f"\nFeatures for {filename}:")
            print(f"  Shape: {features.shape}")
            print(f"  Mean: {features.mean():.4f}")
            print(f"  Std: {features.std():.4f}")
            print(f"  Min: {features.min():.4f}")
            print(f"  Max: {features.max():.4f}")
    else:
        print(f"File not found: {pickle_path}")
        print("Please replace 'path/to/features.pkl' with actual path")
    
    print("\n" + "=" * 60)
    
    # Example 2: Load from npz
    print("\nExample 2: Loading from npz file")
    print("=" * 60)
    
    npz_path = "path/to/features.npz"  # Replace with your path
    
    if Path(npz_path).exists():
        features_dict = load_features_from_npz(npz_path)
        
        print(f"Loaded features for {len(features_dict)} images")
        
        # Access features
        for filename, features in list(features_dict.items())[:3]:
            print(f"{filename}: shape={features.shape}")
    else:
        print(f"File not found: {npz_path}")
        print("Please replace 'path/to/features.npz' with actual path")
    
    print("\n" + "=" * 60)
    
    # Example 3: Load metadata
    print("\nExample 3: Loading metadata")
    print("=" * 60)
    
    metadata_path = "path/to/features_metadata.json"  # Replace with your path
    
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Number of images: {metadata['num_images']}")
        print(f"Feature dimension: {metadata['feature_dim']}")
        print(f"Source folder: {metadata['source_folder']}")
        print(f"First 3 filenames: {metadata['filenames'][:3]}")
    else:
        print(f"File not found: {metadata_path}")
        print("Please replace 'path/to/features_metadata.json' with actual path")
    
    print("\n" + "=" * 60)
    
    # Example 4: Convert features to array for machine learning
    print("\nExample 4: Convert to arrays for ML")
    print("=" * 60)
    
    if Path(pickle_path).exists():
        features_dict = load_features_from_pickle(pickle_path)
        
        # Get all filenames and features as arrays
        filenames = list(features_dict.keys())
        features_array = np.array([features_dict[fn] for fn in filenames])
        
        print(f"Features array shape: {features_array.shape}")
        print(f"Can now use this array for clustering, classification, etc.")
        
        # Example: Compute cosine similarity between first two images
        if len(features_array) >= 2:
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([features_array[0]], [features_array[1]])[0][0]
            print(f"\nCosine similarity between {filenames[0]} and {filenames[1]}: {sim:.4f}")


if __name__ == '__main__':
    main()





