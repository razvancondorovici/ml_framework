#!/usr/bin/env python3
"""Complete workflow example for feature extraction and usage."""

import pickle
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def step1_extract_features():
    """Step 1: Extract features from trained model."""
    print("=" * 70)
    print("STEP 1: Extracting Features from Images")
    print("=" * 70)
    
    from utils.config import load_config
    from models.registry import build_classifier
    from engine.inferencer import Inferencer
    
    # Configuration
    config_path = 'configs/classification_cells_herlev.yaml'
    checkpoint_path = 'runs/test_grayscale/best_model.pth'  # Update this path
    input_folder = 'path/to/your/images'  # Update this path
    output_path = 'results/extracted_features'
    
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output: {output_path}")
    
    # Check if paths exist
    if not Path(checkpoint_path).exists():
        print(f"\nWarning: Checkpoint not found at {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual checkpoint path")
        return None
    
    if not Path(input_folder).exists():
        print(f"\nWarning: Input folder not found at {input_folder}")
        print("Please update the input_folder variable with your actual image folder")
        return None
    
    # Load config
    config = load_config(config_path)
    
    # Create model
    print("\nCreating model...")
    model = build_classifier(
        backbone=config['model']['backbone'],
        num_classes=config['data']['num_classes'],
        pretrained=False,
        dropout=config['model'].get('dropout', 0.2)
    )
    
    # Create inferencer
    print("Creating inferencer...")
    inferencer = Inferencer(model=model, config=config, device='cuda')
    
    # Load checkpoint
    print(f"Loading checkpoint...")
    inferencer.load_checkpoint(checkpoint_path)
    
    # Extract features
    print("Extracting features...")
    features_dict = inferencer.extract_features_from_folder(
        folder_path=input_folder,
        output_path=output_path,
        batch_size=32,
        num_workers=4,
        save_format='both'
    )
    
    print(f"\n✓ Extracted features for {len(features_dict)} images")
    print(f"✓ Saved to {output_path}.pkl and {output_path}.npz")
    
    return f"{output_path}.pkl"


def step2_load_and_explore(features_path):
    """Step 2: Load and explore extracted features."""
    print("\n" + "=" * 70)
    print("STEP 2: Loading and Exploring Features")
    print("=" * 70)
    
    if not Path(features_path).exists():
        print(f"Features file not found: {features_path}")
        return None
    
    # Load features
    with open(features_path, 'rb') as f:
        features_dict = pickle.load(f)
    
    print(f"Loaded features for {len(features_dict)} images")
    
    # Get basic statistics
    filenames = list(features_dict.keys())
    features_array = np.array([features_dict[fn] for fn in filenames])
    
    print(f"\nFeature array shape: {features_array.shape}")
    print(f"Feature dimension: {features_array.shape[1]}")
    print(f"Mean feature magnitude: {np.linalg.norm(features_array, axis=1).mean():.4f}")
    
    # Show sample features
    print("\nSample features (first 3 images):")
    for i, filename in enumerate(filenames[:3]):
        feat = features_dict[filename]
        print(f"  {filename}:")
        print(f"    Shape: {feat.shape}")
        print(f"    Mean: {feat.mean():.4f}, Std: {feat.std():.4f}")
        print(f"    Min: {feat.min():.4f}, Max: {feat.max():.4f}")
    
    return features_dict, filenames, features_array


def step3_similarity_search(features_dict, filenames, features_array):
    """Step 3: Find similar images."""
    print("\n" + "=" * 70)
    print("STEP 3: Image Similarity Search")
    print("=" * 70)
    
    if len(features_array) < 2:
        print("Need at least 2 images for similarity search")
        return
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Pick a query image (first one)
    query_idx = 0
    query_filename = filenames[query_idx]
    query_features = features_array[query_idx].reshape(1, -1)
    
    print(f"Query image: {query_filename}")
    
    # Compute similarities
    similarities = cosine_similarity(query_features, features_array)[0]
    
    # Get top 5 most similar images
    top_k = min(5, len(filenames))
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nTop {top_k} most similar images:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {filenames[idx]}: similarity = {similarities[idx]:.4f}")


def step4_clustering(features_array, filenames):
    """Step 4: Cluster images."""
    print("\n" + "=" * 70)
    print("STEP 4: Clustering Images")
    print("=" * 70)
    
    if len(features_array) < 5:
        print("Need at least 5 images for clustering")
        return
    
    from sklearn.cluster import KMeans
    
    # Determine number of clusters
    n_clusters = min(3, len(features_array))
    
    print(f"Clustering {len(features_array)} images into {n_clusters} clusters...")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_array)
    
    # Show cluster distribution
    print("\nCluster distribution:")
    for cluster_id in range(n_clusters):
        cluster_images = [filenames[i] for i in range(len(filenames)) if clusters[i] == cluster_id]
        print(f"  Cluster {cluster_id}: {len(cluster_images)} images")
        
        # Show first 3 images in each cluster
        for img in cluster_images[:3]:
            print(f"    - {img}")
        if len(cluster_images) > 3:
            print(f"    ... and {len(cluster_images) - 3} more")


def step5_dimensionality_reduction(features_array, filenames):
    """Step 5: Reduce to 2D for visualization."""
    print("\n" + "=" * 70)
    print("STEP 5: Dimensionality Reduction (PCA)")
    print("=" * 70)
    
    if len(features_array) < 3:
        print("Need at least 3 images for dimensionality reduction")
        return
    
    from sklearn.decomposition import PCA
    
    # Reduce to 2D using PCA
    print(f"Reducing {features_array.shape[1]}-dimensional features to 2D...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_array)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Component 1: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"Component 2: {pca.explained_variance_ratio_[1]:.4f}")
    
    print("\n2D coordinates (first 5 images):")
    for i in range(min(5, len(filenames))):
        print(f"  {filenames[i]}: ({features_2d[i, 0]:.4f}, {features_2d[i, 1]:.4f})")
    
    # Try to visualize if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
        plt.title('PCA Visualization of Image Features')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        
        output_path = 'results/feature_visualization.png'
        Path('results').mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {output_path}")
        
    except ImportError:
        print("\nNote: matplotlib not available for visualization")


def main():
    """Run complete workflow."""
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION WORKFLOW")
    print("=" * 70)
    print("\nThis script demonstrates:")
    print("1. Extracting features from a trained model")
    print("2. Loading and exploring features")
    print("3. Finding similar images")
    print("4. Clustering images")
    print("5. Reducing dimensions for visualization")
    print()
    
    # Step 1: Extract features
    features_path = step1_extract_features()
    
    if features_path is None:
        print("\nSkipping remaining steps due to missing files.")
        print("Please update the paths in step1_extract_features() and run again.")
        return
    
    # Step 2: Load and explore
    result = step2_load_and_explore(features_path)
    if result is None:
        return
    
    features_dict, filenames, features_array = result
    
    # Step 3: Similarity search
    step3_similarity_search(features_dict, filenames, features_array)
    
    # Step 4: Clustering
    step4_clustering(features_array, filenames)
    
    # Step 5: Dimensionality reduction
    step5_dimensionality_reduction(features_array, filenames)
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETED!")
    print("=" * 70)


if __name__ == '__main__':
    main()





