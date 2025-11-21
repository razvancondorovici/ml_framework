#!/usr/bin/env python3
"""Test script for feature extraction functionality."""

import sys
from pathlib import Path
import torch
import numpy as np
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.registry import build_classifier
from engine.inferencer import Inferencer
from PIL import Image


def create_dummy_images(output_dir, num_images=5):
    """Create dummy test images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        # Create random image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(output_dir / f"test_image_{i:03d}.jpg")
    
    return output_dir


def test_enable_disable_feature_extraction():
    """Test enabling and disabling feature extraction mode."""
    print("\n" + "=" * 60)
    print("TEST 1: Enable/Disable Feature Extraction")
    print("=" * 60)
    
    # Create model
    model = build_classifier(
        backbone='timm_efficientnet_b0',
        num_classes=2,
        pretrained=False,
        dropout=0.2
    )
    
    # Create inferencer
    inferencer = Inferencer(model=model, config={}, device='cpu')
    
    # Test enabling
    assert not inferencer.feature_extraction_mode, "Should start disabled"
    inferencer.enable_feature_extraction()
    assert inferencer.feature_extraction_mode, "Should be enabled"
    
    # Test double enabling
    inferencer.enable_feature_extraction()  # Should just warn
    
    # Test disabling
    inferencer.disable_feature_extraction()
    assert not inferencer.feature_extraction_mode, "Should be disabled"
    
    print("✓ Enable/disable feature extraction works correctly")


def test_extract_features():
    """Test feature extraction from images."""
    print("\n" + "=" * 60)
    print("TEST 2: Extract Features from Images")
    print("=" * 60)
    
    # Create temporary directory with dummy images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create dummy images
        num_images = 5
        image_dir = create_dummy_images(temp_dir, num_images)
        print(f"Created {num_images} dummy images in {image_dir}")
        
        # Create model
        model = build_classifier(
            backbone='timm_efficientnet_b0',
            num_classes=2,
            pretrained=False,
            dropout=0.2
        )
        
        # Create inferencer
        config = {'transforms': {}, 'amp': False}
        inferencer = Inferencer(model=model, config=config, device='cpu')
        
        # Extract features
        output_path = Path(temp_dir) / "features"
        features_dict = inferencer.extract_features_from_folder(
            folder_path=image_dir,
            output_path=output_path,
            batch_size=2,
            num_workers=0,  # No multiprocessing for testing
            save_format='both'
        )
        
        # Verify results
        assert len(features_dict) == num_images, f"Should have {num_images} features"
        
        # Check feature dimensions
        sample_features = list(features_dict.values())[0]
        expected_dim = 1280  # EfficientNet-B0
        assert sample_features.shape[0] == expected_dim, f"Feature dim should be {expected_dim}"
        
        # Check output files exist
        assert (output_path.with_suffix('.pkl')).exists(), "Pickle file should exist"
        assert (output_path.with_suffix('.npz')).exists(), "NPZ file should exist"
        assert (output_path.parent / f"{output_path.stem}_metadata.json").exists(), "Metadata should exist"
        
        print(f"✓ Extracted features from {num_images} images")
        print(f"✓ Feature dimension: {sample_features.shape[0]}")
        print(f"✓ Output files created successfully")
        
        # Test loading features
        import pickle
        with open(output_path.with_suffix('.pkl'), 'rb') as f:
            loaded_features = pickle.load(f)
        
        assert len(loaded_features) == num_images, "Loaded features should match"
        print(f"✓ Features can be loaded from pickle file")
        
        # Test loading from npz
        npz_data = np.load(output_path.with_suffix('.npz'))
        assert 'filenames' in npz_data, "NPZ should have filenames"
        assert 'features' in npz_data, "NPZ should have features"
        print(f"✓ Features can be loaded from npz file")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temporary directory")


def test_feature_consistency():
    """Test that features are consistent across calls."""
    print("\n" + "=" * 60)
    print("TEST 3: Feature Consistency")
    print("=" * 60)
    
    # Create temporary directory with dummy image
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create one dummy image
        image_dir = create_dummy_images(temp_dir, num_images=1)
        
        # Create model
        model = build_classifier(
            backbone='timm_efficientnet_b0',
            num_classes=2,
            pretrained=False,
            dropout=0.2
        )
        
        # Set model to eval mode for consistency
        model.eval()
        
        # Create inferencer
        config = {'transforms': {}, 'amp': False}
        inferencer = Inferencer(model=model, config=config, device='cpu')
        
        # Extract features twice
        output_path1 = Path(temp_dir) / "features1"
        features_dict1 = inferencer.extract_features_from_folder(
            folder_path=image_dir,
            output_path=output_path1,
            batch_size=1,
            num_workers=0,
            save_format='pickle'
        )
        
        output_path2 = Path(temp_dir) / "features2"
        features_dict2 = inferencer.extract_features_from_folder(
            folder_path=image_dir,
            output_path=output_path2,
            batch_size=1,
            num_workers=0,
            save_format='pickle'
        )
        
        # Compare features
        filename = list(features_dict1.keys())[0]
        features1 = features_dict1[filename]
        features2 = features_dict2[filename]
        
        diff = np.abs(features1 - features2).max()
        assert diff < 1e-5, f"Features should be consistent (max diff: {diff})"
        
        print(f"✓ Features are consistent across multiple extractions (max diff: {diff:.2e})")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION TESTS")
    print("=" * 60)
    
    try:
        test_enable_disable_feature_extraction()
        test_extract_features()
        test_feature_consistency()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)





