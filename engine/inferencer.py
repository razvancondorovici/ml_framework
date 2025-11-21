"""Inference engine for PyTorch models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
import shutil
import pickle

from utils.device import get_device, move_to_device
from utils.logger import StructuredLogger
from utils.checkpoint import load_checkpoint
from transforms.augmentations import get_default_classification_transforms, get_default_segmentation_transforms, get_classification_transforms


class InferenceImageDataset(Dataset):
    """Simple dataset for inference on a list of image files."""
    def __init__(self, image_files: List[Path], transform=None):
        self.image_files = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Log warning using print since we don't have logger access here
            print(f"Warning: Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Return dummy label (not used for inference)
        return image, 0


class Inferencer:
    """Inference engine for PyTorch models."""
    
    def __init__(self,
                 model: nn.Module,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[Union[str, torch.device]] = None):
        """Initialize inferencer.
        
        Args:
            model: PyTorch model for inference
            config: Inference configuration
            device: Device to run inference on
        """
        self.model = model
        self.config = config or {}
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Set up inference components
        self._setup_inference_components()
        
        # Initialize logger
        self.logger = StructuredLogger(self.config.get('log_dir', 'logs'), 'inference')
    
    def _setup_inference_components(self):
        """Set up inference components from config."""
        # Mixed precision
        self.use_amp = self.config.get('amp', True)
        
        # TTA settings
        self.use_tta = self.config.get('tta', False)
        self.tta_transforms = self.config.get('tta_transforms', ['horizontal_flip'])
        
        # Output settings
        self.output_format = self.config.get('output_format', 'probabilities')  # 'probabilities', 'predictions', 'logits'
        self.save_probabilities = self.config.get('save_probabilities', True)
        self.save_predictions = self.config.get('save_predictions', True)
        
        # Feature extraction mode
        self.feature_extraction_mode = False
        self.original_head = None
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        epoch, best_metric, config = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            strict=False
        )
        
        self.logger.info(f"Loaded checkpoint from epoch {epoch}, best metric: {best_metric:.4f}")
    
    def enable_feature_extraction(self):
        """Enable feature extraction mode by removing classification head."""
        if self.feature_extraction_mode:
            self.logger.warning("Feature extraction mode already enabled")
            return
        
        # Store original head and replace with identity
        if hasattr(self.model, 'forward_features'):
            # timm models have forward_features method - we'll use that
            self.feature_extraction_mode = True
            self.logger.info("Using forward_features method for feature extraction")
        elif hasattr(self.model, 'classifier'):
            self.original_head = self.model.classifier
            self.model.classifier = nn.Identity()
            self.feature_extraction_mode = True
            self.logger.info("Replaced classifier with Identity for feature extraction")
        elif hasattr(self.model, 'fc'):
            self.original_head = self.model.fc
            self.model.fc = nn.Identity()
            self.feature_extraction_mode = True
            self.logger.info("Replaced fc with Identity for feature extraction")
        elif hasattr(self.model, 'head'):
            self.original_head = self.model.head
            self.model.head = nn.Identity()
            self.feature_extraction_mode = True
            self.logger.info("Replaced head with Identity for feature extraction")
        else:
            raise ValueError("Could not identify classification head in model")
    
    def disable_feature_extraction(self):
        """Disable feature extraction mode by restoring classification head."""
        if not self.feature_extraction_mode:
            self.logger.warning("Feature extraction mode not enabled")
            return
        
        # Restore original head
        if self.original_head is not None:
            if hasattr(self.model, 'classifier'):
                self.model.classifier = self.original_head
            elif hasattr(self.model, 'fc'):
                self.model.fc = self.original_head
            elif hasattr(self.model, 'head'):
                self.model.head = self.original_head
            self.original_head = None
        
        self.feature_extraction_mode = False
        self.logger.info("Feature extraction mode disabled")
    
    def extract_features(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """Extract features from dataloader.
        
        Args:
            dataloader: Data loader for feature extraction
            
        Returns:
            Dictionary containing extracted features
        """
        if not self.feature_extraction_mode:
            raise ValueError("Feature extraction mode not enabled. Call enable_feature_extraction() first.")
        
        all_features = []
        
        num_batches = len(dataloader)
        self.logger.info(f"Extracting features from {num_batches} batches")
        
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Extracting features")):
                # Move to device
                inputs = move_to_device(inputs, self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        if hasattr(self.model, 'forward_features'):
                            features = self.model.forward_features(inputs)
                        else:
                            features = self.model(inputs)
                else:
                    if hasattr(self.model, 'forward_features'):
                        features = self.model.forward_features(inputs)
                    else:
                        features = self.model(inputs)
                
                # Global average pooling if features are 4D (B, C, H, W)
                if features.dim() == 4:
                    features = features.mean([2, 3])
                
                all_features.append(features.cpu())
        
        # Concatenate all features
        all_features = torch.cat(all_features, dim=0).numpy()
        
        return all_features
    
    def predict(self, 
                dataloader: DataLoader,
                return_probabilities: bool = True,
                return_predictions: bool = True,
                return_logits: bool = False) -> Dict[str, np.ndarray]:
        """Make predictions on given dataloader.
        
        Args:
            dataloader: Data loader for prediction
            return_probabilities: Whether to return probabilities
            return_predictions: Whether to return predictions
            return_logits: Whether to return logits
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        all_predictions = []
        all_probabilities = []
        all_logits = []
        
        num_batches = len(dataloader)
        self.logger.info(f"Starting inference on {num_batches} batches")
        
        if num_batches == 0:
            self.logger.warning("Dataloader is empty - no batches to process")
            # Return empty results with proper structure
            results = {}
            if return_probabilities:
                results['probabilities'] = np.array([])
            if return_predictions:
                results['predictions'] = np.array([])
            if return_logits:
                results['logits'] = np.array([])
            return results
        
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Inferencing")):
                # Move to device
                inputs = move_to_device(inputs, self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # Store outputs
                if return_logits:
                    all_logits.append(outputs.cpu())
                
                if return_probabilities:
                    if outputs.dim() == 4:  # Segmentation
                        probabilities = torch.softmax(outputs, dim=1)
                    else:  # Classification
                        probabilities = torch.softmax(outputs, dim=1)
                    all_probabilities.append(probabilities.cpu())
                
                if return_predictions:
                    if outputs.dim() == 4:  # Segmentation
                        predictions = torch.argmax(outputs, dim=1)
                    else:  # Classification
                        predictions = torch.argmax(outputs, dim=1)
                    all_predictions.append(predictions.cpu())
        
        # Concatenate results
        results = {}
        if all_logits:
            results['logits'] = torch.cat(all_logits).numpy()
        if all_probabilities:
            results['probabilities'] = torch.cat(all_probabilities).numpy()
        if all_predictions:
            results['predictions'] = torch.cat(all_predictions).numpy()
        
        return results
    
    def predict_with_tta(self, 
                        dataloader: DataLoader,
                        tta_transforms: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Make predictions with Test-Time Augmentation.
        
        Args:
            dataloader: Data loader for prediction
            tta_transforms: List of TTA transforms to apply
            
        Returns:
            Dictionary containing averaged predictions and probabilities
        """
        if tta_transforms is None:
            tta_transforms = self.tta_transforms
        
        # Get original predictions
        original_results = self.predict(dataloader, return_probabilities=True, return_predictions=True)
        
        # Apply TTA transforms
        tta_results = []
        for transform_name in tta_transforms:
            # Create transform
            if transform_name == 'horizontal_flip':
                transform = lambda x: torch.flip(x, dims=[3])  # Flip width dimension
            elif transform_name == 'vertical_flip':
                transform = lambda x: torch.flip(x, dims=[2])  # Flip height dimension
            else:
                continue  # Skip unknown transforms
            
            # Apply transform and predict
            tta_predictions = []
            tta_probabilities = []
            
            with torch.no_grad():
                for inputs, _ in tqdm(dataloader, desc=f"TTA {transform_name}"):
                    # Move to device
                    inputs = move_to_device(inputs, self.device)
                    
                    # Apply transform
                    transformed_inputs = transform(inputs)
                    
                    # Forward pass
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(transformed_inputs)
                    else:
                        outputs = self.model(transformed_inputs)
                    
                    # Store results
                    if outputs.dim() == 4:  # Segmentation
                        probabilities = torch.softmax(outputs, dim=1)
                        predictions = torch.argmax(outputs, dim=1)
                    else:  # Classification
                        probabilities = torch.softmax(outputs, dim=1)
                        predictions = torch.argmax(outputs, dim=1)
                    
                    tta_probabilities.append(probabilities.cpu())
                    tta_predictions.append(predictions.cpu())
            
            # Concatenate results
            tta_results.append({
                'probabilities': torch.cat(tta_probabilities).numpy(),
                'predictions': torch.cat(tta_predictions).numpy()
            })
        
        # Average results
        averaged_results = {}
        
        # Average probabilities
        all_probabilities = [original_results['probabilities']]
        for tta_result in tta_results:
            all_probabilities.append(tta_result['probabilities'])
        
        averaged_probabilities = np.mean(all_probabilities, axis=0)
        averaged_results['probabilities'] = averaged_probabilities
        
        # Get predictions from averaged probabilities
        if averaged_probabilities.ndim == 4:  # Segmentation
            averaged_predictions = np.argmax(averaged_probabilities, axis=1)
        else:  # Classification
            averaged_predictions = np.argmax(averaged_probabilities, axis=1)
        
        averaged_results['predictions'] = averaged_predictions
        
        return averaged_results
    
    def predict_folder(self, 
                      folder_path: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None,
                      class_names: Optional[List[str]] = None,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      copy_images_to_class_folders: bool = True) -> Dict[str, Any]:
        """Predict on images in a folder.
        
        Args:
            folder_path: Path to folder containing images (can be flat or have class subdirectories)
            output_path: Path to save results
            class_names: List of class names
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
            copy_images_to_class_folders: Whether to copy images to class folders
            
        Returns:
            Prediction results
        """
        folder_path = Path(folder_path)
        
        # Get image files - search recursively
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']:
            image_files.extend(folder_path.glob(f'**/*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext}'))  # Also check direct files
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")
        
        self.logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        # Create dataset using module-level class (required for multiprocessing)
        transform = get_default_classification_transforms(split='test')
        dataset = InferenceImageDataset(image_files, transform=transform)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Make predictions
        if self.use_tta:
            results = self.predict_with_tta(dataloader)
        else:
            results = self.predict(dataloader)
        
        # Check if we have results
        if 'predictions' not in results or len(results['predictions']) == 0:
            raise ValueError("No predictions generated. Check that the dataloader has batches and images are loading correctly.")
        
        # Create results dataframe
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Convert class_names to regular Python list if it's OmegaConf ListConfig
        try:
            from omegaconf import ListConfig
            if ListConfig is not None and isinstance(class_names, ListConfig):
                class_names = list(class_names)
        except ImportError:
            pass
        
        # Create results
        results_data = []
        for i, image_file in enumerate(image_files):
            # Convert numpy int64 to Python int for indexing
            pred_idx = int(predictions[i])
            
            result = {
                'image_path': str(image_file),
                'prediction': pred_idx,
                'confidence': float(probabilities[i].max())
            }
            
            # Add class name if provided
            if class_names and pred_idx < len(class_names):
                result['class_name'] = class_names[pred_idx]
            
            # Add probabilities for each class
            if class_names:
                for j, class_name in enumerate(class_names):
                    if j < probabilities.shape[1]:
                        result[f'prob_{class_name}'] = float(probabilities[i, j])
            
            results_data.append(result)
        
        # Create dataframe
        results_df = pd.DataFrame(results_data)
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            csv_path = output_path.with_suffix('.csv')
            results_df.to_csv(csv_path, index=False)
            
            # Save JSON
            json_path = output_path.with_suffix('.json')
            results_df.to_json(json_path, orient='records', indent=2)
            
            self.logger.info(f"Results saved to {csv_path} and {json_path}")
            
            # Copy images to class folders if requested
            if copy_images_to_class_folders:
                images_dir = output_path.parent / f"{output_path.stem}_images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Copying images to class folders in {images_dir}")
                
                # Copy each image to its predicted class folder
                for i, image_file in enumerate(tqdm(image_files, desc="Copying images")):
                    pred_idx = int(predictions[i])
                    
                    # Determine class name
                    if class_names and pred_idx < len(class_names):
                        class_name = class_names[pred_idx]
                    else:
                        class_name = f"class_{pred_idx}"
                    
                    # Create class folder
                    class_folder = images_dir / class_name
                    class_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Copy image to class folder (preserve original filename)
                    dest_path = class_folder / image_file.name
                    shutil.copy2(image_file, dest_path)
                
                self.logger.info(f"Images copied to {images_dir}")
        
        return {
            'results': results_df,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def predict_csv(self, 
                   csv_path: Union[str, Path],
                   image_column: str = 'image_path',
                   output_path: Optional[Union[str, Path]] = None,
                   class_names: Optional[List[str]] = None,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   copy_images_to_class_folders: bool = True) -> Dict[str, Any]:
        """Predict on images specified in a CSV file.
        
        Args:
            csv_path: Path to CSV file with image paths
            image_column: Name of column containing image paths
            output_path: Path to save results
            class_names: List of class names
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
            copy_images_to_class_folders: Whether to copy images to class folders
            
        Returns:
            Prediction results
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if image_column not in df.columns:
            raise ValueError(f"Column '{image_column}' not found in CSV")
        
        self.logger.info(f"Loaded {len(df)} images from {csv_path}")
        
        # Create dataset
        from datasets.classification import ImageClassificationDataset
        
        dataset = ImageClassificationDataset(
            data_dir=Path(csv_path).parent,
            annotations_file=csv_path,
            transform=get_default_classification_transforms(split='test')
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Make predictions
        if self.use_tta:
            results = self.predict_with_tta(dataloader)
        else:
            results = self.predict(dataloader)
        
        # Create results dataframe
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Convert class_names to regular Python list if it's OmegaConf ListConfig
        try:
            from omegaconf import ListConfig
            if ListConfig is not None and isinstance(class_names, ListConfig):
                class_names = list(class_names)
        except ImportError:
            pass
        
        # Add predictions to original dataframe (convert numpy int64 to int)
        df['prediction'] = predictions.astype(int)
        df['confidence'] = probabilities.max(axis=1)
        
        # Add class name if provided
        if class_names:
            df['class_name'] = df['prediction'].apply(lambda x: class_names[int(x)] if int(x) < len(class_names) else 'Unknown')
        
        # Add probabilities for each class
        if class_names:
            for j, class_name in enumerate(class_names):
                if j < probabilities.shape[1]:
                    df[f'prob_{class_name}'] = probabilities[:, j]
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            csv_path = output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            
            # Save JSON
            json_path = output_path.with_suffix('.json')
            df.to_json(json_path, orient='records', indent=2)
            
            self.logger.info(f"Results saved to {csv_path} and {json_path}")
            
            # Copy images to class folders if requested
            if copy_images_to_class_folders:
                images_dir = output_path.parent / f"{output_path.stem}_images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Copying images to class folders in {images_dir}")
                
                # Copy each image to its predicted class folder
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
                    pred_idx = int(row['prediction'])
                    image_path = Path(row[image_column])
                    
                    # Determine class name
                    if class_names and pred_idx < len(class_names):
                        class_name = class_names[pred_idx]
                    else:
                        class_name = f"class_{pred_idx}"
                    
                    # Create class folder
                    class_folder = images_dir / class_name
                    class_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Copy image to class folder (preserve original filename)
                    if image_path.exists():
                        dest_path = class_folder / image_path.name
                        shutil.copy2(image_path, dest_path)
                    else:
                        self.logger.warning(f"Image not found: {image_path}")
                
                self.logger.info(f"Images copied to {images_dir}")
        
        return {
            'results': df,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def extract_features_from_folder(self,
                                     folder_path: Union[str, Path],
                                     output_path: Union[str, Path],
                                     batch_size: int = 32,
                                     num_workers: int = 4,
                                     save_format: str = 'pickle') -> Dict[str, np.ndarray]:
        """Extract features from images in a folder and save as dictionary.
        
        Args:
            folder_path: Path to folder containing images
            output_path: Path to save features dictionary
            batch_size: Batch size for feature extraction
            num_workers: Number of workers for data loading
            save_format: Format to save features ('pickle', 'npz', or 'both')
            
        Returns:
            Dictionary with {filename: features} pairs
        """
        folder_path = Path(folder_path)
        output_path = Path(output_path)
        
        # Enable feature extraction mode
        was_enabled = self.feature_extraction_mode
        if not was_enabled:
            self.enable_feature_extraction()
        
        try:
            # Get image files - search recursively
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']:
                image_files.extend(folder_path.glob(f'**/*{ext}'))
                image_files.extend(folder_path.glob(f'*{ext}'))
            
            # Remove duplicates and sort
            image_files = sorted(list(set(image_files)))
            
            if not image_files:
                raise ValueError(f"No image files found in {folder_path}")
            
            self.logger.info(f"Found {len(image_files)} images in {folder_path}")
            
            # Get transforms from config
            transform_config = self.config.get('transforms', {})
            if transform_config:
                # Use transforms from config
                transform = get_classification_transforms(transform_config, split='test')
            else:
                # Use default transforms
                transform = get_default_classification_transforms(split='test')
            
            # Create dataset
            dataset = InferenceImageDataset(image_files, transform=transform)
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            # Extract features
            features = self.extract_features(dataloader)
            
            # Create dictionary with filename as key
            features_dict = {}
            for i, image_file in enumerate(image_files):
                # Use just the filename (not full path) as key
                filename = image_file.name
                features_dict[filename] = features[i]
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save features in requested format(s)
            if save_format in ['pickle', 'both']:
                pickle_path = output_path.with_suffix('.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(features_dict, f)
                self.logger.info(f"Features saved to {pickle_path}")
                print(f"Features saved to {pickle_path}")
            
            if save_format in ['npz', 'both']:
                npz_path = output_path.with_suffix('.npz')
                # Save as npz (need to convert dict to arrays)
                filenames = list(features_dict.keys())
                features_array = np.array(list(features_dict.values()))
                np.savez(npz_path, filenames=filenames, features=features_array)
                self.logger.info(f"Features saved to {npz_path}")
                print(f"Features saved to {npz_path}")
            
            # Also save a JSON with metadata
            metadata = {
                'num_images': len(image_files),
                'feature_dim': features.shape[1],
                'source_folder': str(folder_path),
                'filenames': [str(f.name) for f in image_files]
            }
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Metadata saved to {metadata_path}")
            
            print(f"\nFeature extraction completed!")
            print(f"Number of images: {len(image_files)}")
            print(f"Feature dimension: {features.shape[1]}")
            
            return features_dict
            
        finally:
            # Restore original state if we enabled it
            if not was_enabled:
                self.disable_feature_extraction()
    
    def export_model(self, 
                    export_path: Union[str, Path],
                    export_format: str = 'torchscript',
                    input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Export model to different formats.
        
        Args:
            export_path: Path to save exported model
            export_format: Export format ('torchscript', 'onnx')
            input_shape: Input shape for export (C, H, W)
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if export_format == 'torchscript':
            self._export_torchscript(export_path, input_shape)
        elif export_format == 'onnx':
            self._export_onnx(export_path, input_shape)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _export_torchscript(self, export_path: Path, input_shape: Optional[Tuple[int, ...]] = None):
        """Export model to TorchScript."""
        if input_shape is None:
            input_shape = (3, 224, 224)  # Default input shape
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Export model
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    traced_model = torch.jit.trace(self.model, dummy_input)
            else:
                traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Save model
        traced_model.save(str(export_path))
        self.logger.info(f"Model exported to TorchScript: {export_path}")
    
    def _export_onnx(self, export_path: Path, input_shape: Optional[Tuple[int, ...]] = None):
        """Export model to ONNX."""
        if input_shape is None:
            input_shape = (3, 224, 224)  # Default input shape
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Export model
        torch.onnx.export(
            self.model,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"Model exported to ONNX: {export_path}")
    
    def benchmark_model(self, 
                       dataloader: DataLoader,
                       num_warmup: int = 10,
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            dataloader: Data loader for benchmarking
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        self.model.eval()
        
        # Warmup
        self.logger.info("Warming up model...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_warmup:
                    break
                
                inputs = move_to_device(inputs, self.device)
                
                if self.use_amp:
                    with autocast():
                        _ = self.model(inputs)
                else:
                    _ = self.model(inputs)
        
        # Benchmark
        self.logger.info("Benchmarking model...")
        times = []
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_iterations:
                    break
                
                inputs = move_to_device(inputs, self.device)
                
                # Measure time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                if self.use_amp:
                    with autocast():
                        _ = self.model(inputs)
                else:
                    _ = self.model(inputs)
                
                end_time.record()
                torch.cuda.synchronize()
                
                times.append(start_time.elapsed_time(end_time))
        
        # Compute statistics
        times = np.array(times)
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'median_time_ms': float(np.median(times)),
            'throughput_fps': float(1000.0 / np.mean(times))
        }
        
        self.logger.info("Benchmark completed", **results)
        
        return results
