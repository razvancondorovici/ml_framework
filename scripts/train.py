#!/usr/bin/env python3
"""Training script for PyTorch models."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config, setup_experiment, get_config_parser
from utils.device import get_device_info
from utils.logger import StructuredLogger
from utils.seed import set_seed
from datasets.classification import create_classification_dataset
from datasets.segmentation import create_segmentation_dataset
from transforms.augmentations import get_default_classification_transforms, get_default_segmentation_transforms
from models.registry import build_classifier, build_segmentation_model
from engine.trainer import Trainer
from callbacks.visualization import SampleVisualizer, ConfusionMatrixVisualizer, LearningRateVisualizer
from callbacks.logging import MetricLogger, ProgressLogger, ModelSummaryLogger
from callbacks.checkpoint import ModelCheckpoint, EarlyStopping
from callbacks.base import CallbackList


def create_datasets(config: Dict[str, Any]) -> tuple:
    """Create training and validation datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_config = config['data']
    dataset_type = data_config.get('dataset_type', 'classification')
    
    # Get transforms
    if dataset_type == 'segmentation':
        train_transform = get_default_segmentation_transforms(split='train')
        val_transform = get_default_segmentation_transforms(split='val')
    else:
        train_transform = get_default_classification_transforms(split='train')
        val_transform = get_default_classification_transforms(split='val')
    
    # Create datasets
    if dataset_type == 'segmentation':
        train_dataset = create_segmentation_dataset({
            **data_config,
            'transform': train_transform
        })
        val_dataset = create_segmentation_dataset({
            **data_config,
            'transform': val_transform
        })
        test_dataset = None
    else:
        train_dataset = create_classification_dataset({
            **data_config,
            'transform': train_transform
        })
        val_dataset = create_classification_dataset({
            **data_config,
            'transform': val_transform
        })
        test_dataset = None
    
    return train_dataset, val_dataset, test_dataset


def create_model(config: Dict[str, Any]) -> Any:
    """Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config['model']
    data_config = config['data']
    
    # Get model parameters
    backbone = model_config.get('backbone', 'resnet50')
    num_classes = data_config.get('num_classes', 10)
    pretrained = model_config.get('pretrained', True)
    freeze_backbone = model_config.get('freeze_backbone', False)
    dropout = model_config.get('dropout', 0.0)
    
    # Create model
    if data_config.get('dataset_type') == 'segmentation':
        model = build_segmentation_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        model = build_classifier(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    
    return model


def create_callbacks(config: Dict[str, Any], run_folder: Path) -> CallbackList:
    """Create training callbacks.
    
    Args:
        config: Configuration dictionary
        run_folder: Run folder path
        
    Returns:
        Callback list
    """
    callbacks = []
    
    # Model checkpoint
    checkpoint_config = config.get('callbacks', {}).get('checkpoint', {})
    callbacks.append(ModelCheckpoint(
        checkpoint_dir=str(run_folder / 'checkpoints'),
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=True,
        save_best=True
    ))
    
    # Early stopping
    early_stopping_config = config.get('callbacks', {}).get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        callbacks.append(EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            mode=early_stopping_config.get('mode', 'min'),
            patience=early_stopping_config.get('patience', 10),
            restore_best_weights=True
        ))
    
    # Metric logger
    callbacks.append(MetricLogger(
        log_dir=str(run_folder),
        log_every_n_epochs=1,
        log_every_n_steps=100
    ))
    
    # Progress logger
    callbacks.append(ProgressLogger(
        log_every_n_epochs=1,
        log_every_n_steps=100
    ))
    
    # Model summary logger
    callbacks.append(ModelSummaryLogger(
        log_dir=str(run_folder)
    ))
    
    # Sample visualizer
    sample_config = config.get('callbacks', {}).get('sample_visualizer', {})
    if sample_config.get('enabled', True):
        callbacks.append(SampleVisualizer(
            save_dir=str(run_folder / 'samples'),
            num_samples=sample_config.get('num_samples', 16),
            save_every_n_epochs=sample_config.get('save_every_n_epochs', 5)
        ))
    
    # Confusion matrix visualizer
    cm_config = config.get('callbacks', {}).get('confusion_matrix', {})
    if cm_config.get('enabled', True):
        callbacks.append(ConfusionMatrixVisualizer(
            save_dir=str(run_folder / 'plots'),
            save_every_n_epochs=cm_config.get('save_every_n_epochs', 10)
        ))
    
    # Learning rate visualizer
    lr_config = config.get('callbacks', {}).get('learning_rate', {})
    if lr_config.get('enabled', True):
        callbacks.append(LearningRateVisualizer(
            save_dir=str(run_folder / 'plots'),
            save_every_n_epochs=lr_config.get('save_every_n_epochs', 10)
        ))
    
    return CallbackList(callbacks)


def main():
    """Main training function."""
    # Parse arguments
    parser = get_config_parser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, help='Device to train on (cuda, cpu)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.overrides)
    
    # Setup experiment
    run_folder = setup_experiment(config)
    print(f"Experiment folder: {run_folder}")
    
    # Set random seed
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    
    # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Create logger
    logger = StructuredLogger(run_folder, 'training')
    logger.info("Starting training", config=config)
    
    try:
        # Create datasets
        print("Creating datasets...")
        train_dataset, val_dataset, test_dataset = create_datasets(config)
        print(f"Train dataset: {len(train_dataset)} samples")
        if val_dataset:
            print(f"Val dataset: {len(val_dataset)} samples")
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
        dataloader_config = config.get('dataloader', {})
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=dataloader_config.get('batch_size', 32),
            shuffle=True,
            num_workers=dataloader_config.get('num_workers', 4),
            pin_memory=dataloader_config.get('pin_memory', True),
            persistent_workers=dataloader_config.get('persistent_workers', True),
            drop_last=dataloader_config.get('drop_last', False)
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=dataloader_config.get('batch_size', 32),
                shuffle=False,
                num_workers=dataloader_config.get('num_workers', 4),
                pin_memory=dataloader_config.get('pin_memory', True),
                persistent_workers=dataloader_config.get('persistent_workers', True),
                drop_last=dataloader_config.get('drop_last', False)
            )
        
        # Create model
        print("Creating model...")
        model = create_model(config)
        print(f"Model: {type(model).__name__}")
        
        # Create callbacks
        print("Creating callbacks...")
        callbacks = create_callbacks(config, run_folder)
        
        # Create trainer
        print("Creating trainer...")
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            device=args.device,
            callbacks=callbacks
        )
        
        # Train model
        print("Starting training...")
        epochs = config.get('training', {}).get('epochs', 100)
        history = trainer.fit(epochs=epochs, resume_from_checkpoint=args.resume)
        
        # Save training history
        import json
        history_path = run_folder / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, list) and value and hasattr(value[0], 'tolist'):
                    serializable_history[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)
        
        print(f"Training completed! Results saved to {run_folder}")
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
