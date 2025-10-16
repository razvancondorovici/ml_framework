"""Visualization callbacks for training."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from .base import Callback
from utils.plotting import plot_sample_grid, plot_segmentation_overlay


class SampleVisualizer(Callback):
    """Callback for visualizing sample predictions during training."""
    
    def __init__(self,
                 save_dir: str,
                 num_samples: int = 16,
                 save_every_n_epochs: int = 5,
                 max_samples_per_class: int = 4,
                 class_names: Optional[List[str]] = None):
        """Initialize sample visualizer callback.
        
        Args:
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
            save_every_n_epochs: Save every N epochs
            max_samples_per_class: Maximum samples per class
            class_names: List of class names
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.num_samples = num_samples
        self.save_every_n_epochs = save_every_n_epochs
        self.max_samples_per_class = max_samples_per_class
        self.class_names = class_names
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Save sample visualizations at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer or epoch % self.save_every_n_epochs != 0:
            return
        
        # Get validation data
        val_dataloader = getattr(self.trainer, 'val_dataloader', None)
        if val_dataloader is None:
            return
        
        # Get model and device
        model = self.trainer.model
        device = next(model.parameters()).device
        model.eval()
        
        # Collect samples
        samples = []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_dataloader):
                if len(samples) >= self.num_samples:
                    break
                
                # Move to device
                images = images.to(device)
                targets = targets.to(device)
                
                # Get predictions
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert to predictions
                if outputs.dim() == 4:  # Segmentation
                    preds = torch.argmax(outputs, dim=1)
                else:  # Classification
                    preds = torch.argmax(outputs, dim=1)
                
                # Store samples
                for i in range(min(images.size(0), self.num_samples - len(samples))):
                    samples.append({
                        'image': images[i].cpu(),
                        'target': targets[i].cpu(),
                        'pred': preds[i].cpu()
                    })
        
        if not samples:
            return
        
        # Create visualization
        self._create_visualization(samples, epoch)
    
    def _create_visualization(self, samples: List[Dict[str, torch.Tensor]], epoch: int):
        """Create visualization for samples.
        
        Args:
            samples: List of sample dictionaries
            epoch: Current epoch number
        """
        # Extract data
        images = torch.stack([s['image'] for s in samples])
        targets = torch.stack([s['target'] for s in samples])
        preds = torch.stack([s['pred'] for s in samples])
        
        # Convert to numpy
        images_np = images.numpy()
        targets_np = targets.numpy()
        preds_np = preds.numpy()
        
        # Check if this is segmentation or classification
        if images.dim() == 4 and images.size(1) == 3:  # RGB images
            # Classification
            self._create_classification_visualization(images_np, targets_np, preds_np, epoch)
        else:
            # Segmentation
            self._create_segmentation_visualization(images_np, targets_np, preds_np, epoch)
    
    def _create_classification_visualization(self, images: np.ndarray, targets: np.ndarray, 
                                           preds: np.ndarray, epoch: int):
        """Create classification visualization.
        
        Args:
            images: Image data (N, C, H, W)
            targets: Target labels (N,)
            preds: Predicted labels (N,)
            epoch: Current epoch number
        """
        # Convert images to display format
        if images.shape[1] == 3:  # RGB
            images_display = np.transpose(images, (0, 2, 3, 1))
        else:  # Grayscale
            images_display = np.transpose(images, (0, 2, 3, 1))
            if images_display.shape[-1] == 1:
                images_display = images_display.squeeze(-1)
        
        # Normalize images
        images_display = np.clip(images_display, 0, 1)
        
        # Create visualization
        save_path = self.save_dir / f'epoch_{epoch:03d}_samples.png'
        plot_sample_grid(
            images=images_display,
            labels=targets,
            predictions=preds,
            class_names=self.class_names,
            save_path=save_path,
            title=f'Training Samples - Epoch {epoch}',
            max_samples=self.num_samples
        )
    
    def _create_segmentation_visualization(self, images: np.ndarray, targets: np.ndarray,
                                         preds: np.ndarray, epoch: int):
        """Create segmentation visualization.
        
        Args:
            images: Image data (N, C, H, W)
            targets: Target masks (N, H, W)
            preds: Predicted masks (N, H, W)
            epoch: Current epoch number
        """
        # Convert images to display format
        if images.shape[1] == 3:  # RGB
            images_display = np.transpose(images, (0, 2, 3, 1))
        else:  # Grayscale
            images_display = np.transpose(images, (0, 2, 3, 1))
            if images_display.shape[-1] == 1:
                images_display = images_display.squeeze(-1)
        
        # Normalize images
        images_display = np.clip(images_display, 0, 1)
        
        # Create visualization for each sample
        for i in range(min(len(samples), 4)):  # Show first 4 samples
            save_path = self.save_dir / f'epoch_{epoch:03d}_sample_{i:02d}.png'
            plot_segmentation_overlay(
                image=images_display[i],
                mask=targets[i],
                prediction=preds[i],
                save_path=save_path,
                title=f'Segmentation Sample {i} - Epoch {epoch}'
            )


class ConfusionMatrixVisualizer(Callback):
    """Callback for visualizing confusion matrix during training."""
    
    def __init__(self,
                 save_dir: str,
                 save_every_n_epochs: int = 10,
                 class_names: Optional[List[str]] = None,
                 normalize: bool = True):
        """Initialize confusion matrix visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            save_every_n_epochs: Save every N epochs
            class_names: List of class names
            normalize: Whether to normalize confusion matrix
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.class_names = class_names
        self.normalize = normalize
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Save confusion matrix at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer or epoch % self.save_every_n_epochs != 0:
            return
        
        # Get validation data
        val_dataloader = getattr(self.trainer, 'val_dataloader', None)
        if val_dataloader is None:
            return
        
        # Get model and device
        model = self.trainer.model
        device = next(model.parameters()).device
        model.eval()
        
        # Collect predictions and targets
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = images.to(device)
                targets = targets.to(device)
                
                # Get predictions
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert to predictions
                if outputs.dim() == 4:  # Segmentation
                    preds = torch.argmax(outputs, dim=1)
                    preds = preds.view(-1)
                    targets = targets.view(-1)
                else:  # Classification
                    preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        
        if not all_preds:
            return
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Create confusion matrix
        self._create_confusion_matrix(all_preds, all_targets, epoch)
    
    def _create_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray, epoch: int):
        """Create confusion matrix visualization.
        
        Args:
            preds: Predicted labels
            targets: Target labels
            epoch: Current epoch number
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Compute confusion matrix
        cm = confusion_matrix(targets, preds)
        
        if self.normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Handle None class names
        class_labels = self.class_names if self.class_names is not None else [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='.2f' if self.normalize else 'd',
                   xticklabels=class_labels, yticklabels=class_labels,
                   cmap='Blues')
        
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        save_path = self.save_dir / f'epoch_{epoch:03d}_confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class LearningRateVisualizer(Callback):
    """Callback for visualizing learning rate schedule."""
    
    def __init__(self, save_dir: str, save_every_n_epochs: int = 10):
        """Initialize learning rate visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            save_every_n_epochs: Save every N epochs
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track learning rates
        self.lr_history = []
        self.epochs = []
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Track learning rate at start of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer or not self.trainer.optimizer:
            return
        
        # Get current learning rates
        lrs = [group['lr'] for group in self.trainer.optimizer.param_groups]
        self.lr_history.append(lrs)
        self.epochs.append(epoch)
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Save learning rate plot at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer or epoch % self.save_every_n_epochs != 0:
            return
        
        # Create learning rate plot
        self._create_lr_plot(epoch)
    
    def _create_lr_plot(self, epoch: int):
        """Create learning rate plot.
        
        Args:
            epoch: Current epoch number
        """
        if not self.lr_history:
            return
        
        # Convert to numpy arrays
        lr_history = np.array(self.lr_history)
        epochs = np.array(self.epochs)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        for i in range(lr_history.shape[1]):
            plt.plot(epochs, lr_history[:, i], label=f'Parameter Group {i}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Save plot
        save_path = self.save_dir / f'epoch_{epoch:03d}_learning_rate.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
