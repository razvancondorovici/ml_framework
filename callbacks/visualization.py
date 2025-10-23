"""Visualization callbacks for training."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from .base import Callback
from utils.plotting import plot_sample_grid, plot_segmentation_overlay, plot_loss_curves


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
        
        # Get model and device
        model = self.trainer.model
        device = next(model.parameters()).device
        model.eval()
        
        # Create visualizations for both training and validation sets
        self._create_dataset_visualization(model, device, epoch, 'train')
        self._create_dataset_visualization(model, device, epoch, 'val')
    
    def _create_dataset_visualization(self, model, device, epoch: int, dataset_type: str):
        """Create visualization for a specific dataset (train or val).
        
        Args:
            model: The model to use for predictions
            device: Device to run inference on
            epoch: Current epoch number
            dataset_type: 'train' or 'val'
        """
        # Get the appropriate dataloader
        if dataset_type == 'train':
            dataloader = getattr(self.trainer, 'train_dataloader', None)
        else:  # val
            dataloader = getattr(self.trainer, 'val_dataloader', None)
            
        if dataloader is None:
            return
        
        # Collect samples
        samples = []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
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
        
        # Create visualization with dataset type suffix
        self._create_visualization(samples, epoch, dataset_type)
    
    def _create_visualization(self, samples: List[Dict[str, torch.Tensor]], epoch: int, dataset_type: str = 'val'):
        """Create visualization for samples.
        
        Args:
            samples: List of sample dictionaries
            epoch: Current epoch number
            dataset_type: 'train' or 'val' for filename suffix
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
        if preds.dim() == 3:  # Segmentation: (N, H, W)
            self._create_segmentation_visualization(images_np, targets_np, preds_np, epoch, dataset_type)
        else:  # Classification: (N,)
            self._create_classification_visualization(images_np, targets_np, preds_np, epoch, dataset_type)
    
    def _create_classification_visualization(self, images: np.ndarray, targets: np.ndarray, 
                                           preds: np.ndarray, epoch: int, dataset_type: str = 'val'):
        """Create classification visualization.
        
        Args:
            images: Image data (N, C, H, W)
            targets: Target labels (N,)
            preds: Predicted labels (N,)
            epoch: Current epoch number
            dataset_type: 'train' or 'val' for filename suffix
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
        save_path = self.save_dir / f'epoch_{epoch:03d}_samples_{dataset_type}.png'
        plot_sample_grid(
            images=images_display,
            labels=targets,
            predictions=preds,
            class_names=self.class_names,
            save_path=save_path,
            title=f'{dataset_type.title()} Samples - Epoch {epoch}',
            max_samples=self.num_samples
        )
    
    def _create_segmentation_visualization(self, images: np.ndarray, targets: np.ndarray,
                                         preds: np.ndarray, epoch: int, dataset_type: str = 'val'):
        """Create segmentation visualization.
        
        Args:
            images: Image data (N, C, H, W)
            targets: Target masks (N, H, W)
            preds: Predicted masks (N, H, W)
            epoch: Current epoch number
            dataset_type: 'train' or 'val' for filename suffix
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
        
        # Calculate grid dimensions: each sample has 3 columns (Input, GT, Pred)
        n_samples = min(len(images_display), self.num_samples)
        n_cols = 3  # Input | Ground Truth | Prediction
        n_rows = n_samples
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Add column headers
        fig.suptitle(f'{dataset_type.title()} Segmentation Samples - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Create each sample row
        for i in range(n_samples):
            # Input image
            axes[i, 0].imshow(images_display[i], cmap='gray' if len(images_display[i].shape) == 2 else None)
            axes[i, 0].set_title('Input Image' if i == 0 else '')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(targets[i], cmap='tab10')
            axes[i, 1].set_title('Ground Truth' if i == 0 else '')
            axes[i, 1].axis('off')
            
            # Predicted mask
            axes[i, 2].imshow(preds[i], cmap='tab10')
            axes[i, 2].set_title('Prediction' if i == 0 else '')
            axes[i, 2].axis('off')
        
        # Save the visualization
        save_path = self.save_dir / f'epoch_{epoch:03d}_samples_{dataset_type}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


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


class LossCurveVisualizer(Callback):
    """Callback for visualizing training and validation loss curves."""
    
    def __init__(self, 
                 save_dir: str, 
                 save_every_n_epochs: int = 5,
                 title: str = "Training Progress"):
        """Initialize loss curve visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            save_every_n_epochs: Save every N epochs
            title: Plot title
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.title = title
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track loss history
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Save loss curve plot at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer:
            return
        
        # Get metrics from kwargs
        metrics = kwargs.get('metrics', {})
        
        # Extract losses
        train_loss = metrics.get('train_loss')
        val_loss = metrics.get('val_loss')
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
            self.epochs.append(epoch)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        # Only save plot every N epochs or if we have both train and val losses
        if (epoch % self.save_every_n_epochs == 0 or 
            (len(self.train_losses) > 0 and len(self.val_losses) > 0)):
            self._create_loss_plot(epoch)
    
    def on_train_end(self, **kwargs):
        """Save final loss curve plot at end of training."""
        if len(self.train_losses) > 0:
            self._create_loss_plot(self.epochs[-1] if self.epochs else 0, is_final=True)
    
    def _create_loss_plot(self, epoch: int, is_final: bool = False):
        """Create loss curve plot.
        
        Args:
            epoch: Current epoch number
            is_final: Whether this is the final plot
        """
        if len(self.train_losses) == 0:
            return
        
        # Prepare data for plotting
        train_losses = self.train_losses.copy()
        val_losses = self.val_losses.copy()
        
        # Pad val_losses with None if shorter than train_losses
        while len(val_losses) < len(train_losses):
            val_losses.append(None)
        
        # Filter out None values for plotting
        plot_train_losses = train_losses
        plot_val_losses = [v for v in val_losses if v is not None]
        
        # Create plot
        if len(plot_val_losses) > 0:
            # Both train and val losses available
            plot_loss_curves(
                train_losses=plot_train_losses,
                val_losses=plot_val_losses,
                save_path=self.save_dir / f'loss_curves_epoch_{epoch:03d}.png',
                title=f'{self.title} - Epoch {epoch}'
            )
        else:
            # Only train losses available
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(plot_train_losses) + 1)
            plt.plot(epochs, plot_train_losses, 'b-', label='Training Loss', linewidth=2)
            
            plt.title(f'{self.title} - Epoch {epoch}', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.save_dir / f'loss_curves_epoch_{epoch:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save final plot with a standard name
        if is_final:
            final_save_path = self.save_dir / 'loss_curves_final.png'
            if len(plot_val_losses) > 0:
                plot_loss_curves(
                    train_losses=plot_train_losses,
                    val_losses=plot_val_losses,
                    save_path=final_save_path,
                    title=f'{self.title} - Final'
                )
            else:
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(plot_train_losses) + 1)
                plt.plot(epochs, plot_train_losses, 'b-', label='Training Loss', linewidth=2)
                
                plt.title(f'{self.title} - Final', fontsize=16, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(final_save_path, dpi=150, bbox_inches='tight')
                plt.close()
