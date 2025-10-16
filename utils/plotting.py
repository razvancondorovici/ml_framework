"""Plotting utilities for training visualization."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import pandas as pd


def setup_plotting(style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
    """Setup matplotlib plotting style.
    
    Args:
        style: Matplotlib style
        figsize: Default figure size
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'


def plot_loss_curves(train_losses: List[float], val_losses: List[float], 
                    save_path: Union[str, Path], title: str = "Training Progress"):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_curves(metrics: Dict[str, List[float]], save_path: Union[str, Path],
                      title: str = "Training Metrics"):
    """Plot multiple metric curves.
    
    Args:
        metrics: Dictionary of metric name to list of values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(list(metrics.values())[0]) + 1)
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, values, linewidth=2)
        plt.title(f'{metric_name.replace("_", " ").title()}', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_lr_curve(learning_rates: List[float], save_path: Union[str, Path],
                 title: str = "Learning Rate Schedule"):
    """Plot learning rate curve.
    
    Args:
        learning_rates: List of learning rates per step/epoch
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    steps = range(len(learning_rates))
    plt.plot(steps, learning_rates, 'g-', linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         save_path: Union[str, Path] = None,
                         title: str = "Confusion Matrix"):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        return plt.gcf()


def plot_roc_curves(y_true: np.ndarray, y_scores: np.ndarray, 
                   class_names: Optional[List[str]] = None,
                   save_path: Union[str, Path] = None,
                   title: str = "ROC Curves"):
    """Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded for multi-class)
        y_scores: Prediction scores/probabilities
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    
    n_classes = y_scores.shape[1] if len(y_scores.shape) > 1 else len(np.unique(y_true))
    
    if len(y_true.shape) == 1:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
    else:
        y_true_bin = y_true
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
        
        class_name = class_names[i] if class_names else f'Class {i}'
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        return plt.gcf()


def plot_pr_curves(y_true: np.ndarray, y_scores: np.ndarray,
                  class_names: Optional[List[str]] = None,
                  save_path: Union[str, Path] = None,
                  title: str = "Precision-Recall Curves"):
    """Plot Precision-Recall curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded for multi-class)
        y_scores: Prediction scores/probabilities
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import average_precision_score
    
    n_classes = y_scores.shape[1] if len(y_scores.shape) > 1 else len(np.unique(y_true))
    
    if len(y_true.shape) == 1:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
    else:
        y_true_bin = y_true
    
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        
        class_name = class_names[i] if class_names else f'Class {i}'
        plt.plot(recall, precision, linewidth=2, label=f'{class_name} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        return plt.gcf()


def plot_sample_grid(images: np.ndarray, labels: np.ndarray, predictions: np.ndarray,
                    class_names: Optional[List[str]] = None,
                    save_path: Union[str, Path] = None,
                    title: str = "Sample Predictions",
                    max_samples: int = 16):
    """Plot a grid of sample predictions.
    
    Args:
        images: Array of images (N, H, W, C) or (N, C, H, W)
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
        max_samples: Maximum number of samples to show
    """
    n_samples = min(len(images), max_samples)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        row = i // n_cols
        col = i % n_cols
        
        # Convert image to display format
        img = images[i]
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:  # Grayscale
            img = img.squeeze(-1)
        
        axes[row, col].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        
        # Handle both scalar and array labels/predictions
        true_val = labels[i]
        pred_val = predictions[i]
        
        # If they're arrays, convert to scalar for comparison
        if hasattr(true_val, 'item') and hasattr(true_val, 'size') and true_val.size == 1:
            true_val = true_val.item()
        elif hasattr(true_val, 'mean'):
            # For multi-element arrays, take mean or mode
            true_val = int(true_val.mean().round())
            
        if hasattr(pred_val, 'item') and hasattr(pred_val, 'size') and pred_val.size == 1:
            pred_val = pred_val.item()
        elif hasattr(pred_val, 'argmax'):
            # For segmentation predictions, take argmax
            pred_val = pred_val.argmax()
        elif hasattr(pred_val, 'mean'):
            # For multi-element arrays, take mean or mode
            pred_val = int(pred_val.mean().round())
        
        true_label = class_names[true_val] if class_names else str(true_val)
        pred_label = class_names[pred_val] if class_names else str(pred_val)
        
        color = 'green' if true_val == pred_val else 'red'
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', 
                                color=color, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        return plt.gcf()


def plot_segmentation_overlay(image: np.ndarray, mask: np.ndarray, prediction: np.ndarray,
                             class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
                             save_path: Union[str, Path] = None,
                             title: str = "Segmentation Results"):
    """Plot segmentation results with overlay.
    
    Args:
        image: Input image (H, W, C)
        mask: Ground truth mask (H, W)
        prediction: Predicted mask (H, W)
        class_colors: Dictionary mapping class IDs to RGB colors
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth mask
    if class_colors:
        mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            mask_colored[mask == class_id] = color
        axes[1].imshow(mask_colored)
    else:
        axes[1].imshow(mask, cmap='tab10')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction mask
    if class_colors:
        pred_colored = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            pred_colored[prediction == class_id] = color
        axes[2].imshow(pred_colored)
    else:
        axes[2].imshow(prediction, cmap='tab10')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        return plt.gcf()
