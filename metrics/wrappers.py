"""Metrics wrappers using torchmetrics with Kaggle compatibility."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Literal
import numpy as np


# Try to import torchmetrics, fallback to simple implementation if not available
try:
    from torchmetrics import (
        Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision,
        ConfusionMatrix, MeanSquaredError, MeanAbsoluteError
    )
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
        MulticlassAUROC, MulticlassAveragePrecision, MulticlassConfusionMatrix,
        MulticlassJaccardIndex
    )
    from torchmetrics.regression import (
        MeanSquaredError, MeanAbsoluteError, R2Score
    )
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("Warning: torchmetrics not available, using simple fallback implementation")
    TORCHMETRICS_AVAILABLE = False
    # Import fallback implementation
    from .kaggle_compatible import KaggleCompatibleMetrics, create_metrics as create_kaggle_metrics

# Import KaggleCompatibleMetrics for type hints regardless of torchmetrics availability
try:
    from .kaggle_compatible import KaggleCompatibleMetrics
except ImportError:
    # If we can't import it, create a dummy class for type hints
    class KaggleCompatibleMetrics:
        pass

class MetricsWrapper:
    """Wrapper for torchmetrics with consistent interface."""
    
    def __init__(self, 
                 num_classes: int,
                 task: str = 'multiclass',
                 average: str = 'macro',
                 threshold: float = 0.5):
        """Initialize metrics wrapper.
        
        Args:
            num_classes: Number of classes
            task: Task type ('multiclass', 'multilabel', 'binary')
            average: Averaging method ('macro', 'micro', 'weighted', 'none')
            threshold: Threshold for binary/multilabel tasks
        """
        self.num_classes = num_classes
        self.task = task
        self.average = average
        self.threshold = threshold
        
        # Initialize metrics
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict[str, nn.Module]:
        """Initialize metrics based on task type."""
        metrics = {}
        
        if self.task == 'multiclass':
            metrics['accuracy'] = MulticlassAccuracy(
                num_classes=self.num_classes,
                average='micro'
            )
            metrics['accuracy_macro'] = MulticlassAccuracy(
                num_classes=self.num_classes,
                average='macro'
            )
            metrics['precision'] = MulticlassPrecision(
                num_classes=self.num_classes,
                average=self.average
            )
            metrics['recall'] = MulticlassRecall(
                num_classes=self.num_classes,
                average=self.average
            )
            metrics['f1'] = MulticlassF1Score(
                num_classes=self.num_classes,
                average=self.average
            )
            metrics['auroc'] = MulticlassAUROC(
                num_classes=self.num_classes,
                average=self.average
            )
            metrics['auprc'] = MulticlassAveragePrecision(
                num_classes=self.num_classes,
                average=self.average
            )
            metrics['confusion_matrix'] = MulticlassConfusionMatrix(
                num_classes=self.num_classes
            )
        
        elif self.task == 'multilabel':
            metrics['accuracy'] = Accuracy(
                task='multilabel',
                num_labels=self.num_classes,
                threshold=self.threshold
            )
            metrics['precision'] = Precision(
                task='multilabel',
                num_labels=self.num_classes,
                average=self.average,
                threshold=self.threshold
            )
            metrics['recall'] = Recall(
                task='multilabel',
                num_labels=self.num_classes,
                average=self.average,
                threshold=self.threshold
            )
            metrics['f1'] = F1Score(
                task='multilabel',
                num_labels=self.num_classes,
                average=self.average,
                threshold=self.threshold
            )
            metrics['auroc'] = AUROC(
                task='multilabel',
                num_labels=self.num_classes,
                average=self.average
            )
            metrics['auprc'] = AveragePrecision(
                task='multilabel',
                num_labels=self.num_classes,
                average=self.average
            )
        
        elif self.task == 'binary':
            metrics['accuracy'] = Accuracy(
                task='binary',
                threshold=self.threshold
            )
            metrics['precision'] = Precision(
                task='binary',
                average=self.average,
                threshold=self.threshold
            )
            metrics['recall'] = Recall(
                task='binary',
                average=self.average,
                threshold=self.threshold
            )
            metrics['f1'] = F1Score(
                task='binary',
                average=self.average,
                threshold=self.threshold
            )
            metrics['auroc'] = AUROC(
                task='binary',
                average=self.average
            )
            metrics['auprc'] = AveragePrecision(
                task='binary',
                average=self.average
            )
        
        return metrics
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions and targets.
        
        Args:
            preds: Predictions (logits or probabilities)
            targets: Target labels
        """
        for metric in self.metrics.values():
            metric.update(preds, targets)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute()
            except Exception as e:
                print(f"Error computing metric {name}: {e}")
                results[name] = torch.tensor(0.0)
        
        return results
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def to(self, device: torch.device):
        """Move metrics to device."""
        for metric in self.metrics.values():
            metric.to(device)


class SegmentationMetrics:
    """Metrics for segmentation tasks."""
    
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        """Initialize segmentation metrics.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in metrics
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Initialize metrics
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict[str, nn.Module]:
        """Initialize segmentation metrics."""
        metrics = {}
        
        # IoU (Intersection over Union)
        metrics['iou'] = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            average='none'
        )
        
        # Dice coefficient
        metrics['dice'] = MulticlassF1Score(
            num_classes=self.num_classes,
            average='none'
        )
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = MulticlassAccuracy(
            num_classes=self.num_classes,
            average='micro'
        )
        
        # Mean IoU
        metrics['mean_iou'] = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            average='macro'
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        )
        
        return metrics
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions and targets.
        
        Args:
            preds: Predicted segmentation masks (N, H, W)
            targets: Target segmentation masks (N, H, W)
        """
        # Convert predictions to class indices
        if preds.dim() == 4:  # (N, C, H, W)
            preds = torch.argmax(preds, dim=1)
        
        # Flatten for metrics computation
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # Remove ignored indices
        if self.ignore_index is not None:
            valid_mask = targets_flat != self.ignore_index
            preds_flat = preds_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]
        
        for metric in self.metrics.values():
            metric.update(preds_flat, targets_flat)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute()
            except Exception as e:
                print(f"Error computing metric {name}: {e}")
                results[name] = torch.tensor(0.0)
        
        return results
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def to(self, device: torch.device):
        """Move metrics to device."""
        for metric in self.metrics.values():
            metric.to(device)


class RegressionMetrics:
    """Metrics for regression tasks."""
    
    def __init__(self):
        """Initialize regression metrics."""
        self.metrics = {
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'r2': R2Score()
        }
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions and targets.
        
        Args:
            preds: Predictions
            targets: Target values
        """
        for metric in self.metrics.values():
            metric.update(preds, targets)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute()
            except Exception as e:
                print(f"Error computing metric {name}: {e}")
                results[name] = torch.tensor(0.0)
        
        return results
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def to(self, device: torch.device):
        """Move metrics to device."""
        for metric in self.metrics.values():
            metric.to(device)


def create_metrics(config: Dict[str, Any]) -> Union[MetricsWrapper, SegmentationMetrics, RegressionMetrics, KaggleCompatibleMetrics]:
    """Create metrics from configuration.
    
    Args:
        config: Metrics configuration
        
    Returns:
        Metrics instance
    """
    # Use fallback implementation if torchmetrics is not available
    if not TORCHMETRICS_AVAILABLE:
        return create_kaggle_metrics(config)
    
    task = config.get('task', 'multiclass')
    num_classes = config.get('num_classes', 10)
    
    if task == 'segmentation':
        return SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=config.get('ignore_index')
        )
    elif task == 'regression':
        return RegressionMetrics()
    else:
        return MetricsWrapper(
            num_classes=num_classes,
            task=task,
            average=config.get('average', 'macro'),
            threshold=config.get('threshold', 0.5)
        )


def compute_classification_metrics(preds: torch.Tensor, 
                                 targets: torch.Tensor,
                                 num_classes: int,
                                 task: str = 'multiclass',
                                 average: str = 'macro') -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        preds: Predictions (logits or probabilities)
        targets: Target labels
        num_classes: Number of classes
        task: Task type
        average: Averaging method
        
    Returns:
        Dictionary of metric values
    """
    metrics = MetricsWrapper(
        num_classes=num_classes,
        task=task,
        average=average
    )
    
    metrics.update(preds, targets)
    results = metrics.compute()
    
    # Convert to float values
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in results.items()}


def compute_segmentation_metrics(preds: torch.Tensor,
                               targets: torch.Tensor,
                               num_classes: int,
                               ignore_index: Optional[int] = None) -> Dict[str, float]:
    """Compute segmentation metrics.
    
    Args:
        preds: Predicted segmentation masks
        targets: Target segmentation masks
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        Dictionary of metric values
    """
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=ignore_index
    )
    
    metrics.update(preds, targets)
    results = metrics.compute()
    
    # Convert to float values
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in results.items()}
