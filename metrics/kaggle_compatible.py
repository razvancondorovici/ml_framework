"""Kaggle-compatible metrics wrapper that avoids torchmetrics conflicts."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class SimpleAccuracy(nn.Module):
    """Simple accuracy metric."""
    
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update accuracy with new predictions and targets."""
        preds = torch.argmax(preds, dim=1) if preds.dim() > 1 else preds
        correct = (preds == target).sum().item()
        total = target.numel()
        
        self.correct += correct
        self.total += total
    
    def compute(self) -> float:
        """Compute accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset(self):
        """Reset metrics."""
        self.correct = 0
        self.total = 0


class SimpleIoU(nn.Module):
    """Simple IoU metric for segmentation."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update IoU with new predictions and targets."""
        # Convert predictions to class indices
        if preds.dim() == 4:  # (B, C, H, W)
            preds = torch.argmax(preds, dim=1)
        
        # Flatten tensors
        preds = preds.flatten()
        target = target.flatten()
        
        # Ignore ignore_index
        valid_mask = target != self.ignore_index
        preds = preds[valid_mask]
        target = target[valid_mask]
        
        # Compute intersection and union for each class
        for cls in range(self.num_classes):
            pred_cls = (preds == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()
            
            self.intersection[cls] += intersection
            self.union[cls] += union
    
    def compute(self) -> Dict[str, float]:
        """Compute IoU metrics."""
        ious = {}
        
        # Per-class IoU
        for cls in range(self.num_classes):
            if self.union[cls] > 0:
                ious[f'iou_class_{cls}'] = self.intersection[cls] / self.union[cls]
            else:
                ious[f'iou_class_{cls}'] = 0.0
        
        # Mean IoU
        valid_classes = self.union > 0
        if valid_classes.sum() > 0:
            ious['mean_iou'] = (self.intersection[valid_classes] / self.union[valid_classes]).mean().item()
        else:
            ious['mean_iou'] = 0.0
        
        return ious
    
    def reset(self):
        """Reset metrics."""
        self.intersection.zero_()
        self.union.zero_()


class SimpleDice(nn.Module):
    """Simple Dice coefficient metric."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes)
        self.pred_sum = torch.zeros(num_classes)
        self.target_sum = torch.zeros(num_classes)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update Dice with new predictions and targets."""
        # Convert predictions to class indices
        if preds.dim() == 4:  # (B, C, H, W)
            preds = torch.argmax(preds, dim=1)
        
        # Flatten tensors
        preds = preds.flatten()
        target = target.flatten()
        
        # Ignore ignore_index
        valid_mask = target != self.ignore_index
        preds = preds[valid_mask]
        target = target[valid_mask]
        
        # Compute metrics for each class
        for cls in range(self.num_classes):
            pred_cls = (preds == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().item()
            pred_sum = pred_cls.sum().item()
            target_sum = target_cls.sum().item()
            
            self.intersection[cls] += intersection
            self.pred_sum[cls] += pred_sum
            self.target_sum[cls] += target_sum
    
    def compute(self) -> Dict[str, float]:
        """Compute Dice metrics."""
        dices = {}
        
        # Per-class Dice
        for cls in range(self.num_classes):
            if self.pred_sum[cls] + self.target_sum[cls] > 0:
                dices[f'dice_class_{cls}'] = 2 * self.intersection[cls] / (self.pred_sum[cls] + self.target_sum[cls])
            else:
                dices[f'dice_class_{cls}'] = 0.0
        
        # Mean Dice
        valid_classes = (self.pred_sum + self.target_sum) > 0
        if valid_classes.sum() > 0:
            mean_dice = 0.0
            for cls in range(self.num_classes):
                if valid_classes[cls]:
                    mean_dice += 2 * self.intersection[cls] / (self.pred_sum[cls] + self.target_sum[cls])
            dices['mean_dice'] = mean_dice / valid_classes.sum().item()
        else:
            dices['mean_dice'] = 0.0
        
        return dices
    
    def reset(self):
        """Reset metrics."""
        self.intersection.zero_()
        self.pred_sum.zero_()
        self.target_sum.zero_()


class SimpleLoss(nn.Module):
    """Simple loss wrapper."""
    
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.total_loss = 0.0
        self.count = 0
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update loss."""
        loss = self.loss_fn(preds, target)
        self.total_loss += loss.item()
        self.count += 1
    
    def compute(self) -> float:
        """Compute average loss."""
        return self.total_loss / self.count if self.count > 0 else 0.0
    
    def reset(self):
        """Reset loss."""
        self.total_loss = 0.0
        self.count = 0


class KaggleCompatibleMetrics:
    """Kaggle-compatible metrics wrapper."""
    
    def __init__(self, task: str = 'segmentation', num_classes: int = 2, ignore_index: int = 255):
        self.task = task
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = {}
        
        if task == 'segmentation':
            self.metrics['iou'] = SimpleIoU(num_classes, ignore_index)
            self.metrics['dice'] = SimpleDice(num_classes, ignore_index)
        elif task == 'classification':
            self.metrics['accuracy'] = SimpleAccuracy()
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, loss: Optional[torch.Tensor] = None):
        """Update all metrics."""
        for metric in self.metrics.values():
            metric.update(preds, target)
        
        if loss is not None:
            if 'loss' not in self.metrics:
                self.metrics['loss'] = SimpleLoss(lambda x, y: loss)
            else:
                self.metrics['loss'].update(preds, target)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, 'compute'):
                if name in ['iou', 'dice']:
                    results.update(metric.compute())
                else:
                    results[name] = metric.compute()
        return results
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()


def create_metrics(config: Dict[str, Any]) -> KaggleCompatibleMetrics:
    """Create metrics from configuration."""
    task = config.get('task', 'segmentation')
    num_classes = config.get('num_classes', 2)
    ignore_index = config.get('ignore_index', 255)
    
    return KaggleCompatibleMetrics(task=task, num_classes=num_classes, ignore_index=ignore_index)
