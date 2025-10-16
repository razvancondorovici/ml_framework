"""Loss function registry for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean', ignore_index: int = -100):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted logits (N, C, H, W) for segmentation or (N, C) for classification
            targets: Target labels (N, H, W) for segmentation or (N,) for classification
            
        Returns:
            Focal loss
        """
        # Handle segmentation case where inputs are (N, C, H, W) and targets are (N, H, W)
        if len(inputs.shape) == 4 and len(targets.shape) == 3:
            # Flatten for cross entropy computation
            inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1).permute(0, 2, 1).contiguous().view(-1, inputs.size(1))
            targets_flat = targets.view(-1)
        else:
            # Classification case
            inputs_flat = inputs.view(-1, inputs.size(-1))
            targets_flat = targets.view(-1)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (targets_flat != self.ignore_index)
        
        # If no valid pixels, return zero loss
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Compute cross entropy only on valid pixels
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply valid mask
        focal_loss = focal_loss * valid_mask.float()
        
        if self.reduction == 'mean':
            return focal_loss.sum() / valid_mask.sum()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss.
    
    Reference: https://arxiv.org/abs/1512.00567
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """Initialize Label Smoothing Cross Entropy.
        
        Args:
            smoothing: Label smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Target labels (N,)
            
        Returns:
            Label smoothing loss
        """
        log_preds = F.log_softmax(inputs, dim=1)
        nll_loss = -log_preds.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation.
    
    Reference: https://arxiv.org/abs/1707.03237
    """
    
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean', ignore_index: int = 255):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted probabilities (N, C, H, W)
            targets: Target labels (N, H, W)
            
        Returns:
            Dice loss
        """
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (targets != self.ignore_index)
        
        # Filter out ignore_index pixels
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Apply softmax to inputs
        inputs = F.softmax(inputs, dim=1)
        
        # Get valid targets (clamp to valid range)
        valid_targets = torch.clamp(targets, 0, inputs.size(1) - 1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(valid_targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).expand_as(inputs)
        inputs = inputs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Compute Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        """Initialize Soft Dice Loss.
        
        Args:
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted logits (N, C, H, W)
            targets: Target labels (N, H, W)
            
        Returns:
            Soft Dice loss
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Compute Soft Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class LovaszLoss(nn.Module):
    """Lovász Loss for segmentation.
    
    Reference: https://arxiv.org/abs/1705.08790
    """
    
    def __init__(self, reduction: str = 'mean'):
        """Initialize Lovász Loss.
        
        Args:
            reduction: Reduction method
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted logits (N, C, H, W)
            targets: Target labels (N, H, W)
            
        Returns:
            Lovász loss
        """
        # Flatten inputs and targets
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # Compute Lovász loss for each class
        lovasz_losses = []
        for c in range(inputs.size(1)):
            class_inputs = inputs_flat[:, c, :]
            class_targets = (targets_flat == c).float()
            
            if class_targets.sum() > 0:
                lovasz_loss = self._lovasz_hinge(class_inputs, class_targets)
                lovasz_losses.append(lovasz_loss)
        
        if not lovasz_losses:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        lovasz_loss = torch.stack(lovasz_losses).mean()
        
        if self.reduction == 'mean':
            return lovasz_loss
        elif self.reduction == 'sum':
            return lovasz_loss * len(lovasz_losses)
        else:
            return lovasz_loss
    
    def _lovasz_hinge(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Lovász hinge loss."""
        if len(labels) == 0:
            return logits.sum() * 0
        
        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss
    
    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute Lovász gradient."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        if gts == 0:
            return gt_sorted * 0
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


class TverskyLoss(nn.Module):
    """Tversky Loss for segmentation.
    
    Reference: https://arxiv.org/abs/1706.05721
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5, reduction: str = 'mean'):
        """Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted probabilities (N, C, H, W)
            targets: Target labels (N, H, W)
            
        Returns:
            Tversky loss
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs = F.softmax(inputs, dim=1)
        
        # Compute Tversky coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        fp = (inputs * (1 - targets_one_hot)).sum(dim=(2, 3))
        fn = ((1 - inputs) * targets_one_hot).sum(dim=(2, 3))
        
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = 1 - tversky
        
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss


class CombinedLoss(nn.Module):
    """Combined loss function for segmentation."""
    
    def __init__(self, 
                 ce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_weight: float = 0.0,
                 lovasz_weight: float = 0.0,
                 tversky_weight: float = 0.0,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 **kwargs):
        """Initialize Combined Loss.
        
        Args:
            ce_weight: Weight for Cross Entropy loss
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            lovasz_weight: Weight for Lovász loss
            tversky_weight: Weight for Tversky loss
            tversky_alpha: Alpha parameter for Tversky loss
            tversky_beta: Beta parameter for Tversky loss
            **kwargs: Additional arguments for loss functions
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.tversky_weight = tversky_weight
        
        # Filter out Tversky-specific parameters from kwargs for other losses
        ce_kwargs = {k: v for k, v in kwargs.items() if k not in ['tversky_alpha', 'tversky_beta']}
        
        self.ce_loss = nn.CrossEntropyLoss(**ce_kwargs)
        self.dice_loss = DiceLoss(**ce_kwargs)
        self.focal_loss = FocalLoss(**ce_kwargs) if focal_weight > 0 else None
        self.lovasz_loss = LovaszLoss() if lovasz_weight > 0 else None
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta) if tversky_weight > 0 else None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted logits (N, C, H, W)
            targets: Target labels (N, H, W)
            
        Returns:
            Combined loss
        """
        total_loss = 0.0
        
        # Cross Entropy loss
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(inputs, targets)
            total_loss += self.ce_weight * ce_loss
        
        # Dice loss
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(inputs, targets)
            total_loss += self.dice_weight * dice_loss
        
        # Focal loss
        if self.focal_weight > 0 and self.focal_loss is not None:
            focal_loss = self.focal_loss(inputs, targets)
            total_loss += self.focal_weight * focal_loss
        
        # Lovász loss
        if self.lovasz_weight > 0 and self.lovasz_loss is not None:
            lovasz_loss = self.lovasz_loss(inputs, targets)
            total_loss += self.lovasz_weight * lovasz_loss
        
        # Tversky loss
        if self.tversky_weight > 0 and self.tversky_loss is not None:
            tversky_loss = self.tversky_loss(inputs, targets)
            total_loss += self.tversky_weight * tversky_loss
        
        return total_loss


class LossRegistry:
    """Registry for loss functions."""
    
    def __init__(self):
        self._losses = {
            'cross_entropy': nn.CrossEntropyLoss,
            'bce_with_logits': nn.BCEWithLogitsLoss,
            'bce': nn.BCELoss,
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'smooth_l1': nn.SmoothL1Loss,
            'focal': FocalLoss,
            'label_smoothing': LabelSmoothingCrossEntropy,
            'dice': DiceLoss,
            'soft_dice': SoftDiceLoss,
            'lovasz': LovaszLoss,
            'tversky': TverskyLoss,
            'combined': CombinedLoss
        }
    
    def register(self, name: str, loss_fn):
        """Register a custom loss function.
        
        Args:
            name: Loss function name
            loss_fn: Loss function class or function
        """
        self._losses[name] = loss_fn
    
    def get(self, name: str, **kwargs):
        """Get loss function by name.
        
        Args:
            name: Loss function name
            **kwargs: Arguments for loss function
            
        Returns:
            Loss function instance
        """
        if name not in self._losses:
            raise ValueError(f"Loss '{name}' not found. Available losses: {list(self._losses.keys())}")
        
        loss_fn = self._losses[name]
        return loss_fn(**kwargs)
    
    def list_losses(self) -> List[str]:
        """List all available loss functions.
        
        Returns:
            List of loss function names
        """
        return list(self._losses.keys())


# Global registry instance
loss_registry = LossRegistry()


def create_loss(config: Dict[str, Any]) -> nn.Module:
    """Create loss function from configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function instance
    """
    loss_name = config.get('name', 'cross_entropy')
    loss_kwargs = {k: v for k, v in config.items() if k != 'name'}
    
    return loss_registry.get(loss_name, **loss_kwargs)
