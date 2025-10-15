"""Model registry for building models from configuration."""

import torch
import torch.nn as nn
import timm
from torchvision import models as tv_models
from typing import Dict, Any, Optional, List, Tuple
import warnings


class ModelRegistry:
    """Registry for model architectures."""
    
    def __init__(self):
        self._models = {}
        self._register_timm_models()
        self._register_torchvision_models()
    
    def _register_timm_models(self):
        """Register timm models."""
        # Popular CNN backbones
        timm_models = [
            'resnet50', 'resnet101', 'resnet152',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
            'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
            'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
            'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',
            'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf',
            'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
            'vit_large_patch16_224', 'vit_huge_patch14_224',
            'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
            'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224',
            'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
            'deit_base_distilled_patch16_224', 'deit_large_patch16_224',
            'deit_large_distilled_patch16_224', 'deit_huge_patch14_224'
        ]
        
        for model_name in timm_models:
            self._models[f'timm_{model_name}'] = self._create_timm_model(model_name)
    
    def _register_torchvision_models(self):
        """Register torchvision models."""
        tv_models_list = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
            'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
            'vgg19', 'vgg19_bn', 'alexnet', 'squeezenet1_0', 'squeezenet1_1',
            'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'
        ]
        
        for model_name in tv_models_list:
            self._models[f'tv_{model_name}'] = self._create_torchvision_model(model_name)
    
    def _create_timm_model(self, model_name: str):
        """Create timm model factory."""
        def factory(num_classes: int, pretrained: bool = True, **kwargs):
            return timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                **kwargs
            )
        return factory
    
    def _create_torchvision_model(self, model_name: str):
        """Create torchvision model factory."""
        def factory(num_classes: int, pretrained: bool = True, **kwargs):
            model_fn = getattr(tv_models, model_name)
            return model_fn(pretrained=pretrained, num_classes=num_classes, **kwargs)
        return factory
    
    def register(self, name: str, model_fn):
        """Register a custom model factory.
        
        Args:
            name: Model name
            model_fn: Model factory function
        """
        self._models[name] = model_fn
    
    def get(self, name: str):
        """Get model factory by name.
        
        Args:
            name: Model name
            
        Returns:
            Model factory function
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._models.keys())}")
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all available models.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())


# Global registry instance
model_registry = ModelRegistry()


def build_classifier(backbone: str,
                    num_classes: int,
                    pretrained: bool = True,
                    freeze_backbone: bool = False,
                    dropout: float = 0.0,
                    **kwargs) -> nn.Module:
    """Build a classification model.
    
    Args:
        backbone: Backbone model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        dropout: Dropout rate for classifier head
        **kwargs: Additional arguments for model creation
        
    Returns:
        Classification model
    """
    # Get model factory
    model_fn = model_registry.get(backbone)
    
    # Create model
    model = model_fn(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
    
    # Add dropout if specified
    if dropout > 0:
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    model.classifier
                )
        elif hasattr(model, 'fc'):
            if isinstance(model.fc, nn.Linear):
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    model.fc
                )
        elif hasattr(model, 'head'):
            if isinstance(model.head, nn.Linear):
                model.head = nn.Sequential(
                    nn.Dropout(dropout),
                    model.head
                )
    
    return model


def build_segmentation_model(backbone: str,
                           num_classes: int,
                           pretrained: bool = True,
                           freeze_backbone: bool = False,
                           **kwargs) -> nn.Module:
    """Build a segmentation model.
    
    Args:
        backbone: Backbone model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        **kwargs: Additional arguments for model creation
        
    Returns:
        Segmentation model
    """
    # For segmentation, we'll use timm's segmentation models
    if backbone.startswith('timm_'):
        backbone_name = backbone[5:]  # Remove 'timm_' prefix
        
        # Check if it's a segmentation model
        if 'deeplabv3' in backbone_name or 'fpn' in backbone_name:
            model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=num_classes,
                **kwargs
            )
        else:
            # Create encoder-decoder architecture
            model = SegmentationModel(
                backbone=backbone_name,
                num_classes=num_classes,
                pretrained=pretrained,
                **kwargs
            )
    else:
        # Use torchvision models with custom decoder
        model = SegmentationModel(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'backbone' in name or 'encoder' in name:
                param.requires_grad = False
    
    return model


def get_parameter_groups(model: nn.Module,
                        backbone_lr: float = 1e-4,
                        head_lr: float = 1e-3,
                        backbone_weight_decay: float = 1e-4,
                        head_weight_decay: float = 1e-4) -> List[Dict[str, Any]]:
    """Get parameter groups for different learning rates.
    
    Args:
        model: Model to get parameters from
        backbone_lr: Learning rate for backbone parameters
        head_lr: Learning rate for head parameters
        backbone_weight_decay: Weight decay for backbone parameters
        head_weight_decay: Weight decay for head parameters
        
    Returns:
        List of parameter groups
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(keyword in name.lower() for keyword in ['backbone', 'encoder', 'features']):
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    param_groups = []
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': backbone_weight_decay,
            'name': 'backbone'
        })
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'weight_decay': head_weight_decay,
            'name': 'head'
        })
    
    return param_groups


class SegmentationModel(nn.Module):
    """Simple segmentation model with encoder-decoder architecture."""
    
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, **kwargs):
        """Initialize segmentation model.
        
        Args:
            backbone: Backbone model name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Create backbone
        if backbone.startswith('timm_'):
            backbone_name = backbone[5:]
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                **kwargs
            )
            # Get feature dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.feature_dims = [f.shape[1] for f in features]
        else:
            # Use torchvision model
            if backbone == 'resnet50':
                self.backbone = tv_models.resnet50(pretrained=pretrained)
                self.feature_dims = [2048, 1024, 512, 256, 64]
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Create decoder
        self.decoder = SegmentationDecoder(
            feature_dims=self.feature_dims,
            num_classes=num_classes
        )
    
    def forward(self, x):
        """Forward pass."""
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        return self.decoder(features)


class SegmentationDecoder(nn.Module):
    """Simple segmentation decoder."""
    
    def __init__(self, feature_dims: List[int], num_classes: int):
        """Initialize decoder.
        
        Args:
            feature_dims: List of feature dimensions from encoder
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Simple decoder with upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dims[0], 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, 4, 2, 1)
        )
    
    def forward(self, features):
        """Forward pass."""
        if isinstance(features, (list, tuple)):
            # Use the last feature map
            x = features[-1]
        else:
            x = features
        
        return self.decoder(x)
