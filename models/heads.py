"""Custom model heads for classification and segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ClassificationHead(nn.Module):
    """Custom classification head with configurable architecture."""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 use_batch_norm: bool = True):
        """Initialize classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.head(x)


class MultiLabelClassificationHead(nn.Module):
    """Multi-label classification head with sigmoid activation."""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 use_batch_norm: bool = True):
        """Initialize multi-label classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer with sigmoid activation
        layers.append(nn.Linear(prev_dim, num_classes))
        layers.append(nn.Sigmoid())
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.head(x)


class SegmentationHead(nn.Module):
    """Custom segmentation head with upsampling and skip connections."""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: Optional[List[int]] = None,
                 use_skip_connections: bool = True,
                 upsampling_method: str = 'transpose'):
        """Initialize segmentation head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            use_skip_connections: Whether to use skip connections
            upsampling_method: Upsampling method ('transpose' or 'interpolate')
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.use_skip_connections = use_skip_connections
        self.upsampling_method = upsampling_method
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if upsampling_method == 'transpose':
                layer = nn.Sequential(
                    nn.ConvTranspose2d(prev_dim, hidden_dim, 4, 2, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            else:  # interpolate
                layer = nn.Sequential(
                    nn.Conv2d(prev_dim, hidden_dim, 3, 1, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            
            self.decoder_layers.append(layer)
            prev_dim = hidden_dim
        
        # Skip connection layers
        if use_skip_connections:
            self.skip_layers = nn.ModuleList()
            for hidden_dim in hidden_dims:
                self.skip_layers.append(
                    nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
                )
        
        # Final output layer
        self.output_layer = nn.Conv2d(prev_dim, num_classes, 1, 1, 0)
    
    def forward(self, x, skip_features: Optional[List[torch.Tensor]] = None):
        """Forward pass.
        
        Args:
            x: Input features
            skip_features: List of skip connection features
            
        Returns:
            Segmentation output
        """
        for i, layer in enumerate(self.decoder_layers):
            if self.upsampling_method == 'interpolate':
                # Upsample using interpolation
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            x = layer(x)
            
            # Add skip connection if available
            if (self.use_skip_connections and 
                skip_features is not None and 
                i < len(skip_features)):
                skip_feat = self.skip_layers[i](skip_features[i])
                x = x + skip_feat
        
        # Final output
        output = self.output_layer(x)
        
        return output


class FPNHead(nn.Module):
    """Feature Pyramid Network head for segmentation."""
    
    def __init__(self, 
                 in_channels_list: List[int],
                 out_channels: int,
                 num_classes: int,
                 upsampling_method: str = 'interpolate'):
        """Initialize FPN head.
        
        Args:
            in_channels_list: List of input channel dimensions
            out_channels: Output channel dimension
            num_classes: Number of output classes
            upsampling_method: Upsampling method
        """
        super().__init__()
        
        self.upsampling_method = upsampling_method
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            )
        
        # Output convolutions
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(out_channels, num_classes, 1, 1, 0)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: List of feature maps at different scales
            
        Returns:
            Segmentation output
        """
        # Apply lateral connections
        lateral_features = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            lateral_feat = lateral_conv(feat)
            lateral_features.append(lateral_feat)
        
        # Upsample and combine features
        output_features = []
        prev_feat = None
        
        for i, lateral_feat in enumerate(reversed(lateral_features)):
            if prev_feat is not None:
                if self.upsampling_method == 'interpolate':
                    prev_feat = F.interpolate(
                        prev_feat, 
                        size=lateral_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:  # transpose
                    prev_feat = F.conv_transpose2d(
                        prev_feat,
                        weight=torch.ones(prev_feat.size(1), prev_feat.size(1), 2, 2).to(prev_feat.device),
                        stride=2
                    )
                
                lateral_feat = lateral_feat + prev_feat
            
            output_feat = self.output_convs[-(i+1)](lateral_feat)
            output_features.append(output_feat)
            prev_feat = output_feat
        
        # Final output
        final_feat = output_features[-1]
        output = self.final_conv(final_feat)
        
        return output


class AttentionHead(nn.Module):
    """Attention-based classification head."""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 attention_dim: int = 256,
                 dropout: float = 0.0):
        """Initialize attention head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            attention_dim: Attention dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention_dim = attention_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input features (B, L, D) where L is sequence length
            
        Returns:
            Classification output
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # (B, L, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_features = torch.sum(x * attention_weights, dim=1)  # (B, D)
        
        # Classify
        output = self.classifier(attended_features)
        
        return output


def create_classification_head(input_dim: int,
                             num_classes: int,
                             head_type: str = 'linear',
                             **kwargs) -> nn.Module:
    """Create classification head by type.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        head_type: Type of head ('linear', 'mlp', 'attention')
        **kwargs: Additional arguments
        
    Returns:
        Classification head
    """
    if head_type == 'linear':
        return nn.Linear(input_dim, num_classes)
    elif head_type == 'mlp':
        return ClassificationHead(input_dim, num_classes, **kwargs)
    elif head_type == 'attention':
        return AttentionHead(input_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def create_segmentation_head(input_dim: int,
                           num_classes: int,
                           head_type: str = 'simple',
                           **kwargs) -> nn.Module:
    """Create segmentation head by type.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        head_type: Type of head ('simple', 'fpn')
        **kwargs: Additional arguments
        
    Returns:
        Segmentation head
    """
    if head_type == 'simple':
        return SegmentationHead(input_dim, num_classes, **kwargs)
    elif head_type == 'fpn':
        return FPNHead([input_dim], 256, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
