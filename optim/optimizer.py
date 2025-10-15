"""Optimizer factory and utilities."""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union
from torch.optim import Optimizer


class OptimizerRegistry:
    """Registry for optimizers."""
    
    def __init__(self):
        self._optimizers = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adamax': optim.Adamax,
            'rprop': optim.Rprop,
            'adadelta': optim.Adadelta
        }
    
    def register(self, name: str, optimizer_fn):
        """Register a custom optimizer.
        
        Args:
            name: Optimizer name
            optimizer_fn: Optimizer class or function
        """
        self._optimizers[name] = optimizer_fn
    
    def get(self, name: str, **kwargs):
        """Get optimizer by name.
        
        Args:
            name: Optimizer name
            **kwargs: Arguments for optimizer
            
        Returns:
            Optimizer instance
        """
        if name not in self._optimizers:
            raise ValueError(f"Optimizer '{name}' not found. Available optimizers: {list(self._optimizers.keys())}")
        
        optimizer_fn = self._optimizers[name]
        return optimizer_fn(**kwargs)
    
    def list_optimizers(self) -> List[str]:
        """List all available optimizers.
        
        Returns:
            List of optimizer names
        """
        return list(self._optimizers.keys())


# Global registry instance
optimizer_registry = OptimizerRegistry()


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> Optimizer:
    """Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_name = config.get('name', 'adamw')
    optimizer_kwargs = {k: v for k, v in config.items() if k != 'name'}
    
    # Handle parameter groups
    if 'parameter_groups' in config:
        param_groups = config['parameter_groups']
        return optimizer_registry.get(optimizer_name, params=param_groups, **optimizer_kwargs)
    else:
        return optimizer_registry.get(optimizer_name, params=model.parameters(), **optimizer_kwargs)


def create_parameter_groups(model: torch.nn.Module, 
                          backbone_lr: float = 1e-4,
                          head_lr: float = 1e-3,
                          backbone_weight_decay: float = 1e-4,
                          head_weight_decay: float = 1e-4) -> List[Dict[str, Any]]:
    """Create parameter groups for different learning rates.
    
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


def apply_gradient_clipping(optimizer: Optimizer, 
                          max_norm: float = 1.0,
                          norm_type: float = 2.0) -> float:
    """Apply gradient clipping to optimizer.
    
    Args:
        optimizer: Optimizer to clip gradients for
        max_norm: Maximum norm for gradients
        norm_type: Type of norm to use
        
    Returns:
        Total norm of gradients before clipping
    """
    total_norm = 0.0
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1.0 / norm_type)
    
    if total_norm > max_norm:
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(max_norm / total_norm)
    
    return total_norm


def get_optimizer_state_dict(optimizer: Optimizer) -> Dict[str, Any]:
    """Get optimizer state dictionary.
    
    Args:
        optimizer: Optimizer to get state from
        
    Returns:
        Optimizer state dictionary
    """
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer: Optimizer, state_dict: Dict[str, Any]):
    """Load optimizer state dictionary.
    
    Args:
        optimizer: Optimizer to load state into
        state_dict: State dictionary to load
    """
    optimizer.load_state_dict(state_dict)


def get_learning_rate(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer to get learning rate from
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer: Optimizer, lr: float):
    """Set learning rate for optimizer.
    
    Args:
        optimizer: Optimizer to set learning rate for
        lr: Learning rate to set
    """
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """Get parameter count for model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def print_optimizer_info(optimizer: Optimizer):
    """Print optimizer information.
    
    Args:
        optimizer: Optimizer to print info for
    """
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}:")
        print(f"  Learning rate: {group['lr']}")
        print(f"  Weight decay: {group.get('weight_decay', 0)}")
        print(f"  Number of parameters: {sum(p.numel() for p in group['params'])}")
        
        if 'name' in group:
            print(f"  Name: {group['name']}")


def create_optimizer_from_config(model: torch.nn.Module, config: Dict[str, Any]) -> Optimizer:
    """Create optimizer from configuration with parameter groups.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    # Create parameter groups if specified
    if 'parameter_groups' in config:
        param_groups = config['parameter_groups']
    elif 'backbone_lr' in config or 'head_lr' in config:
        param_groups = create_parameter_groups(
            model,
            backbone_lr=config.get('backbone_lr', 1e-4),
            head_lr=config.get('head_lr', 1e-3),
            backbone_weight_decay=config.get('backbone_weight_decay', 1e-4),
            head_weight_decay=config.get('head_weight_decay', 1e-4)
        )
    else:
        param_groups = None
    
    # Create optimizer
    optimizer_name = config.get('name', 'adamw')
    optimizer_kwargs = {k: v for k, v in config.items() 
                       if k not in ['name', 'parameter_groups', 'backbone_lr', 'head_lr', 
                                   'backbone_weight_decay', 'head_weight_decay']}
    
    if param_groups is not None:
        return optimizer_registry.get(optimizer_name, params=param_groups, **optimizer_kwargs)
    else:
        return optimizer_registry.get(optimizer_name, params=model.parameters(), **optimizer_kwargs)
