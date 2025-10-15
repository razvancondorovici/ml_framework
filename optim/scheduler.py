"""Learning rate scheduler factory and utilities."""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any, List, Optional, Union
from torch.optim.lr_scheduler import _LRScheduler
import math


class SchedulerRegistry:
    """Registry for learning rate schedulers."""
    
    def __init__(self):
        self._schedulers = {
            'step': lr_scheduler.StepLR,
            'multistep': lr_scheduler.MultiStepLR,
            'exponential': lr_scheduler.ExponentialLR,
            'cosine': lr_scheduler.CosineAnnealingLR,
            'cosine_warmup': CosineAnnealingWarmupLR,
            'reduce_on_plateau': lr_scheduler.ReduceLROnPlateau,
            'linear': lr_scheduler.LinearLR,
            'polynomial': lr_scheduler.PolynomialLR,
            'constant': lr_scheduler.ConstantLR,
            'sequential': lr_scheduler.SequentialLR,
            'chained': lr_scheduler.ChainedScheduler
        }
    
    def register(self, name: str, scheduler_fn):
        """Register a custom scheduler.
        
        Args:
            name: Scheduler name
            scheduler_fn: Scheduler class or function
        """
        self._schedulers[name] = scheduler_fn
    
    def get(self, name: str, **kwargs):
        """Get scheduler by name.
        
        Args:
            name: Scheduler name
            **kwargs: Arguments for scheduler
            
        Returns:
            Scheduler instance
        """
        if name not in self._schedulers:
            raise ValueError(f"Scheduler '{name}' not found. Available schedulers: {list(self._schedulers.keys())}")
        
        scheduler_fn = self._schedulers[name]
        return scheduler_fn(**kwargs)
    
    def list_schedulers(self) -> List[str]:
        """List all available schedulers.
        
        Returns:
            List of scheduler names
        """
        return list(self._schedulers.keys())


class CosineAnnealingWarmupLR(_LRScheduler):
    """Cosine annealing with warmup learning rate scheduler.
    
    Reference: https://arxiv.org/abs/1706.02677
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 min_lr: float = 1e-6,
                 last_epoch: int = -1):
        """Initialize cosine annealing with warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch.
        
        Returns:
            List of learning rates
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


class LinearWarmupLR(_LRScheduler):
    """Linear warmup learning rate scheduler."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 5,
                 last_epoch: int = -1):
        """Initialize linear warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch.
        
        Returns:
            List of learning rates
        """
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class PolynomialWarmupLR(_LRScheduler):
    """Polynomial warmup learning rate scheduler."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 5,
                 power: float = 2.0,
                 last_epoch: int = -1):
        """Initialize polynomial warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            power: Power for polynomial warmup
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch.
        
        Returns:
            List of learning rates
        """
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch / self.warmup_epochs) ** self.power
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class OneCycleLR(_LRScheduler):
    """One cycle learning rate scheduler.
    
    Reference: https://arxiv.org/abs/1708.07120
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 max_lr: float,
                 total_epochs: int,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 last_epoch: int = -1):
        """Initialize one cycle scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            max_lr: Maximum learning rate
            total_epochs: Total number of epochs
            pct_start: Percentage of epochs for warmup
            anneal_strategy: Annealing strategy ('cos' or 'linear')
            div_factor: Division factor for initial learning rate
            final_div_factor: Final division factor
            last_epoch: Last epoch index
        """
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Calculate warmup and annealing phases
        self.warmup_epochs = int(total_epochs * pct_start)
        self.anneal_epochs = total_epochs - self.warmup_epochs
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch.
        
        Returns:
            List of learning rates
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            progress = self.last_epoch / self.warmup_epochs
            return [self.max_lr * progress for _ in self.base_lrs]
        else:
            # Annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / self.anneal_epochs
            
            if self.anneal_strategy == 'cos':
                anneal_factor = 0.5 * (1 + math.cos(math.pi * progress))
            else:  # linear
                anneal_factor = 1 - progress
            
            return [self.max_lr * anneal_factor for _ in self.base_lrs]


# Global registry instance
scheduler_registry = SchedulerRegistry()

# Register custom schedulers
scheduler_registry.register('linear_warmup', LinearWarmupLR)
scheduler_registry.register('polynomial_warmup', PolynomialWarmupLR)
scheduler_registry.register('one_cycle', OneCycleLR)


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> _LRScheduler:
    """Create scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Scheduler instance
    """
    scheduler_name = config.get('name', 'cosine_warmup')
    scheduler_kwargs = {k: v for k, v in config.items() if k != 'name'}
    
    # Add optimizer to kwargs
    scheduler_kwargs['optimizer'] = optimizer
    
    return scheduler_registry.get(scheduler_name, **scheduler_kwargs)


def get_scheduler_state_dict(scheduler: _LRScheduler) -> Dict[str, Any]:
    """Get scheduler state dictionary.
    
    Args:
        scheduler: Scheduler to get state from
        
    Returns:
        Scheduler state dictionary
    """
    return scheduler.state_dict()


def load_scheduler_state_dict(scheduler: _LRScheduler, state_dict: Dict[str, Any]):
    """Load scheduler state dictionary.
    
    Args:
        scheduler: Scheduler to load state into
        state_dict: State dictionary to load
    """
    scheduler.load_state_dict(state_dict)


def get_current_lr(optimizer: torch.optim.Optimizer) -> List[float]:
    """Get current learning rates from optimizer.
    
    Args:
        optimizer: Optimizer to get learning rates from
        
    Returns:
        List of current learning rates
    """
    return [group['lr'] for group in optimizer.param_groups]


def print_scheduler_info(scheduler: _LRScheduler):
    """Print scheduler information.
    
    Args:
        scheduler: Scheduler to print info for
    """
    print(f"Scheduler: {type(scheduler).__name__}")
    
    if hasattr(scheduler, 'warmup_epochs'):
        print(f"Warmup epochs: {scheduler.warmup_epochs}")
    
    if hasattr(scheduler, 'total_epochs'):
        print(f"Total epochs: {scheduler.total_epochs}")
    
    if hasattr(scheduler, 'min_lr'):
        print(f"Minimum learning rate: {scheduler.min_lr}")
    
    if hasattr(scheduler, 'max_lr'):
        print(f"Maximum learning rate: {scheduler.max_lr}")


def create_scheduler_from_config(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> _LRScheduler:
    """Create scheduler from configuration with default values.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Scheduler instance
    """
    scheduler_name = config.get('name', 'cosine_warmup')
    
    # Set default values based on scheduler type
    if scheduler_name == 'cosine_warmup':
        default_config = {
            'warmup_epochs': 5,
            'total_epochs': 100,
            'min_lr': 1e-6
        }
    elif scheduler_name == 'step':
        default_config = {
            'step_size': 30,
            'gamma': 0.1
        }
    elif scheduler_name == 'multistep':
        default_config = {
            'milestones': [30, 60, 90],
            'gamma': 0.1
        }
    elif scheduler_name == 'exponential':
        default_config = {
            'gamma': 0.95
        }
    elif scheduler_name == 'cosine':
        default_config = {
            'T_max': 100,
            'eta_min': 1e-6
        }
    elif scheduler_name == 'reduce_on_plateau':
        default_config = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'threshold': 1e-4,
            'threshold_mode': 'rel',
            'cooldown': 0,
            'min_lr': 1e-6,
            'eps': 1e-8
        }
    else:
        default_config = {}
    
    # Merge with provided config
    scheduler_config = {**default_config, **config}
    scheduler_config.pop('name', None)
    scheduler_config['optimizer'] = optimizer
    
    return scheduler_registry.get(scheduler_name, **scheduler_config)
