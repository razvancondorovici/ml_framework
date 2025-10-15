"""Logging callbacks for training."""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base import Callback
from utils.logger import StructuredLogger


class MetricLogger(Callback):
    """Callback for logging metrics during training."""
    
    def __init__(self, 
                 log_dir: str,
                 log_every_n_epochs: int = 1,
                 log_every_n_steps: int = 100,
                 log_learning_rate: bool = True,
                 log_gradients: bool = False):
        """Initialize metric logger callback.
        
        Args:
            log_dir: Directory to save logs
            log_every_n_epochs: Log every N epochs
            log_every_n_steps: Log every N steps
            log_learning_rate: Whether to log learning rate
            log_gradients: Whether to log gradient norms
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.log_learning_rate = log_learning_rate
        self.log_gradients = log_gradients
        
        # Create logger
        self.logger = StructuredLogger(self.log_dir, "training")
        
        # Track metrics
        self.epoch_metrics = {}
        self.step_metrics = {}
        self.current_epoch = 0
        self.current_step = 0
    
    def on_train_start(self, **kwargs):
        """Initialize logging at start of training."""
        self.logger.info("Training started", **kwargs)
    
    def on_train_end(self, **kwargs):
        """Finalize logging at end of training."""
        self.logger.info("Training completed", **kwargs)
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Log epoch start.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        self.epoch_metrics = {}
        self.logger.info(f"Epoch {epoch} started")
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Log epoch end and metrics.
        
        Args:
            epoch: Current epoch number
        """
        # Get metrics from kwargs
        metrics = kwargs.get('metrics', {})
        
        # Add epoch info
        metrics['epoch'] = epoch
        
        # Add learning rate if requested
        if self.log_learning_rate and self.trainer and self.trainer.optimizer:
            lrs = [group['lr'] for group in self.trainer.optimizer.param_groups]
            metrics['learning_rate'] = lrs[0] if len(lrs) == 1 else lrs
        
        # Add gradient norms if requested
        if self.log_gradients and self.trainer and self.trainer.model:
            grad_norms = self._compute_gradient_norms()
            metrics['gradient_norm'] = grad_norms
        
        # Log metrics
        self.logger.log_scalars(
            step=self.current_step,
            epoch=epoch,
            scalars=metrics
        )
        
        # Log epoch summary
        self.logger.info(f"Epoch {epoch} completed", **metrics)
    
    def on_batch_end(self, batch_idx: int, **kwargs):
        """Log batch metrics.
        
        Args:
            batch_idx: Current batch index
        """
        if batch_idx % self.log_every_n_steps != 0:
            return
        
        # Get metrics from kwargs
        metrics = kwargs.get('metrics', {})
        
        # Add step info
        metrics['step'] = self.current_step
        metrics['batch'] = batch_idx
        
        # Add learning rate if requested
        if self.log_learning_rate and self.trainer and self.trainer.optimizer:
            lrs = [group['lr'] for group in self.trainer.optimizer.param_groups]
            metrics['learning_rate'] = lrs[0] if len(lrs) == 1 else lrs
        
        # Log metrics
        self.logger.log_scalars(
            step=self.current_step,
            epoch=self.current_epoch,
            scalars=metrics
        )
        
        # Log batch summary
        self.logger.info(f"Step {self.current_step} completed", **metrics)
    
    def on_validation_start(self, **kwargs):
        """Log validation start."""
        self.logger.info("Validation started")
    
    def on_validation_end(self, **kwargs):
        """Log validation end and metrics.
        
        Args:
            **kwargs: Validation metrics
        """
        metrics = kwargs.get('metrics', {})
        self.logger.info("Validation completed", **metrics)
    
    def on_optimizer_step(self, **kwargs):
        """Update step counter after optimizer step."""
        self.current_step += 1
    
    def _compute_gradient_norms(self) -> List[float]:
        """Compute gradient norms for all parameters.
        
        Returns:
            List of gradient norms
        """
        if not self.trainer or not self.trainer.model:
            return []
        
        grad_norms = []
        for param in self.trainer.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
        
        return grad_norms


class ProgressLogger(Callback):
    """Callback for logging training progress."""
    
    def __init__(self, 
                 log_every_n_epochs: int = 1,
                 log_every_n_steps: int = 100):
        """Initialize progress logger callback.
        
        Args:
            log_every_n_epochs: Log every N epochs
            log_every_n_steps: Log every N steps
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.current_epoch = 0
        self.current_step = 0
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Log epoch start.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        print(f"Epoch {epoch} started")
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Log epoch end and metrics.
        
        Args:
            epoch: Current epoch number
        """
        if epoch % self.log_every_n_epochs != 0:
            return
        
        # Get metrics from kwargs
        metrics = kwargs.get('metrics', {})
        
        # Format metrics for display
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        
        print(f"Epoch {epoch} completed - {metric_str}")
    
    def on_batch_end(self, batch_idx: int, **kwargs):
        """Log batch progress.
        
        Args:
            batch_idx: Current batch index
        """
        if batch_idx % self.log_every_n_steps != 0:
            return
        
        # Get metrics from kwargs
        metrics = kwargs.get('metrics', {})
        
        # Format metrics for display
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        
        print(f"Step {self.current_step}, Batch {batch_idx} - {metric_str}")
    
    def on_optimizer_step(self, **kwargs):
        """Update step counter after optimizer step."""
        self.current_step += 1


class ModelSummaryLogger(Callback):
    """Callback for logging model summary."""
    
    def __init__(self, log_dir: str):
        """Initialize model summary logger.
        
        Args:
            log_dir: Directory to save model summary
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_start(self, **kwargs):
        """Log model summary at start of training."""
        if not self.trainer or not self.trainer.model:
            return
        
        model = self.trainer.model
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Create summary
        summary = {
            'model_name': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'layers': len(list(model.modules()))
        }
        
        # Save summary
        summary_path = self.log_dir / 'model_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("Model Summary:")
        print(f"  Model: {summary['model_name']}")
        print(f"  Total parameters: {summary['total_parameters']:,}")
        print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"  Frozen parameters: {summary['frozen_parameters']:,}")
        print(f"  Model size: {summary['model_size_mb']:.2f} MB")
        print(f"  Number of layers: {summary['layers']}")
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Log epoch-specific model info.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer or not self.trainer.model:
            return
        
        # Get gradient norms
        grad_norms = []
        for param in self.trainer.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            max_grad_norm = max(grad_norms)
            min_grad_norm = min(grad_norms)
            
            print(f"Epoch {epoch} - Gradient norms: avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f}, min={min_grad_norm:.4f}")


class CheckpointLogger(Callback):
    """Callback for logging checkpoint information."""
    
    def __init__(self, log_dir: str):
        """Initialize checkpoint logger.
        
        Args:
            log_dir: Directory to save checkpoint logs
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = StructuredLogger(self.log_dir, "checkpoints")
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Log checkpoint information at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer:
            return
        
        # Get checkpoint info
        checkpoint_info = {
            'epoch': epoch,
            'model_state_dict_size': len(self.trainer.model.state_dict()),
            'optimizer_state_dict_size': len(self.trainer.optimizer.state_dict()) if self.trainer.optimizer else 0,
            'scheduler_state_dict_size': len(self.trainer.scheduler.state_dict()) if self.trainer.scheduler else 0,
            'scaler_state_dict_size': len(self.trainer.scaler.state_dict()) if self.trainer.scaler else 0
        }
        
        # Log checkpoint info
        self.logger.info(f"Checkpoint saved at epoch {epoch}", **checkpoint_info)
    
    def on_train_end(self, **kwargs):
        """Log final checkpoint information.
        
        Args:
            **kwargs: Final training info
        """
        if not self.trainer:
            return
        
        # Get final info
        final_info = {
            'total_epochs': self.trainer.current_epoch,
            'total_steps': self.trainer.current_step,
            'final_learning_rate': self.trainer.optimizer.param_groups[0]['lr'] if self.trainer.optimizer else 0
        }
        
        # Log final info
        self.logger.info("Training completed", **final_info)
