"""Checkpoint callback for saving model states."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base import Callback
from utils.checkpoint import save_checkpoint, cleanup_old_checkpoints


class ModelCheckpoint(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(self,
                 checkpoint_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_top_k: int = 3,
                 save_last: bool = True,
                 save_best: bool = True,
                 filename: str = 'epoch_{epoch:03d}_{monitor:.4f}.pt',
                 save_optimizer: bool = True,
                 save_scheduler: bool = True,
                 save_scaler: bool = True):
        """Initialize model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for monitoring
            save_top_k: Number of best models to keep
            save_last: Whether to save last model
            save_best: Whether to save best model
            filename: Filename template for checkpoints
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            save_scaler: Whether to save scaler state
        """
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_best = save_best
        self.filename = filename
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_scaler = save_scaler
        
        # Track best models
        self.best_models = []
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.last_epoch = 0
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Save checkpoint at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer:
            return
        
        # Get current metrics
        metrics = kwargs.get('metrics', {})
        current_score = metrics.get(self.monitor, float('inf') if self.mode == 'min' else float('-inf'))
        
        # Check if this is the best model
        is_best = False
        if self.mode == 'min':
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score
        
        if is_best:
            self.best_score = current_score
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / self.filename.format(
            epoch=epoch,
            monitor=current_score
        )
        
        # Get model, optimizer, scheduler, scaler
        model = self.trainer.model
        optimizer = self.trainer.optimizer if self.save_optimizer else None
        scheduler = self.trainer.scheduler if self.save_scheduler else None
        scaler = self.trainer.scaler if self.save_scaler else None
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=self.best_score,
            config=self.trainer.config,
            save_path=checkpoint_path,
            is_best=is_best,
            is_last=True
        )
        
        # Update best models list
        if is_best:
            self.best_models.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'score': current_score
            })
            
            # Keep only top-k models
            if len(self.best_models) > self.save_top_k:
                # Remove worst model
                if self.mode == 'min':
                    worst_idx = max(range(len(self.best_models)), key=lambda i: self.best_models[i]['score'])
                else:
                    worst_idx = min(range(len(self.best_models)), key=lambda i: self.best_models[i]['score'])
                
                worst_model = self.best_models.pop(worst_idx)
                try:
                    worst_model['path'].unlink()  # Delete file
                except FileNotFoundError:
                    # File already deleted by another process or cleanup function
                    pass
        
        # Note: We don't call cleanup_old_checkpoints here because ModelCheckpoint
        # already manages the top-k models based on performance above
        
        self.last_epoch = epoch
    
    def get_best_model_path(self) -> Optional[Path]:
        """Get path to best model.
        
        Returns:
            Path to best model or None
        """
        if not self.best_models:
            return None
        
        if self.mode == 'min':
            best_model = min(self.best_models, key=lambda x: x['score'])
        else:
            best_model = max(self.best_models, key=lambda x: x['score'])
        
        return best_model['path']
    
    def get_last_model_path(self) -> Optional[Path]:
        """Get path to last model.
        
        Returns:
            Path to last model or None
        """
        last_path = self.checkpoint_dir / 'last.pt'
        return last_path if last_path.exists() else None


class EarlyStopping(Callback):
    """Early stopping callback to stop training when metric stops improving."""
    
    def __init__(self,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 patience: int = 10,
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        """Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max' for monitoring
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        # Track best score and patience
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Check for early stopping at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer:
            return
        
        # Get current metrics
        metrics = kwargs.get('metrics', {})
        current_score = metrics.get(self.monitor, float('inf') if self.mode == 'min' else float('-inf'))
        
        # Check if score improved
        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = current_score
            self.patience_counter = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = self.trainer.model.state_dict().copy()
        else:
            self.patience_counter += 1
        
        # Check if we should stop
        if self.patience_counter >= self.patience:
            self.stopped_epoch = epoch
            self.trainer.should_stop = True
            
            # Restore best weights
            if self.restore_best_weights and self.best_weights is not None:
                self.trainer.model.load_state_dict(self.best_weights)
            
            print(f"Early stopping triggered at epoch {epoch}. Best {self.monitor}: {self.best_score:.4f}")
    
    def on_train_end(self, **kwargs):
        """Called at end of training."""
        if self.stopped_epoch > 0:
            print(f"Training stopped early at epoch {self.stopped_epoch}")


class LearningRateMonitor(Callback):
    """Callback to monitor learning rate changes."""
    
    def __init__(self, log_momentum: bool = False):
        """Initialize learning rate monitor.
        
        Args:
            log_momentum: Whether to log momentum values
        """
        super().__init__()
        self.log_momentum = log_momentum
        self.lr_history = []
        self.momentum_history = []
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Log learning rate at start of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if not self.trainer or not self.trainer.optimizer:
            return
        
        # Get current learning rates
        lrs = [group['lr'] for group in self.trainer.optimizer.param_groups]
        self.lr_history.append(lrs)
        
        # Log momentum if requested
        if self.log_momentum:
            momentums = []
            for group in self.trainer.optimizer.param_groups:
                if 'momentum' in group:
                    momentums.append(group['momentum'])
                elif 'betas' in group:
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0.0)
            self.momentum_history.append(momentums)
        
        # Log to trainer logger
        if hasattr(self.trainer, 'logger'):
            self.trainer.logger.info(f"Epoch {epoch} - Learning rates: {lrs}")
    
    def get_lr_history(self) -> List[List[float]]:
        """Get learning rate history.
        
        Returns:
            List of learning rates per epoch
        """
        return self.lr_history
    
    def get_momentum_history(self) -> List[List[float]]:
        """Get momentum history.
        
        Returns:
            List of momentum values per epoch
        """
        return self.momentum_history
