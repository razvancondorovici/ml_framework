"""Training engine for PyTorch models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import time
from tqdm import tqdm

from utils.device import get_device, move_to_device
from utils.seed import set_seed
from utils.logger import StructuredLogger
from utils.checkpoint import load_checkpoint
from callbacks.base import CallbackList
from callbacks.checkpoint import ModelCheckpoint, EarlyStopping
from callbacks.logging import MetricLogger, ProgressLogger, ModelSummaryLogger
from callbacks.visualization import SampleVisualizer, ConfusionMatrixVisualizer, LearningRateVisualizer
from losses.registry import create_loss
from metrics.wrappers import create_metrics
from optim.optimizer import create_optimizer_from_config
from optim.scheduler import create_scheduler_from_config


class Trainer:
    """Main trainer class for PyTorch models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 callbacks: Optional[List[Callable]] = None):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on
            callbacks: List of callback functions
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up training components
        self._setup_training_components()
        
        # Set up callbacks
        self._setup_callbacks(callbacks)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.should_stop = False
        
        # Initialize logger
        self.logger = StructuredLogger(self.config.get('log_dir', 'logs'), 'training')
    
    def _setup_training_components(self):
        """Set up training components from config."""
        # Loss function
        loss_config = self.config.get('loss', {'name': 'cross_entropy'})
        self.criterion = create_loss(loss_config)
        
        # Optimizer
        optimizer_config = self.config.get('optimizer', {'name': 'adamw', 'lr': 1e-3})
        self.optimizer = create_optimizer_from_config(self.model, optimizer_config)
        
        # Scheduler
        scheduler_config = self.config.get('scheduler', {'name': 'cosine_warmup'})
        self.scheduler = create_scheduler_from_config(self.optimizer, scheduler_config)
        
        # Metrics
        metrics_config = self.config.get('metrics', {'task': 'multiclass', 'num_classes': 10})
        self.metrics = create_metrics(metrics_config)
        self.metrics.to(self.device)
        
        # Mixed precision
        self.use_amp = self.config.get('amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # Gradient clipping
        self.gradient_clip_norm = self.config.get('gradient_clip_norm', 0.0)
    
    def _setup_callbacks(self, callbacks: Optional[List[Callable]]):
        """Set up callbacks."""
        # Default callbacks
        default_callbacks = [
            ModelCheckpoint(
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
                monitor=self.config.get('monitor_metric', 'val_loss'),
                mode=self.config.get('monitor_mode', 'min'),
                save_top_k=self.config.get('save_top_k', 3),
                save_last=True,
                save_best=True
            ),
            EarlyStopping(
                monitor=self.config.get('monitor_metric', 'val_loss'),
                mode=self.config.get('monitor_mode', 'min'),
                patience=self.config.get('patience', 10),
                restore_best_weights=True
            ),
            MetricLogger(
                log_dir=self.config.get('log_dir', 'logs'),
                log_every_n_epochs=1,
                log_every_n_steps=100
            ),
            ProgressLogger(
                log_every_n_epochs=1,
                log_every_n_steps=100
            ),
            ModelSummaryLogger(
                log_dir=self.config.get('log_dir', 'logs')
            )
        ]
        
        # Add custom callbacks
        if callbacks:
            default_callbacks.extend(callbacks)
        
        # Create callback list
        self.callbacks = CallbackList(default_callbacks)
        self.callbacks.set_trainer(self)
    
    def fit(self, 
            epochs: int,
            resume_from_checkpoint: Optional[Union[str, Path]] = None,
            **kwargs) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            **kwargs: Additional arguments
            
        Returns:
            Training history
        """
        # Set random seed
        if 'seed' in self.config:
            set_seed(self.config['seed'])
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Call training start callbacks
        self.callbacks.on_train_start(**kwargs)
        
        try:
            # Training loop
            for epoch in range(self.current_epoch, epochs):
                if self.should_stop:
                    break
                
                self.current_epoch = epoch
                
                # Call epoch start callbacks
                self.callbacks.on_epoch_start(epoch, **kwargs)
                
                # Train epoch
                train_metrics = self._train_epoch()
                
                # Validate epoch
                val_metrics = {}
                if self.val_dataloader:
                    val_metrics = self._validate_epoch()
                
                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                    else:
                        self.scheduler.step()
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Update history
                history['train_loss'].append(train_metrics['train_loss'])
                if 'val_loss' in val_metrics:
                    history['val_loss'].append(val_metrics['val_loss'])
                history['train_metrics'].append(train_metrics)
                history['val_metrics'].append(val_metrics)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
                
                # Call epoch end callbacks
                self.callbacks.on_epoch_end(epoch, metrics=epoch_metrics, **kwargs)
                
                # Check if we should stop
                if self.should_stop:
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Call training end callbacks
            self.callbacks.on_train_end(**kwargs)
        
        return history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Training metrics
        """
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # Progress bar
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)
            
            # Call batch start callbacks
            self.callbacks.on_batch_start(batch_idx, inputs=inputs, targets=targets)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Call backward end callbacks
            self.callbacks.on_backward_end(loss=loss.item())
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Call optimizer step callbacks
                self.callbacks.on_optimizer_step()
                
                self.current_step += 1
            
            # Update metrics
            self.metrics.update(outputs, targets)
            
            # Update loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Call batch end callbacks
            self.callbacks.on_batch_end(batch_idx, loss=loss.item(), **kwargs)
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in epoch_metrics.items()}
        epoch_metrics['train_loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Validation metrics
        """
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        # Call validation start callbacks
        self.callbacks.on_validation_start()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                # Move to device
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Update metrics
                self.metrics.update(outputs, targets)
                
                # Update loss
                total_loss += loss.item()
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in epoch_metrics.items()}
        epoch_metrics['val_loss'] = total_loss / num_batches
        
        # Call validation end callbacks
        self.callbacks.on_validation_end(metrics=epoch_metrics)
        
        return epoch_metrics
    
    def _resume_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        epoch, best_metric, config = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            strict=False
        )
        
        # Update training state
        self.current_epoch = epoch
        self.best_metric = best_metric
        
        self.logger.info(f"Resumed from epoch {epoch}, best metric: {best_metric:.4f}")
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given dataloader.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Update metrics
                self.metrics.update(outputs, targets)
                
                # Update loss
                total_loss += loss.item()
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def predict(self, dataloader: DataLoader) -> List[torch.Tensor]:
        """Make predictions on given dataloader.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            List of predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Predicting"):
                # Move to device
                inputs = move_to_device(inputs, self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # Store predictions
                predictions.append(outputs.cpu())
        
        return predictions
