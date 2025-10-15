"""Base callback classes for training."""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class Callback(ABC):
    """Base callback class."""
    
    def __init__(self):
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set trainer reference.
        
        Args:
            trainer: Trainer instance
        """
        self.trainer = trainer
    
    def on_train_start(self, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Called at the start of each epoch.
        
        Args:
            epoch: Current epoch number
        """
        pass
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
        """
        pass
    
    def on_batch_start(self, batch_idx: int, **kwargs):
        """Called at the start of each batch.
        
        Args:
            batch_idx: Current batch index
        """
        pass
    
    def on_batch_end(self, batch_idx: int, **kwargs):
        """Called at the end of each batch.
        
        Args:
            batch_idx: Current batch index
        """
        pass
    
    def on_validation_start(self, **kwargs):
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, **kwargs):
        """Called at the end of validation."""
        pass
    
    def on_backward_end(self, **kwargs):
        """Called after backward pass."""
        pass
    
    def on_optimizer_step(self, **kwargs):
        """Called after optimizer step."""
        pass


class CallbackList:
    """List of callbacks with batch operations."""
    
    def __init__(self, callbacks: List[Callback]):
        """Initialize callback list.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks
    
    def set_trainer(self, trainer):
        """Set trainer for all callbacks.
        
        Args:
            trainer: Trainer instance
        """
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_start(self, **kwargs):
        """Call on_train_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_start(**kwargs)
    
    def on_train_end(self, **kwargs):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Call on_epoch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, **kwargs):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, **kwargs)
    
    def on_batch_start(self, batch_idx: int, **kwargs):
        """Call on_batch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_start(batch_idx, **kwargs)
    
    def on_batch_end(self, batch_idx: int, **kwargs):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, **kwargs)
    
    def on_validation_start(self, **kwargs):
        """Call on_validation_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_start(**kwargs)
    
    def on_validation_end(self, **kwargs):
        """Call on_validation_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(**kwargs)
    
    def on_backward_end(self, **kwargs):
        """Call on_backward_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_backward_end(**kwargs)
    
    def on_optimizer_step(self, **kwargs):
        """Call on_optimizer_step for all callbacks."""
        for callback in self.callbacks:
            callback.on_optimizer_step(**kwargs)
    
    def append(self, callback: Callback):
        """Add callback to list.
        
        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)
    
    def extend(self, callbacks: List[Callback]):
        """Add multiple callbacks to list.
        
        Args:
            callbacks: List of callbacks to add
        """
        self.callbacks.extend(callbacks)
    
    def __len__(self):
        """Get number of callbacks."""
        return len(self.callbacks)
    
    def __getitem__(self, index):
        """Get callback by index."""
        return self.callbacks[index]
    
    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)
