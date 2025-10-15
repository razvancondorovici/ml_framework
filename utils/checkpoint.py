"""Checkpoint utilities for saving and loading model states."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import shutil


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   scaler: Optional[torch.cuda.amp.GradScaler],
                   epoch: int,
                   best_metric: float,
                   config: Dict[str, Any],
                   save_path: Union[str, Path],
                   is_best: bool = False,
                   is_last: bool = False) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        scaler: Gradient scaler state (for AMP)
        epoch: Current epoch
        best_metric: Best metric value achieved
        config: Configuration dictionary
        save_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        is_last: Whether this is the last checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    # Create symlinks for best and last checkpoints
    if is_best:
        best_path = save_path.parent / 'best.pt'
        if best_path.exists():
            best_path.unlink()
        best_path.symlink_to(save_path.name)
    
    if is_last:
        last_path = save_path.parent / 'last.pt'
        if last_path.exists():
            last_path.unlink()
        last_path.symlink_to(save_path.name)


def load_checkpoint(checkpoint_path: Union[str, Path],
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   scaler: Optional[torch.cuda.amp.GradScaler] = None,
                   strict: bool = False) -> Tuple[int, float, Dict[str, Any]]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: Scaler to load state into (optional)
        strict: Whether to strictly enforce key matching
        
    Returns:
        Tuple of (epoch, best_metric, config)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if strict:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_state = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        # Filter out missing keys
        missing_keys = []
        unexpected_keys = []
        
        for key in model_state_dict.keys():
            if key not in model_state:
                missing_keys.append(key)
        
        for key in model_state.keys():
            if key not in model_state_dict:
                unexpected_keys.append(key)
        
        # Load matching keys
        filtered_state = {k: v for k, v in model_state.items() 
                         if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        model_state_dict.update(filtered_state)
        model.load_state_dict(model_state_dict)
        
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    config = checkpoint.get('config', {})
    
    return epoch, best_metric, config


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for last.pt symlink first
    last_path = checkpoint_dir / 'last.pt'
    if last_path.exists():
        return last_path.resolve()
    
    # Look for epoch-based checkpoints
    checkpoint_files = list(checkpoint_dir.glob('epoch_*.pt'))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    def extract_epoch(path):
        try:
            return int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return latest_checkpoint


def find_best_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Find the best checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for best.pt symlink first
    best_path = checkpoint_dir / 'best.pt'
    if best_path.exists():
        return best_path.resolve()
    
    return None


def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path], 
                          keep_last: int = 3) -> None:
    """Clean up old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all epoch-based checkpoints
    checkpoint_files = list(checkpoint_dir.glob('epoch_*.pt'))
    if len(checkpoint_files) <= keep_last:
        return
    
    # Sort by epoch number
    def extract_epoch(path):
        try:
            return int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    sorted_checkpoints = sorted(checkpoint_files, key=extract_epoch, reverse=True)
    
    # Remove old checkpoints
    for checkpoint in sorted_checkpoints[keep_last:]:
        checkpoint.unlink()
        print(f"Removed old checkpoint: {checkpoint}")


def copy_checkpoint(src_path: Union[str, Path], 
                   dst_path: Union[str, Path]) -> None:
    """Copy a checkpoint to a new location.
    
    Args:
        src_path: Source checkpoint path
        dst_path: Destination checkpoint path
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_path}")
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
