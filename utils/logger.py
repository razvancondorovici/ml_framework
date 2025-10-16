"""Logging utilities for structured logging."""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

try:
    from omegaconf import DictConfig, ListConfig
except ImportError:
    DictConfig = None
    ListConfig = None


class OmegaConfJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles OmegaConf objects."""
    
    def default(self, obj):
        if DictConfig is not None and isinstance(obj, DictConfig):
            return dict(obj)
        elif ListConfig is not None and isinstance(obj, ListConfig):
            return list(obj)
        return super().default(obj)


class StructuredLogger:
    """Structured logger that writes to JSONL and CSV files."""
    
    def __init__(self, log_dir: Union[str, Path], name: str = "training"):
        """Initialize structured logger.
        
        Args:
            log_dir: Directory to save log files
            name: Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # JSONL log file
        self.jsonl_path = self.log_dir / f"{name}.jsonl"
        
        # CSV file for scalar metrics
        self.csv_path = self.log_dir / "scalars.csv"
        self.csv_writer = None
        self.csv_fieldnames = None
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add file handler
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log(self, message: str, level: str = "INFO", **kwargs):
        """Log a message with optional structured data.
        
        Args:
            message: Log message
            level: Log level
            **kwargs: Additional structured data
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        # Write to JSONL
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(log_entry, cls=OmegaConfJSONEncoder) + '\n')
        
        # Write to Python logger
        getattr(self.logger, level.lower())(message)
    
    def log_scalars(self, step: int, epoch: int, scalars: Dict[str, float]):
        """Log scalar metrics to CSV.
        
        Args:
            step: Training step
            epoch: Training epoch
            scalars: Dictionary of scalar metrics
        """
        # Initialize CSV writer if needed
        if self.csv_writer is None:
            fieldnames = ['step', 'epoch'] + sorted(scalars.keys())
            self.csv_fieldnames = fieldnames
            
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        
        # Write row
        row = {'step': step, 'epoch': epoch, **scalars}
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(message, "INFO", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(message, "WARNING", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(message, "ERROR", **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(message, "DEBUG", **kwargs)


def get_logger(name: str, log_dir: Optional[Union[str, Path]] = None) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name
        log_dir: Log directory (optional)
        
    Returns:
        StructuredLogger instance
    """
    if log_dir is None:
        log_dir = Path("logs")
    
    return StructuredLogger(log_dir, name)
