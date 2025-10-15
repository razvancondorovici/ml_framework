"""Configuration management utilities."""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig


def parse_cli_overrides(overrides: list) -> Dict[str, Any]:
    """Parse command line overrides in key.subkey=value format.
    
    Args:
        overrides: List of override strings
        
    Returns:
        Dictionary of parsed overrides
    """
    parsed = {}
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Expected 'key.subkey=value'")
        
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Convert value to appropriate type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.lower() == 'null':
            value = None
        elif value.replace('.', '').replace('-', '').isdigit():
            value = float(value) if '.' in value else int(value)
        
        # Set nested value
        current = parsed
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return parsed


def load_config(config_path: Union[str, Path], overrides: Optional[list] = None) -> DictConfig:
    """Load and merge configuration from YAML file with CLI overrides.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: List of CLI overrides
        
    Returns:
        Merged configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load base config
    config = OmegaConf.load(config_path)
    
    # Apply overrides if provided
    if overrides:
        override_dict = parse_cli_overrides(overrides)
        override_config = OmegaConf.create(override_dict)
        config = OmegaConf.merge(config, override_config)
    
    return config


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def create_run_folder(experiment_name: str, base_dir: str = "runs") -> Path:
    """Create a unique run folder with timestamp.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for runs
        
    Returns:
        Path to created run folder
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(base_dir) / experiment_name / timestamp
    run_folder.mkdir(parents=True, exist_ok=True)
    
    return run_folder


def setup_experiment(config: DictConfig) -> Path:
    """Setup experiment folder and save resolved config.
    
    Args:
        config: Resolved configuration
        
    Returns:
        Path to experiment folder
    """
    experiment_name = config.get('experiment', {}).get('name', 'unnamed_experiment')
    run_folder = create_run_folder(experiment_name)
    
    # Save resolved config
    config_path = run_folder / 'config.yaml'
    save_config(config, config_path)
    
    # Create subdirectories
    (run_folder / 'checkpoints').mkdir(exist_ok=True)
    (run_folder / 'plots').mkdir(exist_ok=True)
    (run_folder / 'samples').mkdir(exist_ok=True)
    
    return run_folder


def get_config_parser() -> argparse.ArgumentParser:
    """Get argument parser for CLI scripts.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description='PyTorch Training Framework')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--overrides', nargs='*', default=[],
                       help='Configuration overrides in key.subkey=value format')
    
    return parser
