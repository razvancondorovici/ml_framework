"""Device management utilities."""

import torch
from typing import Union, Optional


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. If None, auto-detect best available device.
        
    Returns:
        torch.device object
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    return device


def move_to_device(data, device: torch.device):
    """Move data to specified device.
    
    Args:
        data: Data to move (tensor, dict, list, tuple)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        info['cuda_devices'] = [
            {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i)
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info
