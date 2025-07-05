"""
Utility functions for MinillM.
Common helper functions used across the codebase.
"""

import os
import logging
import sys
from typing import Optional, Union
from pathlib import Path

import torch

from .config import LoggingConfig, ComputeConfig


def setup_device(device_config: Union[str, ComputeConfig]) -> torch.device:
    """
    Setup and return the appropriate device.
    
    Args:
        device_config: Device configuration or device string
        
    Returns:
        PyTorch device object
    """
    if isinstance(device_config, ComputeConfig):
        device_str = device_config.device
    else:
        device_str = device_config
    
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    # Validate device is available
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    elif device.type == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Warning: MPS requested but not available, falling back to CPU")
        device = torch.device("cpu")
    
    return device


def setup_dtype(dtype_config: str, device: torch.device) -> torch.dtype:
    """
    Setup and return the appropriate dtype.
    
    Args:
        dtype_config: Data type configuration
        device: Target device
        
    Returns:
        PyTorch dtype
    """
    if dtype_config == "auto":
        # Use bfloat16 on supported devices, float16 on others, float32 on CPU
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif device.type in ["cuda", "mps"]:
            return torch.float16
        else:
            return torch.float32
    elif dtype_config == "float32":
        return torch.float32
    elif dtype_config == "float16":
        return torch.float16
    elif dtype_config == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_config}")


def setup_logging(logging_config: LoggingConfig) -> None:
    """
    Setup logging configuration.
    
    Args:
        logging_config: Logging configuration
    """
    # Create log directory if it doesn't exist
    log_dir = Path(logging_config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'minillm.log')
        ]
    )
    
    # Setup Weights & Biases if enabled
    if logging_config.use_wandb:
        try:
            import wandb
            wandb.init(project=logging_config.wandb_project)
            print(f"Weights & Biases logging enabled for project: {logging_config.wandb_project}")
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
    
    # Setup TensorBoard if enabled
    if logging_config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = log_dir / 'tensorboard'
            tb_log_dir.mkdir(exist_ok=True)
            writer = SummaryWriter(tb_log_dir)
            print(f"TensorBoard logging enabled at: {tb_log_dir}")
            return writer
        except ImportError:
            print("Warning: tensorboard not installed, skipping TensorBoard logging")
    
    return None


def setup_threading(num_threads: Optional[int] = None) -> None:
    """
    Setup optimal threading configuration.
    
    Args:
        num_threads: Number of threads to use (None for auto-detection)
    """
    if num_threads is None:
        # Auto-detect optimal thread count
        num_threads = torch.get_num_threads()
    
    # Set thread counts for various libraries
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    # Intel CPU optimizations
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    print(f"Threading configured for {num_threads} threads")


def format_model_size(num_params: int) -> str:
    """
    Format model parameter count in human-readable form.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string (e.g., "505M", "1.2B")
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.0f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.0f}K"
    else:
        return str(num_params)


def format_memory_size(size_bytes: int) -> str:
    """
    Format memory size in human-readable form.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.2GB", "512MB")
    """
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f}GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}KB"
    else:
        return f"{size_bytes}B"


def get_model_info(model: torch.nn.Module) -> dict:
    """
    Get comprehensive information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in bytes (assuming float32)
    model_size_bytes = total_params * 4
    
    # Get layer count
    num_layers = len(list(model.modules()))
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_formatted': format_model_size(total_params),
        'trainable_parameters_formatted': format_model_size(trainable_params),
        'model_size_bytes': model_size_bytes,
        'model_size_formatted': format_memory_size(model_size_bytes),
        'num_layers': num_layers,
    }


def safe_save_checkpoint(model: torch.nn.Module, path: str, metadata: Optional[dict] = None) -> None:
    """
    Safely save a model checkpoint with metadata.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
        metadata: Optional metadata to include
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_info': get_model_info(model),
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Save to temporary file first, then rename (atomic operation)
    temp_path = f"{path}.tmp"
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, path)
    
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool = True) -> dict:
    """
    Load a model checkpoint with error handling.
    
    Args:
        model: Model to load checkpoint into
        path: Path to checkpoint file
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Checkpoint metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint, strict=strict)
        
        print(f"Checkpoint loaded from: {path}")
        
        # Return metadata if available
        return checkpoint.get('metadata', {})
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}")


def count_tokens_in_text(text: str, tokenizer) -> int:
    """
    Count tokens in a text string.
    
    Args:
        text: Input text
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    if hasattr(tokenizer, 'encode'):
        return len(tokenizer.encode(text))
    else:
        # Fallback for different tokenizer interfaces
        return len(str(text).split())


def validate_config_paths(config) -> bool:
    """
    Validate that all required paths in config exist.
    
    Args:
        config: Configuration object
        
    Returns:
        True if all paths are valid
    """
    required_paths = [
        config.paths.model_file,
        config.paths.tokenizer_dir,
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Required path does not exist: {path}")
            return False
    
    # Check tokenizer files
    vocab_path = config.paths.vocab_path
    merges_path = config.paths.merges_path
    
    if not os.path.exists(vocab_path):
        print(f"Tokenizer vocab file not found: {vocab_path}")
        return False
        
    if not os.path.exists(merges_path):
        print(f"Tokenizer merges file not found: {merges_path}")
        return False
    
    return True


def create_directory_structure(base_dir: str) -> None:
    """
    Create standard directory structure for MinillM projects.
    
    Args:
        base_dir: Base directory path
    """
    directories = [
        'checkpoints',
        'logs',
        'data',
        'configs',
        'outputs'
    ]
    
    base_path = Path(base_dir)
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created in: {base_dir}")