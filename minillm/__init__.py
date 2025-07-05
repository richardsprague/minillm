"""
MinillM: A modular, scalable transformer-based language model implementation.

Based on LLaMA architecture with performance optimizations and production features.
"""

__version__ = "0.1.0"
__author__ = "Nathan Sprague"
__email__ = "nspragu@purdue.edu"

from .config import ModelConfig, TrainingConfig, GenerationConfig

# Conditional imports for torch-dependent modules
__all__ = [
    "ModelConfig",
    "TrainingConfig", 
    "GenerationConfig",
]

try:
    from .models import TransformerModel
    from .tokenizer import TokenizerManager
    __all__.extend(["TransformerModel", "TokenizerManager"])
except ImportError:
    # Torch-dependent modules not available
    pass