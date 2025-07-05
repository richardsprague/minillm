"""
MinillM: A modular, scalable transformer-based language model implementation.

Based on LLaMA architecture with performance optimizations and production features.
"""

__version__ = "0.1.0"
__author__ = "Richard Sprague"
__email__ = "richard@richardsprague.com"

from .config import ModelConfig, TrainingConfig, GenerationConfig
from .models import TransformerModel
from .tokenizer import TokenizerManager

__all__ = [
    "ModelConfig",
    "TrainingConfig", 
    "GenerationConfig",
    "TransformerModel",
    "TokenizerManager",
]