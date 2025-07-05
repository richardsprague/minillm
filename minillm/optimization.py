"""
Performance optimization utilities for MinillM.
Implements quantization, compilation, and other performance enhancements.
"""

import warnings
from typing import Optional, Union
import torch
import torch.nn as nn

from .models import TransformerModel


def quantize_model(
    model: TransformerModel, 
    bits: int = 8,
    device: Optional[torch.device] = None
) -> TransformerModel:
    """
    Quantize model weights for memory efficiency and speed.
    
    Args:
        model: Model to quantize
        bits: Quantization bits (4 or 8)
        device: Target device
        
    Returns:
        Quantized model
    """
    if bits not in [4, 8]:
        raise ValueError("Only 4-bit and 8-bit quantization are supported")
    
    try:
        if bits == 8:
            return _quantize_int8(model, device)
        else:
            return _quantize_int4(model, device)
    except ImportError as e:
        warnings.warn(f"Quantization dependencies not available: {e}")
        return model


def _quantize_int8(model: TransformerModel, device: Optional[torch.device] = None) -> TransformerModel:
    """Apply INT8 dynamic quantization."""
    # Use PyTorch's built-in quantization for CPU
    if device is None or device.type == 'cpu':
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    else:
        # For GPU, we'd need bitsandbytes
        try:
            import bitsandbytes as bnb
            # This would require converting linear layers to bnb.nn.Linear8bitLt
            warnings.warn("GPU INT8 quantization requires bitsandbytes integration")
            return model
        except ImportError:
            warnings.warn("bitsandbytes not available for GPU quantization")
            return model


def _quantize_int4(model: TransformerModel, device: Optional[torch.device] = None) -> TransformerModel:
    """Apply INT4 quantization using bitsandbytes."""
    try:
        import bitsandbytes as bnb
        # This would require converting to 4-bit linear layers
        warnings.warn("4-bit quantization requires bitsandbytes integration")
        return model
    except ImportError:
        warnings.warn("bitsandbytes not available for 4-bit quantization")
        return model


def enable_tf32():
    """Enable TensorFloat-32 for faster training on Ampere GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def setup_intel_optimization(model: TransformerModel) -> TransformerModel:
    """
    Apply Intel Extension for PyTorch optimizations.
    
    Args:
        model: Model to optimize
        
    Returns:
        Optimized model
    """
    try:
        import intel_extension_for_pytorch as ipex
        
        # Optimize model for inference
        model = ipex.optimize(
            model.eval(), 
            dtype=torch.bfloat16, 
            inplace=True
        )
        
        print("Applied Intel PyTorch Extension optimizations")
        return model
        
    except ImportError:
        warnings.warn("Intel Extension for PyTorch not available")
        return model


def compile_model(model: TransformerModel, mode: str = "reduce-overhead") -> TransformerModel:
    """
    Compile model using torch.compile for better performance.
    
    Args:
        model: Model to compile
        mode: Compilation mode ('reduce-overhead', 'max-autotune', etc.)
        
    Returns:
        Compiled model
    """
    try:
        # Set cache size limit to avoid memory issues
        torch._dynamo.config.cache_size_limit = 64
        
        # Compile the model
        compiled_model = torch.compile(model, mode=mode)
        
        print(f"Model compiled with mode: {mode}")
        return compiled_model
        
    except Exception as e:
        warnings.warn(f"Model compilation failed: {e}")
        return model


def setup_memory_efficient_attention():
    """Setup memory efficient attention if available."""
    try:
        # Check if FlashAttention is available
        import flash_attn
        print("FlashAttention available - consider enabling in config")
        return True
    except ImportError:
        # Check if PyTorch 2.0+ SDPA is available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("Using PyTorch 2.0+ scaled_dot_product_attention")
            return True
        else:
            warnings.warn("No memory efficient attention available")
            return False


def optimize_for_inference(
    model: TransformerModel,
    quantize: bool = False,
    quantization_bits: int = 8,
    compile_model_flag: bool = True,
    use_intel_extension: bool = False,
    device: Optional[torch.device] = None
) -> TransformerModel:
    """
    Apply comprehensive optimizations for inference.
    
    Args:
        model: Model to optimize
        quantize: Whether to apply quantization
        quantization_bits: Bits for quantization (4 or 8)
        compile_model_flag: Whether to compile the model
        use_intel_extension: Whether to use Intel PyTorch Extension
        device: Target device
        
    Returns:
        Optimized model
    """
    print("Applying inference optimizations...")
    
    # Set to evaluation mode
    model.eval()
    
    # Enable TF32 for faster computation on supported GPUs
    enable_tf32()
    
    # Apply quantization
    if quantize:
        print(f"Applying {quantization_bits}-bit quantization...")
        model = quantize_model(model, quantization_bits, device)
    
    # Apply Intel optimizations (CPU only)
    if use_intel_extension and (device is None or device.type == 'cpu'):
        model = setup_intel_optimization(model)
    
    # Compile model (do this last, after other optimizations)
    if compile_model_flag:
        print("Compiling model...")
        model = compile_model(model)
    
    # Setup memory efficient attention
    setup_memory_efficient_attention()
    
    print("Inference optimizations complete")
    return model


def profile_model(model: TransformerModel, input_tokens: torch.Tensor, num_steps: int = 10):
    """
    Profile model performance.
    
    Args:
        model: Model to profile
        input_tokens: Sample input tokens
        num_steps: Number of profiling steps
    """
    print("Profiling model performance...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tokens)
    
    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(num_steps):
                _ = model(input_tokens)
    
    # Print results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Save trace if needed
    # prof.export_chrome_trace("trace.json")


def estimate_memory_usage(model: TransformerModel, seq_len: int = 128, batch_size: int = 1) -> dict:
    """
    Estimate memory usage for model inference.
    
    Args:
        model: Model to analyze
        seq_len: Sequence length
        batch_size: Batch size
        
    Returns:
        Dictionary with memory estimates
    """
    # Calculate parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Estimate activation memory (rough approximation)
    config = model.config
    hidden_size = config.dim
    
    # Attention activations
    attention_memory = (
        batch_size * seq_len * hidden_size * 4 *  # Q, K, V, O
        config.n_layers * 4  # float32 bytes
    )
    
    # FFN activations
    ffn_memory = (
        batch_size * seq_len * config.ffn_dim * 2 *  # w1, w3 outputs
        config.n_layers * 4  # float32 bytes
    )
    
    # KV cache memory
    kv_cache_memory = (
        batch_size * seq_len * config.n_kv_heads * config.head_dim * 2 *  # K and V
        config.n_layers * 4  # float32 bytes
    )
    
    total_memory = param_memory + attention_memory + ffn_memory + kv_cache_memory
    
    return {
        'parameter_memory_mb': param_memory / (1024 * 1024),
        'attention_memory_mb': attention_memory / (1024 * 1024),
        'ffn_memory_mb': ffn_memory / (1024 * 1024),
        'kv_cache_memory_mb': kv_cache_memory / (1024 * 1024),
        'total_memory_mb': total_memory / (1024 * 1024),
    }