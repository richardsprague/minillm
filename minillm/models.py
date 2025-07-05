"""
Main transformer model implementation for MinillM.
Refactored for better modularity and performance optimization support.
"""

from typing import Optional
import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import TransformerBlock, RMSNorm, precompute_freqs_cis


class TransformerModel(nn.Module):
    """
    Transformer model based on LLaMA architecture.
    
    Features:
    - Grouped Query Attention (GQA)
    - RoPE positional encoding
    - SwiGLU feed-forward networks
    - RMS normalization
    - KV caching for efficient inference
    - Optional gradient checkpointing
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(
                config.head_dim, 
                config.max_seq_len * 2, 
                config.rope_theta
            ),
            persistent=False
        )
        
        # Performance flags
        self.use_gradient_checkpointing = False
        self.is_compiled = False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def set_gradient_checkpointing(self, use_checkpointing: bool) -> None:
        """Enable/disable gradient checkpointing for memory efficiency."""
        self.use_gradient_checkpointing = use_checkpointing
    
    def clear_kv_cache(self) -> None:
        """Clear KV caches in all attention layers."""
        for layer in self.layers:
            layer.attention.clear_cache()
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            tokens: Input token IDs [batch_size, seq_len]
            start_pos: Starting position for KV cache (inference only)
            mask: Attention mask (optional)
            
        Returns:
            Logits over vocabulary [batch_size, seq_len, vocab_size]
        """
        _bsz, seqlen = tokens.shape
        
        # Token embeddings
        h = self.tok_embeddings(tokens)
        
        # Get RoPE frequencies for this sequence
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        # Forward through transformer layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, start_pos, freqs_cis, mask, use_reentrant=False
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask)
        
        # Final normalization and output projection
        h = self.norm(h)
        output = self.output(h)
        return output
    
    def compile_model(self) -> None:
        """Compile the model for better performance."""
        if not self.is_compiled:
            torch._dynamo.config.cache_size_limit = 64
            self = torch.compile(self, mode="reduce-overhead")
            self.is_compiled = True
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        """
        # First estimate the number of flops we do per iteration.
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @classmethod
    def from_config(cls, config: ModelConfig) -> "TransformerModel":
        """Create model from configuration."""
        return cls(config)
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: ModelConfig) -> "TransformerModel":
        """Load pretrained model from checkpoint."""
        model = cls(config)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Handle potential key mismatches (legacy compatibility)
        model_state_dict = model.state_dict()
        
        # Filter out keys that don't match (like cache buffers)
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        return model


# Legacy compatibility: Use original TransformerModel directly
class LegacyTransformerModel(nn.Module):
    """
    Legacy wrapper that uses the original TransformerModel implementation directly.
    This ensures 100% compatibility with the original checkpoint.
    """
    
    def __init__(
        self, 
        ntokens: int, 
        max_seq_len: int, 
        emsize: int, 
        nhead: int, 
        nlayers: int,
        ffn_dim: int = 1536,
        dim: int = 512,
        batch_size: int = 32,
        device: str = 'cuda'
    ):
        super().__init__()
        
        # Import and use the original model directly
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from transformer_model_llama_june2025 import TransformerModel as OriginalTransformerModel
        
        # Create the original model with exact parameters
        self.model = OriginalTransformerModel(
            ntokens=ntokens,
            max_seq_len=max_seq_len,
            emsize=emsize,
            nhead=nhead,
            nlayers=nlayers,
            ffn_dim=ffn_dim,
            dim=dim,
            batch_size=batch_size,
            device=device
        )
    
    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass with legacy interface."""
        return self.model(tokens, start_pos)
    
    def clear_kv_cache(self) -> None:
        """Clear KV cache."""
        self.model.clear_kv_cache()
    
    def eval(self):
        """Set to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def to(self, device):
        """Move to device."""
        self.model.to(device)
        return self
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict."""
        return self.model.load_state_dict(state_dict, strict)
    
    def state_dict(self):
        """Get state dict."""
        return self.model.state_dict()
    
    def parameters(self):
        """Get parameters."""
        return self.model.parameters()