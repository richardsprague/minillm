"""
Modular transformer layers for MinillM.
Extracted from the original implementation for better maintainability and performance.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis) for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis tensor to be broadcastable with x."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and KV caching."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        
        # Linear projections
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # KV cache buffers (not saved to state_dict)
        self.register_buffer(
            'cache_k', 
            torch.zeros((
                config.max_batch_size,
                config.max_seq_len,
                config.n_kv_heads,
                self.head_dim,
            )),
            persistent=False
        )
        self.register_buffer(
            'cache_v',
            torch.zeros((
                config.max_batch_size,
                config.max_seq_len,
                config.n_kv_heads,
                self.head_dim,
            )),
            persistent=False
        )

    def clear_cache(self) -> None:
        """Clear KV cache."""
        self.cache_k.zero_()
        self.cache_v.zero_()

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with optional KV caching for inference."""
        bsz, seqlen, _ = x.shape
        
        # Compute Q, K, V projections
        queries = self.wq(x)
        keys = self.wk(x) 
        values = self.wv(x)

        # Reshape for multi-head attention
        queries = queries.view(bsz, seqlen, self.config.n_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.config.n_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.config.n_kv_heads, self.head_dim)

        # Apply rotary positional embedding
        queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)

        # KV caching for inference
        if not self.training:
            # Move cache to correct device
            self.cache_k = self.cache_k.to(queries.device)
            self.cache_v = self.cache_v.to(queries.device)
            
            # Update cache
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = keys
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = values
            
            # Retrieve full context (cached + current)
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

        # Grouped Query Attention: repeat KV heads to match Q heads
        keys = torch.repeat_interleave(keys, dim=2, repeats=self.config.n_kv_head_rep)
        values = torch.repeat_interleave(values, dim=2, repeats=self.config.n_kv_head_rep)

        # Rearrange for efficient attention computation
        queries = queries.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Efficient attention computation
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            is_causal=mask is None and self.training,  # Auto-causal mask in training
        )
        
        # Combine heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation: SiLU(W1(x)) * W3(x)."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention with residual connection
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        
        # Feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out