from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelArgs:
    DIM = 512
    FFN_DIM = 1536
    N_LAYERS = 16
    N_HEADS = 16
    N_KV_HEADS = 16
    VOCAB_SIZE = 12825
    NORM_EPS = 1e-5
    ROPE_THETA = 50000
    MAX_BATCH_SIZE = 4
    MAX_SEQ_LEN = 128

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, norm_eps):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wq = nn.Linear(args.DIM, args.N_HEADS * args.HEAD_DIM, bias=False)
        self.wk = nn.Linear(args.DIM, args.N_KV_HEADS * args.HEAD_DIM, bias=False)
        self.wv = nn.Linear(args.DIM, args.N_KV_HEADS * args.HEAD_DIM, bias=False)
        self.wo = nn.Linear(args.N_HEADS * args.HEAD_DIM, args.DIM, bias=False)

        self.register_buffer("cache_k", torch.zeros(args.MAX_BATCH_SIZE, args.MAX_SEQ_LEN, args.N_KV_HEADS, args.HEAD_DIM), persistent=False)
        self.register_buffer("cache_v", torch.zeros(args.MAX_BATCH_SIZE, args.MAX_SEQ_LEN, args.N_KV_HEADS, args.HEAD_DIM), persistent=False)

    def clear_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.args.N_HEADS, self.args.HEAD_DIM)
        k = self.wk(x).view(bsz, seqlen, self.args.N_KV_HEADS, self.args.HEAD_DIM)
        v = self.wv(x).view(bsz, seqlen, self.args.N_KV_HEADS, self.args.HEAD_DIM)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if not self.training:
            self.cache_k[:bsz, start_pos:start_pos+seqlen] = k
            self.cache_v[:bsz, start_pos:start_pos+seqlen] = v
            k = self.cache_k[:bsz, :start_pos+seqlen]
            v = self.cache_v[:bsz, :start_pos+seqlen]

        k = torch.repeat_interleave(k, dim=2, repeats=self.args.N_KV_HEAD_REP)
        v = torch.repeat_interleave(v, dim=2, repeats=self.args.N_KV_HEAD_REP)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and self.training)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.DIM, args.FFN_DIM, bias=False)
        self.w3 = nn.Linear(args.DIM, args.FFN_DIM, bias=False)
        self.w2 = nn.Linear(args.FFN_DIM, args.DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.DIM, args.NORM_EPS)
        self.ffn_norm = RMSNorm(args.DIM, args.NORM_EPS)

    def forward_fn(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward(self, x, start_pos, freqs_cis, mask):
        return self.forward_fn(x, start_pos, freqs_cis, mask)

class TransformerModel(nn.Module):
    def __init__(self, ntokens, max_seq_len, emsize, nhead, nlayers, ffn_dim=1536, dim=512, batch_size=4, device='cuda'):
        super().__init__()
        self.args = ModelArgs()
        self.args.VOCAB_SIZE = ntokens
        self.args.MAX_SEQ_LEN = max_seq_len
        self.args.FFN_DIM = ffn_dim
        self.args.N_LAYERS = nlayers
        self.args.N_HEADS = nhead
        self.args.DIM = dim
        self.args.MAX_BATCH_SIZE = batch_size
        self.args.N_KV_HEAD_REP = self.args.N_HEADS // self.args.N_KV_HEADS
        self.args.HEAD_DIM = self.args.DIM // self.args.N_HEADS

        self.use_checkpointing = True
        self.tok_embeddings = nn.Embedding(self.args.VOCAB_SIZE, self.args.DIM)
        self.layers = nn.ModuleList([TransformerBlock(self.args) for _ in range(self.args.N_LAYERS)])
        self.norm = RMSNorm(self.args.DIM, self.args.NORM_EPS)
        self.output = nn.Linear(self.args.DIM, self.args.VOCAB_SIZE, bias=False)
        self.freqs_cis = precompute_freqs_cis(self.args.HEAD_DIM, self.args.MAX_SEQ_LEN * 2, self.args.ROPE_THETA)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def clear_kv_cache(self):
        for layer in self.layers:
            layer.attention.clear_cache()

    def forward(self, tokens, start_pos):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            if self.use_checkpointing and self.training:
                h = checkpoint(layer.forward_fn, h, start_pos, freqs_cis, mask)
            else:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        out = self.output(h).float()
        return out

    def loss_calc(self, tokens: torch.Tensor, targets=None, masks=None, pad_token_id=93):
        self.clear_kv_cache()
        logits = self.forward(tokens=tokens, start_pos=0)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.reshape(B * T)
        non_pad_mask = (targets != pad_token_id).view(B * T) if masks is None else masks.reshape(B * T) == 1
        logits = logits[non_pad_mask]
        targets = targets[non_pad_mask]
        n_tokens = non_pad_mask.sum().item()
        loss = F.cross_entropy(logits, targets)
        return loss, n_tokens
