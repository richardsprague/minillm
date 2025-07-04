# model modified from https://github.com/uygarkurt/Llama-3-PyTorch

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F




@dataclass
class ModelArgs:
    DIM = 512
    FFN_DIM = 1536
    N_LAYERS = 16
    N_HEADS = 16
    N_KV_HEADS = 16
    VOCAB_SIZE = 50000
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
        return out * self.weight # (2, 8, DIM) Values stays the same. We make the tensor grad_fn.

def precompute_freqs_cis(dim, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# GQA With Cache
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.wq = nn.Linear(self.args.DIM, self.args.N_HEADS * self.args.HEAD_DIM, bias=False)
        self.wk = nn.Linear(self.args.DIM, self.args.N_KV_HEADS * self.args.HEAD_DIM, bias=False)
        self.wv = nn.Linear(self.args.DIM, self.args.N_KV_HEADS * self.args.HEAD_DIM, bias=False)
        self.wo = nn.Linear(self.args.N_HEADS * self.args.HEAD_DIM, self.args.DIM, bias=False)

        # Register as persistent=False to exclude from state_dict
        self.register_buffer(
            'cache_k', 
            torch.zeros((
                self.args.MAX_BATCH_SIZE,
                self.args.MAX_SEQ_LEN,
                self.args.N_KV_HEADS,
                self.args.HEAD_DIM,
            )),
            persistent=False
        )
        self.register_buffer(
            'cache_v',
            torch.zeros((
                self.args.MAX_BATCH_SIZE,
                self.args.MAX_SEQ_LEN,
                self.args.N_KV_HEADS,
                self.args.HEAD_DIM,
            )),
            persistent=False
        )

    def clear_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.args.N_HEADS, self.args.HEAD_DIM)
        keys = keys.view(bsz, seqlen, self.args.N_KV_HEADS, self.args.HEAD_DIM)
        values = values.view(bsz, seqlen, self.args.N_KV_HEADS, self.args.HEAD_DIM)

        queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)

        # Only use cache during inference (eval mode)
        if not self.training:
            self.cache_k = self.cache_k.to(queries.device)
            self.cache_v = self.cache_v.to(queries.device)
            
            # Update cache
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = keys
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = values
            
            # Retrieve full context (cached + current)
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        # During training: use current keys/values directly (no cache)
        else:
            keys = keys
            values = values

        # Duplicate KV for GQA
        keys = torch.repeat_interleave(keys, dim=2, repeats=self.args.N_KV_HEAD_REP)
        values = torch.repeat_interleave(values, dim=2, repeats=self.args.N_KV_HEAD_REP)

        # Rearrange for efficient attention
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Efficient attention
        out = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            is_causal=mask is None and self.training,  # Auto-causal mask in training
        )
        
        # Combine heads
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Bias is false. It usually adds overhead to the transformer models.
        self.w1 = nn.Linear(args.DIM, args.FFN_DIM, bias=False)
        self.w3 = nn.Linear(args.DIM, args.FFN_DIM, bias=False)
        self.w2 = nn.Linear(args.FFN_DIM, args.DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) # (2, 8, DIM) = (bsz, seqlen, DIM) - use the SwiGLU activation function (llama3) Table 3.





class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(self.args.DIM, self.args.NORM_EPS)
        self.ffn_norm = RMSNorm(self.args.DIM, self.args.NORM_EPS)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask) # (2, 8, 4096) = (bsz, seqlen, DIM)
        out = h + self.feed_forward(self.ffn_norm(h)) # (2, 8, DIM) = (bsz, seqlen, DIM)
        return out # (2, 8, DIM) = (bsz, seqlen, DIM)
    
class TransformerModel(nn.Module):
    def __init__(self, ntokens, max_seq_len, emsize, nhead, nlayers, ffn_dim=14336, dim=512, batch_size=32, device='cuda'):
        super().__init__()

        self.args = ModelArgs()
        self.args.VOCAB_SIZE = ntokens
        self.args.MAX_SEQ_LEN = max_seq_len
        self.args.FFN_DIM = ffn_dim
        self.args.N_LAYERS = nlayers
        self.args.N_HEADS = nhead
        self.args.MAX_BATCH_SIZE = batch_size
        self.args.DIM = dim

        self.args.N_KV_HEAD_REP = self.args.N_HEADS // self.args.N_KV_HEADS # How many times you repeat KV to match your queries(N_HEADS).
        self.args.HEAD_DIM = self.args.DIM // self.args.N_HEADS # Divide dimension by number of heads to get dimension per head.


        self.tok_embeddings = nn.Embedding(
            self.args.VOCAB_SIZE, self.args.DIM
        )
        
        self.layers = torch.nn.ModuleList()
        for _ in range(self.args.N_LAYERS):
            self.layers.append(TransformerBlock(self.args))

        self.norm = RMSNorm(self.args.DIM, self.args.NORM_EPS)
        self.output = nn.Linear(self.args.DIM, self.args.VOCAB_SIZE, bias=False,)

        self.freqs_cis = precompute_freqs_cis(
            self.args.HEAD_DIM,
            self.args.MAX_SEQ_LEN * 2,
            self.args.ROPE_THETA,
        )

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
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

            if hasattr(layer, 'attention'):
                layer.attention.cache_k = layer.attention.cache_k.detach()
                layer.attention.cache_v = layer.attention.cache_v.detach()

    # @torch.inference_mode()
    def forward(self, tokens, start_pos):       
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens) # (bsz, seqlen, DIM)
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None # When we take the tokens from the cached values (seqlen=1) we don't need any aditional mask.
        if seqlen > 1: # Because of KV Cache, we process only 1 token. However, the first run doesn't have any cache. So it has a seqlen > 1.
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device) # Since this is the first pass, we don't have any KV Cache. So we need a mask. Create (seqlen, seqlen) matrix with float("-inf") values.

            mask = torch.triu(mask, diagonal=1).to(tokens.device) # Take the upper triangle excluding diagonal since it's casual LM.

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask) # (2, 8, 4096) = (bsz, seqlen, DIM)
        h = self.norm(h) # (2, 8, 4096) = (bsz, seqlen, DIM)
        out = self.output(h).float() # (2, 8, 128256) = (bsz, seqlen, VOCAB_SIZE)
        return out # (2, 8, 128256) = (bsz, seqlen, VOCAB_SIZE)

    def loss_calc(self, tokens: torch.Tensor, targets=None, masks=None, pad_token_id=93):
        self.clear_kv_cache()

        # Forward pass through the entire sequence
        logits = self.forward(tokens=tokens, start_pos=0)

        B, T, C = logits.shape
        logits = logits.view(B * T, C) 
        targets = targets.reshape(B * T)

        if masks == None:
            non_pad_mask = (targets != pad_token_id).view(B * T) 
        else:
            non_pad_mask = masks.reshape(B * T) ==1
        
        logits = logits[non_pad_mask]
        targets = targets[non_pad_mask]
        n_tokens = non_pad_mask.sum().item()

        # print(logits.shape, targets.shape)
        loss = F.cross_entropy(logits, targets)

        # print("loss", loss, non_pad_mask.sum().item())
        return loss, n_tokens


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9,
                      repetition_penalty=1.1, stream=False):
        model.eval()
        model.clear_kv_cache()
        
        tokens = torch.tensor([tokenizer.encode(prompt).ids + [1, 7]], dtype=torch.long).to(device)

        # print(prompt, end='', flush=True)
        print(prompt)

        generated = tokens
        # for g in range(generated.shape[1]):
        #     model(generated[:, g:g+1], start_pos=g)
        model(generated[:, 0:-1], start_pos=0)
        thinking = True
        last_token = 0
        last_recent_tokens = []
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(generated[:, -1:], start_pos=generated.shape[1])
                next_token_logits = outputs[0, :] / temperature

                # Apply repetition penalty
                for token_id in set(last_recent_tokens):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

                # # (Optional) prevent repeating last token explicitly
                # next_token_logits[0, last_token] -= 10

                # print("gen", generated[:, -1])
                newline_tokens = [208, 230, 15078, 19]#, 35, 23, 1820]
                if generated[:, -1].item() in newline_tokens and generated[:, -2].item() in newline_tokens: # bug where it generates new line forever
                    for n in newline_tokens:
                        next_token_logits[0, n] -= 1000
                    # print("PREVENTING TOKEN")

                # print("Gen->",generated[:, -1].item(), generated[:, -2].item())
                if generated[:, -1].item() == 8:
                    if thinking:
                        thinking = False
                if not thinking:
                    next_token_logits[0, 8] -= 1000

                next_token_logits = next_token_logits.squeeze()
                filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)

                next_token = torch.multinomial(probabilities, 1)
                # if next_token != last_token:
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                last_recent_tokens.append(next_token.item())
                if len(last_recent_tokens)>100:
                    last_recent_tokens = last_recent_tokens[1::]
                # print("next", next_token)
                last_token = next_token
                if last_token == 2:
                    break

                if stream:
                    print(tokenizer.decode([next_token.item()]), end='', flush=True)


        return tokenizer.decode(generated[0].tolist())


    def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        return logits



    # device = torch.device('cpu')



    emsize = -1  # embedding dimension
    d_hid = 6144  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 24  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 16  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.01  # dropout probability
    ntokens = 50000
    block_size = 4096
    ffn_dim = 4096
    dim = 1024

    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_filename="my_tokenizer_50k_2025/tokenizer_50k_2025-vocab.json",
        merges_filename="my_tokenizer_50k_2025/tokenizer_50k_2025-merges.txt"
    )



    model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=1, device=device).to(device)
    model.eval()

    torch.manual_seed(2)

    model.load_state_dict(torch.load("model_finetuned.pt", weights_only=True, map_location=device))

    prompt = "What is AI?"


    generate_text(model, tokenizer, prompt, stream=True, temperature=0.3, top_p=0.9, max_length=4096, repetition_penalty=1.2)