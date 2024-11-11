## LLaMA - Large Language Model with Attention

import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the keys and values. If None, defaults to n_heads
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None # If None, defaults to 4.0
    norm_eps: float = 1e-5
    
    # Needed for KV cache
    max_batch_size: int = 64
    max_seq_len: int = 2048

    device: str = None

def precomputed_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paper, the dimentions o the embedding must be even
    assert head_dim % 2 == 0, "The head_dim must be even"
    # Built the theta parameters
    # According to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,3,..dim/2]
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape : (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the "m" parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using the outer product
    # shape : (seq_len) outer_product * (head_dim / 2) -> (seq_len, head_dim / 2)
    freq = torch.outer(m, theta).float()
    # we can computer complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follow
    # shape: (seq_len, head_dim/2) -> (seq-len, head_dim/2)
    freq_complex = torch.polar(torch.ones_like(freq), freq)
    return freq_complex

def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
    # We transform the each subsequent pair of tokens into a pair of complex numbers
    # shape : (B, seq_len, head_dim) -> (B, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # shape : (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    # shape : (B, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (B, seq_len, h, head_dim / 2)
    x_rotate = x_complex * freq_complex
    # (B, seq_len, h, head_dim / 2) -> (B, seq_len, h, head_dim/2 ,2)
    x_out = torch.view_as_real(x_rotate)
    # (B, seq_len, h, head_dim/2, 2) -> (B, seq_len, h * head_dim / 2 * 2)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int)-> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, mask: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :seq_len, :start_pos + seq_len] == 0, float('-inf'))

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

    def clear_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Assuming 'hidden_dim' is calculated as per your specifications
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)  # Applying your specific transformation
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        #hidden_dim = int(2 * hidden_dim / 3)  # Applying your specific transformation
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # This layer seems to be missing in your original setup
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # Corrected to match checkpoint

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))  # Apply first transformation
        x_V = self.w3(x) 
        x = swish * x_V        # Apply contraction to original dimension
        x = self.w2(x)  # Apply optional additional transformation
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalize BEFORE the self-attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # normalize BEFORE the feed-forward
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, mask: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim) -> (B, seq_len, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # dim : (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class TransformerModel(nn.Module):
    
    def __init__(self, vocab_size, block_size, n_embed, n_heads, n_layers, batch_size=32, device="cuda"):
        super().__init__()

        self.args = ModelArgs()

        self.args.vocab_size = vocab_size
        self.args.max_seq_len = block_size
        self.args.dim = n_embed
        self.args.n_layers = n_layers
        self.args.device = device
        self.args.max_batch_size = batch_size
        self.args.n_heads = n_heads


        assert self.args.vocab_size != -1, "Vocab size must be set"

        self.vocab_size = self.args.vocab_size
        self.n_layers = self.args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.args.n_layers):
            self.layers.append(EncoderBlock(self.args))

        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.output = nn.Linear(self.args.dim, self.vocab_size, bias=False)

        # To precompute the frequencies of the Rotary Positional Encodings
        self.freqs_complex = precomputed_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=h.device)).unsqueeze(0).unsqueeze(0)

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex, mask)
        
        h = self.norm(h)
        output = self.output(h).float()

        return output



    def clear_kv_cache(self):
        for layer in self.layers:
            layer.attention.clear_cache()

            if hasattr(layer, 'attention'):
                layer.attention.cache_k = layer.attention.cache_k.detach()
                layer.attention.cache_v = layer.attention.cache_v.detach()

    def loss_calc(self, tokens: torch.Tensor, targets=None, pad_token_id=93, query_token_id=94, answer_token_id=95):
        self.clear_kv_cache()
        batch_size, seq_len = tokens.shape

        # Forward pass through the entire sequence
        logits = self.forward(tokens=tokens, start_pos=0)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.reshape(B * T)

        non_pad_mask = (targets != pad_token_id).view(B * T)

        if False: # just pad tokens
            # Create a mask for non-pad tokens
            logits = logits[non_pad_mask]
            targets = targets[non_pad_mask]
            n_tokens = non_pad_mask.sum().item()

        else: # only between query and answer tokens, like "<query> blah blah? <answer> use this loss blah. <query> blah dont use blah? <answer>"
            # s = [[1,1,5,4,4,6,1,1,1,5,7,7,7,6,1], [1,1,5,8,8,6,1,1,1,5,9,9,9,6,1]]
            x = targets.flatten()

            # Identify indices of `5` and `6`
            indices_of_answer = (x == answer_token_id).nonzero(as_tuple=True)[0]
            indices_of_query = (x == query_token_id).nonzero(as_tuple=True)[0]

            # Initialize a mask with all False values
            mask = torch.zeros_like(x, dtype=torch.bool)

            # Iterate over pairs of `5` and `6` indices and set mask in between
            for start, end in zip(indices_of_answer, indices_of_query):
                mask[start+1:end+1] = True

            # print(mask)
            # Combine the answer-only mask with the non-pad mask
            final_mask = mask.view(B * T) & non_pad_mask
            # print("final", final_mask)
            # print("fm", final_mask)


            logits = logits[final_mask]
            targets = targets[final_mask]
            # print("targets",targets)

        loss = F.cross_entropy(logits, targets)

        # print("loss", loss, non_pad_mask.sum().item())
        return 0, loss, non_pad_mask.sum().item()



if __name__ == "__main__":
    import numpy as np



    device = 'cuda'

    emsize = 256  # embedding dimension
    d_hid = 256  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 4  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.1  # dropout probability
    ntokens = 100#50257
    block_size = 1024
    model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, device=device).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("num parameters", str(int(params/10**6)) + "M")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    # generate_text(model, [[94] + [1]*10 + [95]])

    input_tokens = (torch.rand(32, 128)*1000).to(torch.long)
    target_tokens = (torch.rand(32, 128)*1000).to(torch.long)

    accumulation_steps = 1  # Number of batches to accumulate gradients over
    # Make sure to zero gradients at the start
    optimizer.zero_grad()

    for i in range(500):
        

        for s in range(accumulation_steps):
            # pad_token_id=93, query_token_id=94, answer_token_id=95
            l = [94] + [1]*10 + [95] + [2] * 10 + [94] + [3] * 10 + [95] + [5]*10 + [94]

            tokens = torch.tensor([l, l], dtype=torch.long).to(device)
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            # input_tokens = (torch.rand(32, 128)*1000).to(torch.long)
            # target_tokens = (torch.rand(32, 128)*1000).to(torch.long)

            logits, loss, tokens = model.loss_calc(input_tokens, targets=target_tokens)
            # Normalize the loss to account for accumulation
            loss = loss / accumulation_steps
            loss.backward()
        print(f"Step {i+1}/{accumulation_steps}, Loss: {loss.item()}")

        # After accumulating gradients, perform the optimization step
        optimizer.step()

        # Zero gradients after the step
        optimizer.zero_grad()
    
    # l = [94] + [1]*10 + [95] + [2] * 10 + [94] + [3] * 10 + [95] #+ [5]*10 + [94]

    generate_text(model, [l])