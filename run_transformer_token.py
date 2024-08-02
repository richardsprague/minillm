from torch.utils.data import Dataset, DataLoader
import argparse
import torch
import torch.nn as nn
import json
import numpy as np
import math
# from transformers import BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import sys
import shutil
from transformers import GPT2Tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)


from torch import nn, Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


# model architecture
class AttentionHead(nn.Module):
  """a single head of self attention"""
  
  def __init__(self, n_embed, head_size, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    B, T, C = x.shape
    K = self.key(x) # (B, T, C)
    Q = self.query(x) # (B, T, C)
    
    wei = Q @ K.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, H, C) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    V = self.value(x) # (B, T, C)
    out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out
  
class MultiHeadAttention(nn.Module):
  """a multi-head self attention layer"""
  
  def __init__(self, n_embed, n_heads, head_size, block_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
    self.fc = nn.Linear(head_size * n_heads, n_embed)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads*C)    
    out = self.fc(out) # (B, T, C)
    out = self.dropout(out) 
    return out
  
class FeedForward(nn.Module):
  def __init__(self, n_embed, n_hidden, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_embed),
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
   return self.net(x)
  
class Block(nn.Module):
  def __init__(self, n_embed, n_heads, block_size, dropout):
    super().__init__()
    self.sa_heads = MultiHeadAttention(n_embed, n_heads, n_embed // n_heads, block_size, dropout)
    self.ffwd = FeedForward(n_embed, n_embed*4, dropout)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
    
    
  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x)) #  [batch_size, block_size, n_embed]
    x = x + self.ffwd(self.ln2(x)) # [batch_size, block_size, n_embed]
    return x

class TransformerModel(nn.Module):
  def __init__(self, vocab_size, block_size, n_embed, n_heads, n_layers, dropout, device="cpu"):
    super().__init__()
    # Adapted from https://github.com/broskicodes/slms/tree/master

    print(vocab_size, block_size, n_embed, n_heads, n_layers, dropout)
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
    
    self.device = device
    self.block_size = block_size

  def forward(self, idx, targets=None, pad_token_id=0):
    # idx and target are both [batch_size, block_size]
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx) # [batch_size, block_size, n_embed]
    pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # [block_size, n_embed]
    x = tok_emb + pos_emb # [batch_size, block_size, n_embed]
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # [batch_size, block_size, vocab_size]
    
    if targets is None:
        loss = None
        n_tokens = 0
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.reshape(B*T)
        
        # Create a mask for non-padding tokens
        non_pad_mask = (targets != pad_token_id)
        
        # Apply the mask to both logits and targets
        logits = logits[non_pad_mask]
        targets = targets[non_pad_mask]
        
        # Calculate the number of non-padding tokens
        n_tokens = non_pad_mask.sum().item()
        
        loss = F.cross_entropy(logits, targets)
    
    return logits, loss, n_tokens
      

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    model.eval()
    generated = tokenizer.encode(prompt, return_tensors='pt').to(device)

    last_token = 0
    with torch.no_grad():
        for _ in range(max_length):
            outputs, _, _ = model(generated)
            next_token_logits = outputs[0, -1, :] / temperature
            next_token_logits[last_token] = -10000
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, 1)
            
            if next_token != last_token:
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            
            last_token = next_token        
            # if next_token.item() == tokenizer.sep_token_id:
            #     break
    # print(generated[0])
    return tokenizer.decode(generated[0], skip_special_tokens=True)
    # return tokenizer.decode(list(generated[0]))

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


ntokens = len(tokenizer.get_vocab())  # size of vocabulary
emsize = 512  # embedding dimension
d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 6  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability
block_size = 1024
model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, dropout, device=device).to(device) # 30522 1024 512 8 6 0.1


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("num parameters", str(int(params/10**6)) + "M")


# model.load_state_dict(torch.load("llm_token.pt", map_location=device))
# model.load_state_dict(torch.load("llm_simple_conv_large.pt", map_location=device))
model.load_state_dict(torch.load("model70m.pt", map_location=device))

# model.eval()

# torch.manual_seed(1)

# prompt = "The boiling point of"

# with torch.set_grad_enabled(False):
#     generated_text = generate_text(model, tokenizer, prompt, max_length=30, temperature=0.8, top_p=0.9, device=device)
# print(generated_text)



# model.load_state_dict(torch.load("llm_token.pt", map_location=device))
# model.load_state_dict(torch.load("llm_simple_conv_large.pt", map_location=device))
# model.load_state_dict(torch.load("llm_57m_gpt2.pt", map_location=device))


def main():
    parser = argparse.ArgumentParser(description='Transformer Inference Script')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input text to replace the string')
    
    args = parser.parse_args()
    input_text = args.input

    model.eval()

    str_txt = ""
    while True:
        i = input_text
        str_txt += i
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_txt))
        
        l = len(str_txt)
        with torch.set_grad_enabled(False):
            t = generate_text(model, tokenizer, i, max_length=20, temperature=0.99, device=device)

        print(t)
        break

if __name__ == "__main__":
    main()
