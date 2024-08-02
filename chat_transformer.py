from torch.utils.data import Dataset, DataLoader
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


# PAD is 93, Sep is 94

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
      # nn.ReLU(),
      nn.GELU(),
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

  def forward(self, idx, targets=None, pad_token_id=0, sep_token_id=1):
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
      

def generate_text(model, tokenizer, tokens, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    model.eval()
    # tokens = tokenizer.encode(prompt, return_tensors='pt')
    # q_token = torch.tensor([[94]], dtype=torch.long)
    # tokens = torch.cat([tokens, q_token], dim=1)

    generated = torch.tensor([tokens], dtype=torch.long).to(device)
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
            if next_token.item() == 94:
                break
    # print(generated[0])
    return generated[0]#tokenizer.decode(generated[0], skip_special_tokens=True), list(generated)
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
# mobilellm shape(124.6M - really 151M 178m)
# emsize = 576  # embedding dimension
# d_hid = 1536  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 30  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 9  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.1  # dropout probability

#30m
# emsize = 256  # embedding dimension
# d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 6  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 8  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.1  # dropout probability


# 70m
emsize = 512  # embedding dimension
d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 6  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability
block_size = 1024

# 30m
emsize = 256  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 6  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability
block_size = 1024


model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, dropout, device=device).to(device)




model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("num parameters", str(int(params/10**6)) + "M")


# model.load_state_dict(torch.load("llm_token.pt", map_location=device))
# model.load_state_dict(torch.load("llm_simple_conv_large.pt", map_location=device))
# model.load_state_dict(torch.load("/home/nathan/Desktop/llm/model70m_july25.pt", map_location=device))
# model.load_state_dict(torch.load("chat_70m_july29.pt", map_location=device, weights_only=True))
# model.load_state_dict(torch.load("model_261_july26.pt", map_location=device, weights_only=True))
model.load_state_dict(torch.load("model_30m_aug2_eye_qa.pt", map_location=device, weights_only=True))

model.eval()

print("loaded")
torch.manual_seed(0)

text = ""
prompt_num = 0 
tokens_all = []
q_token = torch.tensor([[94]], dtype=torch.long)
while True:
    prompt = input(">>")

    

    tokens_prompt = tokenizer.encode(prompt) + [94]
    tokens_all += tokens_prompt
    l = len(tokens_all)

    
    with torch.set_grad_enabled(False):
      tokens_all = list(generate_text(model, tokenizer, tokens_all, max_length=100, temperature=0.8, top_p=0.9, device=device).detach())
    
    print(tokenizer.decode(tokens_all[l::], skip_special_tokens=True))

    tokens_all.append(94)