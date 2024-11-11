from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json
import numpy as np
import math
import time
# from transformers import BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import sys
import shutil
from transformers import GPT2Tokenizer
from transformer_model_llama import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)


from torch import nn, Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


# PAD is 93, Sep is 94


      

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


def generate_text_llama(model, tokenizer, tokens, max_length=100, temperature=0.7, top_p=0.9, device='cuda', stream=True):
    model.eval()
    model.clear_kv_cache()
    # tokens = tokenizer.encode(prompt, return_tensors='pt')
    # q_token = torch.tensor([[94]], dtype=torch.long)
    # tokens = torch.cat([tokens, q_token], dim=1)
    generated = torch.tensor([tokens], dtype=torch.long).to(device)
    # generated = tokens.to(device)
    for g in range(generated.shape[1]):
        model(generated[:, g:g+1], start_pos=g)
    # res = model.generate(idx, max_new_tokens=100):
    chunk_len = len(tokenizer.decode(generated[0], skip_special_tokens=True))
    last_token = 0
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated[:, -1::], start_pos=generated.shape[1])
            next_token_logits = outputs[0, :] / temperature
            next_token_logits[0, last_token] = -10000
            next_token_logits = next_token_logits.squeeze()
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            # print(filtered_logits, filtered_logits.shape)
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            next_token = torch.multinomial(probabilities, 1)


            # print(probabilities.shape)
            # next_token = torch.argmax(next_token_logits).unsqueeze(0)
            # print(generated.shape, next_token, next_token.shape, next_token_logits)
        
            
            if next_token != last_token:
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            if next_token.item() == 94:
                break
            last_token = next_token        
            if stream:
              chunk = tokenizer.decode(generated[0], skip_special_tokens=True)
              print(chunk[chunk_len::], end='', flush=True)
              chunk_len = len(chunk)
              
              # time.sleep(0.1)
    # print(generated[0])
    return generated[0]#tokenizer.decode(generated[0], skip_special_tokens=True)


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

# 124 llama
emsize = 768  # embedding dimension
d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 12  # number of heads in ``nn.MultiheadAttention``
dropout = 0.1  # dropout probability
head_size = 48


# 500m llama
emsize = 1024+128
d_hid = emsize*4
nlayers = 24
nhead = 16
dropout = 0.1 

model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, device=device).to(device)




model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("num parameters", str(int(params/10**6)) + "M")

model_name = "model_498m_nov10.pt"

model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))

model.eval()

print("loaded")
torch.manual_seed(0)

text = ""
prompt_num = 0 
tokens_all = []
while True:
    prompt = input("\n>>")

    if len(prompt) == 0:
        print("reset")
        tokens_all = []
        continue
    

    tokens_prompt = tokenizer.encode(prompt) + [95]
    tokens_all += tokens_prompt
    l = len(tokens_all)

    
    with torch.set_grad_enabled(False):
      tokens_all = list(generate_text_llama(model, tokenizer, tokens_all, max_length=100, temperature=0.8, top_p=0.9, device=device, stream=True).detach())

    # print(tokens_all)    
    # print(tokenizer.decode(tokens_all[l:-1], skip_special_tokens=True))

    tokens_all.append(94)
