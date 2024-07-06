from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json
import numpy as np
import math
from transformers import AutoTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import sys
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)


from torch import nn, Tensor
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None, pad_mask: bool = False) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        if pad_mask:
            pad_mask = (src==0)
            # print(pad_mask)
            pad_mask = pad_mask.t()
        else:
            pad_mask = None
        # print(pad_mask.shape, src.shape)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(src.device)
        # src_mask = src_mask.to(torch.bfloat16)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=pad_mask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def create_padding_mask(seq: Tensor, pad_token: int) -> Tensor:
    """
    Create a padding mask for the given sequence.
    Arguments:
        seq: Tensor, shape ``[batch_size, seq_len]``
        pad_token: int, the value used for padding

    Returns:
        mask: Tensor, shape ``[batch_size, seq_len]``, where positions of padding tokens are 1, else 0
    """
    return (seq == pad_token).transpose(0, 1)  # Transpose to shape [seq_len, batch_size]


def generate_text(model, tokenizer, prompt, max_length=10, temperature=0.8, top_p=0.9, device='cuda'):
    model.eval()
    generated = tokenizer.encode(prompt, return_tensors='pt').to(device)
    last_token = -1
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            # print(outputs[0].shape)
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            if next_token.item() != last_token:
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            last_token = next_token.item()        
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

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


tokenizer = AutoTokenizer.from_pretrained("gpt2")


# ntokens = len(tokenizer.get_vocab())  # size of vocabulary
# emsize = 400  # embedding dimension
# d_hid = 400  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 2  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability
# model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)



ntokens = len(tokenizer.get_vocab())  # size of vocabulary
emsize = 400  # embedding dimension
d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 12  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)



# model.load_state_dict(torch.load("llm_token.pt", map_location=device))
# model.load_state_dict(torch.load("llm_simple_conv_large.pt", map_location=device))
model.load_state_dict(torch.load("llm_57m_gpt2.pt", map_location=device))

model.eval()

str_txt = ""
while True:
    # i = input(">>")
    i = "I am happy and glad that"
    str_txt += i
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_txt))
    # print("tokens", tokens)

    l = len(str_txt)
    with torch.set_grad_enabled(False):
        t = generate_text(model, tokenizer, i, max_length=20, temperature=0.99, device=device)

    print(t)
   
    break