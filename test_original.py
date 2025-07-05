#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/sprague/dev/nathan/llm')

from transformer_model_llama_june2025 import TransformerModel
import torch
import torch.nn.functional as F
import os
from tokenizers import ByteLevelBPETokenizer

# Copy the exact setup from chat_transformer.py
model_name = "../model505m_july3_2025/model505m_july3_2025.pt"
tokenizer_name = "../model505m_july3_2025/my_tokenizer_50k_2025"

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-vocab.json"),
    merges_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-merges.txt")
)

question_end_token = 1
answer_end_token = 2
think_start_token = 7
think_end_token = 8

# Model parameters
emsize = -1
d_hid = 6144
nlayers = 24
nhead = 16
dropout = 0.01
ntokens = 50000
block_size = 4096
ffn_dim = 4096
dim = 1024

# Create and load model
model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=1, device=device).to(device)
model.eval()
model.load_state_dict(torch.load(model_name, weights_only=True, map_location=device))

# Test simple generation
prompt = "Hello!"
print(f"Testing prompt: {prompt}")

# Encode like original
encoded = tokenizer.encode(prompt)
print(f"Encoded: {encoded.ids}")
tokens = encoded.ids + [question_end_token, think_start_token]
print(f"Tokens: {tokens}")

# Convert to tensor
tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
print(f"Tokens tensor shape: {tokens_tensor.shape}")

# Clear cache and prefill
model.clear_kv_cache()
model(tokens_tensor[:, :-1], start_pos=0)

# Generate one token
with torch.no_grad():
    logits = model(tokens_tensor[:, -1:], start_pos=tokens_tensor.shape[1])
    next_token_logits = logits[0, :] / 0.7  # temperature
    
    # Apply top-p filtering
    def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        return logits
    
    next_token_logits = next_token_logits.squeeze()
    filtered_logits = top_p_filtering(next_token_logits, top_p=0.9)
    probabilities = F.softmax(filtered_logits, dim=-1)
    
    next_token = torch.multinomial(probabilities, 1)
    print(f"Next token: {next_token.item()}")
    print(f"Next token decoded: {tokenizer.decode([next_token.item()])}")
    
    # Generate a few more tokens to see the pattern
    generated = tokens_tensor
    generated_tokens = []
    for i in range(10):
        with torch.no_grad():
            logits = model(generated[:, -1:], start_pos=generated.shape[1])
            next_token_logits = logits[0, :] / 0.7
            next_token_logits = next_token_logits.squeeze()
            filtered_logits = top_p_filtering(next_token_logits, top_p=0.9)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            generated_tokens.append(next_token.item())
            
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generated text: {tokenizer.decode(generated_tokens)}")