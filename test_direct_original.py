#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/Users/sprague/dev/nathan/llm')

# Import the exact original model
from transformer_model_llama_june2025 import TransformerModel
import torch

# Use the exact parameters from chat_transformer
model_name = "../model505m_july3_2025/model505m_july3_2025.pt"

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
device = 'cpu'  # Use CPU to avoid GPU timeout issues

print("Creating original model...")
model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=1, device=device)
print("Model created, loading weights...")

model.load_state_dict(torch.load(model_name, weights_only=True, map_location=device))
print("Weights loaded!")

model.eval()

# Test simple forward pass
tokens = torch.tensor([[34564, 10, 1, 7]], dtype=torch.long)
print(f"Testing forward pass with tokens: {tokens}")

with torch.no_grad():
    logits = model(tokens, start_pos=0)
    print(f"Output shape: {logits.shape}")
    next_token = torch.argmax(logits[0, -1, :])
    print(f"Next token: {next_token.item()}")