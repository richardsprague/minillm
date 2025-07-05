#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/sprague/dev/nathan/llm')

from transformer_model_llama_june2025 import TransformerModel
import torch
import torch.nn.functional as F
import os
from tokenizers import ByteLevelBPETokenizer

# Copy the exact generation function from chat_transformer.py
def generate_text(model, tokenizer, tokens, max_length=100, temperature=0.7, top_p=0.9,
                  repetition_penalty=1.1, stream=False):
    model.clear_kv_cache()
    
    generated = tokens
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

            newline_tokens = [208, 230, 15078, 19]
            if generated[:, -1].item() in newline_tokens and generated[:, -2].item() in newline_tokens:
                for n in newline_tokens:
                    next_token_logits[0, n] -= 1000

            if generated[:, -1].item() == 8:  # think_end_token
                if thinking:
                    thinking = False
            if not thinking:
                next_token_logits[0, 8] -= 1000  # think_end_token

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
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, 1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            last_recent_tokens.append(next_token.item())
            if len(last_recent_tokens)>100:
                last_recent_tokens = last_recent_tokens[1::]
            last_token = next_token
            if last_token == 2:  # answer_end_token
                break

            if stream:
                print(tokenizer.decode([next_token.item()]), end='', flush=True)

    return generated[0].tolist()

# Setup
model_name = "../model505m_july3_2025/model505m_july3_2025.pt"
tokenizer_name = "../model505m_july3_2025/my_tokenizer_50k_2025"

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = TransformerModel(ntokens, block_size, emsize, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=1, device=device).to(device)
model.eval()
model.load_state_dict(torch.load(model_name, weights_only=True, map_location=device))

# Test generation exactly like original
prompt = "Hello!"
encoded = tokenizer.encode(prompt)
tokens = encoded.ids + [question_end_token, think_start_token]
tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

print(f"Input: {prompt}")
print(f"Tokens: {tokens}")

# Generate
all_tokens = generate_text(model, tokenizer, tokens_tensor, temperature=0.3, top_p=0.9, max_length=50, repetition_penalty=1.2)

print(f"All tokens: {all_tokens}")

# Process like original
if all_tokens[-1] == answer_end_token:
    all_tokens = all_tokens[0:-1]

if think_end_token in all_tokens:
    msg = all_tokens[all_tokens.index(think_end_token)+1::]
else:
    msg = all_tokens[len(tokens)::]

print(f"Message tokens: {msg}")
print(f"Final response: {tokenizer.decode(msg)}")