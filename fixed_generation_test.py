#!/usr/bin/env python3

# Test generation with original working components
import sys
sys.path.insert(0, '/Users/sprague/dev/nathan/llm')

from transformer_model_llama_june2025 import TransformerModel
import torch
import torch.nn.functional as F
import os
from tokenizers import ByteLevelBPETokenizer

# Setup exactly like working original
model_name = "../model505m_july3_2025/model505m_july3_2025.pt"
tokenizer_name = "../model505m_july3_2025/my_tokenizer_50k_2025"

device = torch.device("cpu")  # Use CPU for testing

# Load tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-vocab.json"),
    merges_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-merges.txt")
)

# Special tokens
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

print("Model loaded successfully!")

# Define simplified generation function
def simple_generate(prompt, max_tokens=20):
    # Encode exactly like original
    encoded = tokenizer.encode(prompt)
    tokens = encoded.ids + [question_end_token, think_start_token]
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    model.clear_kv_cache()
    model(tokens_tensor[:, :-1], start_pos=0)
    
    generated = tokens_tensor
    all_generated = []
    
    def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        return logits
    
    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(generated[:, -1:], start_pos=generated.shape[1])
            next_token_logits = outputs[0, :] / 0.7  # temperature
            
            next_token_logits = next_token_logits.squeeze()
            filtered_logits = top_p_filtering(next_token_logits, top_p=0.9)
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            next_token = torch.multinomial(probabilities, 1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            all_generated.append(next_token.item())
            
            if next_token.item() == answer_end_token:
                break
    
    # Process like original
    if all_generated and all_generated[-1] == answer_end_token:
        all_generated = all_generated[:-1]
    
    if think_end_token in all_generated:
        think_idx = all_generated.index(think_end_token)
        message_tokens = all_generated[think_idx + 1:]
    else:
        message_tokens = all_generated
    
    return tokenizer.decode(message_tokens)

# Test the function
print("Testing generation...")
response = simple_generate("Hello!")
print(f"Response: {repr(response)}")

# Test with different prompts
test_prompts = ["Hi", "How are you?", "What is 2+2?"]
for prompt in test_prompts:
    response = simple_generate(prompt, max_tokens=10)
    print(f"Prompt: {prompt} -> Response: {repr(response)}")