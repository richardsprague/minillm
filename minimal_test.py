#!/usr/bin/env python3

from minillm.config import load_config
from minillm.models import TransformerModel
from minillm.tokenizer import TokenizerManager
import torch
import torch.nn.functional as F

# Load with our new system
config = load_config()
tokenizer = TokenizerManager(config.paths, config.tokens)
model = TransformerModel.from_pretrained(config.paths.model_file, config.model)
model.eval()

# Create the exact same input as the working test
prompt = "Hello!"
encoded = tokenizer.encode(prompt, add_special_tokens=False)
tokens = encoded + [1, 7]  # question_end_token, think_start_token
print(f"Input tokens: {tokens}")

# Convert to tensor
tokens_tensor = torch.tensor([tokens], dtype=torch.long)

# Use the exact generation logic from the original
def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

# Generate exactly like the original
model.clear_kv_cache()
generated = tokens_tensor
model(generated[:, :-1], start_pos=0)

generated_tokens = []
with torch.no_grad():
    for i in range(10):  # Just a few tokens for testing
        outputs = model(generated[:, -1:], start_pos=generated.shape[1])
        next_token_logits = outputs[0, :] / 0.7  # temperature
        
        next_token_logits = next_token_logits.squeeze()
        filtered_logits = top_p_filtering(next_token_logits, top_p=0.9)
        probabilities = F.softmax(filtered_logits, dim=-1)
        
        next_token = torch.multinomial(probabilities, 1)
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        generated_tokens.append(next_token.item())
        
        print(f"Token {i}: {next_token.item()} -> {repr(tokenizer.decode([next_token.item()]))}")

print(f"Generated tokens: {generated_tokens}")
print(f"Full decode: {repr(tokenizer.decode(generated_tokens))}")