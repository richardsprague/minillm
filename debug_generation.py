#!/usr/bin/env python3

import os
import sys
sys.path.append('.')

from minillm.config import load_config
from minillm.tokenizer import TokenizerManager
from tokenizers import ByteLevelBPETokenizer

# Compare tokenizer directly
config = load_config()
our_tokenizer = TokenizerManager(config.paths, config.tokens)

# Load the raw tokenizer like the original
tokenizer_name = "../model505m_july3_2025/my_tokenizer_50k_2025"
raw_tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-vocab.json"),
    merges_filename=os.path.join(tokenizer_name, "tokenizer_50k_2025-merges.txt")
)

print("=== Tokenizer Comparison ===")
test_text = "Hello!"
print(f"Input: {test_text}")

our_encoded = our_tokenizer.encode(test_text, add_special_tokens=False)
raw_encoded = raw_tokenizer.encode(test_text, add_special_tokens=False).ids

print(f"Our tokenizer: {our_encoded}")
print(f"Raw tokenizer: {raw_encoded}")
print(f"Match: {our_encoded == raw_encoded}")

# Test decode
test_tokens = [208, 35194, 21, 272, 3538]
our_decoded = our_tokenizer.decode(test_tokens, skip_special_tokens=True)
raw_decoded = raw_tokenizer.decode(test_tokens, skip_special_tokens=True)

print(f"Our decode: {repr(our_decoded)}")
print(f"Raw decode: {repr(raw_decoded)}")
print(f"Decode match: {our_decoded == raw_decoded}")

# Test problematic tokens that might be causing garbled output
problematic_tokens = [181, 69, 10]  # Tokens we've seen causing issues
for token in problematic_tokens:
    our_decoded = our_tokenizer.decode([token], skip_special_tokens=True)
    raw_decoded = raw_tokenizer.decode([token], skip_special_tokens=True)
    print(f"Token {token}: our={repr(our_decoded)}, raw={repr(raw_decoded)}")