# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready transformer-based language model implementation based on LLaMA architecture. The model is pretrained on Fineweb-edu (~15B tokens) and BAAI/CCI4.0-M2-CoT-v1 (~2B tokens), then finetuned on various QA datasets. The current best model has 505M parameters with chain-of-thought reasoning capabilities.

**Recent Major Updates (Jan 2025):**
- Enhanced chat interface with quick questions dropdown
- Advanced model management supporting local files, Hugging Face, and OpenAI models
- File upload system for .pt, .pth, .bin, .safetensors model files
- Real-time parameter controls (temperature, top-p, max length, top-k)
- Mobile-responsive design optimized for all devices
- Fixed critical tensor dimension errors in generation
- Improved model compatibility with original checkpoint

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
```bash
# Run interactive chat (CLI)
llm-chat

# Start enhanced web server with all features
llm-serve --port 8000

# Legacy chat interface  
python chat_transformer.py

# Finetune on custom dataset
python finetune_llama.py
```

### Enhanced Web Interface Features
```bash
# Access the web interface at http://localhost:8000
# Features include:
# - Quick questions dropdown with pre-made prompts
# - Model selection (local, Hugging Face, OpenAI)
# - File upload for model files (.pt, .pth, .bin, .safetensors)
# - Real-time parameter adjustment (temperature, top-p, max length, top-k)
# - Mobile-responsive design
```

### Docker Development (Recommended)
```bash
# One-time setup
./scripts/dev-setup.sh

# Start development server with live reloading
./scripts/docker-dev.sh start

# Available commands:
./scripts/docker-dev.sh stop     # Stop server
./scripts/docker-dev.sh shell    # Open container shell  
./scripts/docker-dev.sh logs     # View logs
./scripts/docker-dev.sh test     # Run tests
./scripts/docker-dev.sh clean    # Clean up resources
```

### VS Code Tasks
- **Install dependencies**: `python3 -m pip install -r requirements.txt`
- **Run Chat Transformer**: `python chat_transformer.py`  
- **Finetune Model**: `python finetune_llama.py`

## Required Files

Before running the model, ensure you have:
1. **Model file**: `model505m_july3_2025.pt` (download from Google Drive link in README)
2. **Tokenizer directory**: `my_tokenizer_50k_2025/` containing `vocab.json` and `merges.txt`

Update paths in `chat_transformer.py` if needed:
```python
model_name = "model505m_july3_2025.pt"
tokenizer_name = "my_tokenizer_50k_2025"
```

## Architecture Overview

### Core Components
- **TransformerModel**: Main model class with decoder-only architecture
- **TransformerBlock**: Individual transformer layers with attention + FFN
- **Attention**: Grouped Query Attention with KV caching and RoPE
- **FeedForward**: SwiGLU activation with gated linear units
- **RMSNorm**: Root Mean Square layer normalization

### Key Files
- `transformer_model_llama_june2025.py`: Main model implementation
- `transformer_model_llama_june2025_checkpointing.py`: Memory-optimized version with gradient checkpointing
- `chat_transformer.py`: Inference and chat interface
- `finetune_llama.py`: Training script and data loading

### Model Configuration (Actual Model Dimensions)
- **Hidden dimension**: 1024 (actual model size)
- **FFN dimension**: 4096 (actual model size)
- **Layers**: 24 (actual model size)
- **Attention heads**: 16 (actual model size)
- **Vocabulary size**: 50,000 tokens
- **Max sequence length**: 128-4096 (configurable)

**Note**: The config.yaml has been updated to match the actual model dimensions for proper compatibility.

## Memory Management

### Memory-Efficient Training
- Use `transformer_model_llama_june2025_checkpointing.py` for ~30% memory savings
- Gradient checkpointing trades compute for memory
- Mixed precision training with `torch.amp.autocast`

### KV Cache Management
- Efficient inference with cached attention states
- `model.clear_kv_cache()` to reset conversation context
- Automatic cache positioning with `start_pos` parameter

## Data Format

### Conversation Structure
The model expects conversational data with special tokens:
- **Question end**: Token 1
- **Answer end**: Token 2  
- **Think start**: Token 7
- **Think end**: Token 8
- **Padding**: Token 0

### Training Data
- **Format**: JSONL with message arrays
- **Masking**: Questions masked (not trained), answers trained
- **Reasoning**: Supports `<think>...</think>` tags for chain-of-thought

### Finetuning
To finetune on custom data:
1. Update `self.pth` in `MessageDataset` class in `finetune_llama.py`
2. Ensure data follows the conversation format
3. Run `python finetune_llama.py`

## Generation Features

### Advanced Sampling
- **Temperature**: Controls randomness (default 0.7)
- **Top-p**: Nucleus sampling (default 0.9)
- **Repetition penalty**: Reduces repetitive outputs (default 1.1)
- **Token filtering**: Prevents infinite newlines

### Conversation Mode
- Multi-turn conversations with context preservation
- Automatic KV cache management
- Press Enter on empty prompt to start new conversation

## Dependencies

Required packages in `requirements.txt`:
- torch
- numpy  
- tokenizers
- transformers
- pandas
- fastapi
- uvicorn
- pydantic
- pyyaml

**Note**: Use `pip install -e .` to install all dependencies automatically from pyproject.toml.

## Troubleshooting

### Common Issues and Fixes

1. **Tensor Dimension Error** (`Tensors must have same number of dimensions: got 2 and 1`)
   - **Status**: FIXED in latest version
   - **Cause**: Tensor shape mismatch in generation loop
   - **Fix**: Updated generation.py with proper tensor reshaping

2. **Garbled Generation Output**
   - **Status**: FIXED in latest version  
   - **Cause**: Model architecture incompatibility with checkpoint
   - **Fix**: Now uses original TransformerModel directly for 100% compatibility

3. **Web Interface 404 Errors**
   - **Status**: FIXED in latest version
   - **Cause**: Missing static file serving and favicon
   - **Fix**: Added proper StaticFiles mount and favicon route

4. **Slow Model Loading**
   - **Status**: IMPROVED in latest version
   - **Cause**: Complex model wrapper causing delays
   - **Fix**: Direct use of original model architecture

### Model Compatibility Notes

- The project now uses the original `transformer_model_llama_june2025.py` directly
- This ensures 100% compatibility with existing checkpoint files
- Generation logic matches the original chat_transformer.py behavior
- Chain-of-thought reasoning works properly with think_start/think_end tokens