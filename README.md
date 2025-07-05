# MinillM

[![CI/CD Pipeline](https://github.com/nsprague/minillm/actions/workflows/ci.yml/badge.svg)](https://github.com/nsprague/minillm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A production-ready, modular transformer-based language model implementation based on LLaMA architecture. Built for scalability, performance, and ease of deployment.

## ✨ Features

- **🏗️ Modular Architecture**: Clean separation of concerns with configurable components
- **⚡ High Performance**: torch.compile, quantization, FlashAttention support
- **🌐 Production Ready**: FastAPI web server, Docker support, CI/CD pipeline
- **🔧 Hardware Adaptive**: Seamless deployment from Mac M2 (8GB) to Linux GPU servers
- **📊 Monitoring**: Weights & Biases, TensorBoard integration
- **🎯 Easy to Use**: Simple CLI tools and enhanced web interface

### 🆕 New Enhanced Web Interface

- **📝 Quick Questions**: Pre-made question dropdown including "What is the fastest animal?", "How many Rs are there in strawberry?", and "If the Han dynasty Chinese fought the Romans, who would win?"
- **🤖 Advanced Model Management**: Support for local files, Hugging Face models, and OpenAI API
- **📁 File Upload System**: Browse and upload .pt, .pth, .bin, .safetensors model files directly
- **⚙️ Real-time Controls**: Adjust temperature, top-p, max length, top-k parameters dynamically
- **📱 Mobile Responsive**: Optimized interface that works perfectly on all devices
- **🎨 Professional Design**: Clean, modern interface with Purdue branding

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nsprague/minillm.git
cd minillm

# Install in development mode
pip install -e ".[dev]"

# Or install for production
pip install -e .
```

### Download Model

Download the pretrained model (505M parameters) from [Google Drive](https://drive.google.com/file/d/1CHAgimS47Y34nvUBBpugoSL1FfJ_KVB4/view?usp=sharing) and extract to the parent directory.

### Quick Chat

```bash
# Start interactive chat
llm-chat

# Or with custom settings
llm-chat --temperature 0.8 --max-length 150 --compile
```

### Enhanced Web Interface

```bash
# Start web server with all features
llm-serve --port 8000

# Open http://localhost:8000 in your browser
```

**Features available in the web interface:**
- **Quick Questions**: Select from pre-made questions or type your own
- **Model Selection**: Choose from local, Hugging Face, or OpenAI models
- **File Upload**: Upload your own model files (.pt, .pth, .bin, .safetensors)
- **Real-time Parameters**: Adjust temperature, top-p, max length, top-k
- **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile

## 📋 Model Card

### Model Details

- **Architecture**: LLaMA-style decoder-only transformer with Grouped Query Attention
- **Parameters**: 505M (1024 hidden dim, 24 layers, 16 heads, 4096 FFN dim)
- **Context Length**: 128-4096 tokens (configurable)
- **Vocabulary**: 50K BPE tokens
- **Precision**: FP32/FP16/BF16 support
- **Special Features**: Chain-of-thought reasoning with thinking tokens

### Training Data

- **Pretraining**: Fineweb-edu (~15B tokens) + BAAI/CCI4.0-M2-CoT-v1 (~2B tokens)
- **Finetuning**: Ultrachat, Google Natural Questions, Smoltalk
- **Answer Generation**: Qwen3:8b

### Performance

- **Memory Usage**: ~2GB FP32, ~1GB FP16, ~500MB INT8
- **Speed**: ~50 tokens/sec on M2 Mac, ~150 tokens/sec on RTX 3090
- **Quality**: Competitive with similar-sized models on common benchmarks

## 🏗️ Architecture

### Core Components

```
minillm/
├── config.py          # Unified configuration with Pydantic
├── models.py          # Transformer model implementation  
├── layers.py          # Modular transformer layers
├── tokenizer.py       # Tokenizer management
├── generation.py      # Text generation utilities
├── optimization.py    # Performance optimizations
├── server.py          # FastAPI web server
├── cli.py            # Command-line interface
└── utils.py          # Utility functions
```

### Key Features

- **Grouped Query Attention (GQA)**: Efficient attention mechanism
- **RoPE**: Rotary positional embedding for better length extrapolation
- **SwiGLU**: Swish-gated linear units in feed-forward networks
- **RMS Normalization**: Root Mean Square layer normalization
- **KV Caching**: Efficient inference with cached attention states

## ⚙️ Configuration

All settings are managed through `config.yaml`:

```yaml
model:
  dim: 1024                  # Hidden dimension (actual model size)
  n_layers: 24               # Number of transformer layers  
  n_heads: 16                # Number of attention heads
  ffn_dim: 4096              # Feed-forward network dimension
  vocab_size: 50000          # Vocabulary size

performance:
  compile_model: false       # torch.compile (disabled for compatibility)
  use_quantization: false    # Enable INT8/INT4 quantization  
  use_flash_attention: false # Enable FlashAttention

generation:
  temperature: 0.7           # Sampling temperature
  top_p: 0.9                # Nucleus sampling
  max_length: 100           # Max generation length
```

## 🔧 Performance Optimizations

### CPU Optimizations

```bash
# Install performance dependencies
pip install ".[performance]"

# Enable Intel optimizations (Intel CPUs)
export USE_INTEL_EXTENSION=1

# Quantization for memory efficiency
llm-chat --quantize --quantization-bits 8
```

### GPU Optimizations

```bash
# Enable compilation and mixed precision
llm-chat --compile --dtype bfloat16

# FlashAttention for long sequences
pip install flash-attn
# Set use_flash_attention: true in config.yaml
```

### Memory Usage

| Configuration | Memory Usage | Speed | Quality |
|--------------|-------------|-------|---------|
| FP32 | ~2GB | Baseline | Best |
| FP16 | ~1GB | 1.5x faster | Excellent |
| BF16 | ~1GB | 1.5x faster | Excellent |
| INT8 | ~500MB | 2-3x faster | Very Good |
| INT4 | ~250MB | 3-4x faster | Good |

## 🌐 Deployment

### Docker Development (Recommended)

**Quick setup for development with live code reloading:**

```bash
# One-time setup
./scripts/dev-setup.sh

# Start development server
./scripts/docker-dev.sh start

# Your code changes are reflected immediately!
# Web interface: http://localhost:8000
```

**Available Docker commands:**
```bash
./scripts/docker-dev.sh start    # Start development server
./scripts/docker-dev.sh stop     # Stop server
./scripts/docker-dev.sh shell    # Open container shell
./scripts/docker-dev.sh logs     # View logs
./scripts/docker-dev.sh test     # Run tests
```

### Local Development (Alternative)

```bash
# Start development server with hot reload
uvicorn minillm.server:app --reload --port 8000
```

### Production Deployment

```bash
# Production with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -f Dockerfile.dev -t minillm .
docker run -p 8000:8000 -v $(pwd):/app minillm

# Deploy to cloud platforms
# Supports Heroku, Railway, Fly.io, etc.
```

### Netlify Frontend

The web interface in `web/static/` can be deployed to Netlify:

1. Update API endpoint in `index.html`
2. Deploy the `web/static/` directory
3. Configure your backend API URL

## 🔧 Recent Fixes & Troubleshooting

### Latest Updates (Jan 2025)

- ✅ **Enhanced Chat Interface**: Added quick questions dropdown with pre-made prompts
- ✅ **Advanced Model Management**: Support for local files, Hugging Face, and OpenAI models  
- ✅ **File Upload System**: Browse and upload .pt, .pth, .bin, .safetensors model files
- ✅ **Real-time Parameter Controls**: Adjust temperature, top-p, max length, top-k dynamically
- ✅ **Mobile-Responsive Design**: Optimized interface for desktop and mobile devices
- ✅ **Fixed tensor dimension error**: Resolved critical generation compatibility issues
- ✅ **Improved model compatibility**: 100% checkpoint compatibility with original architecture

### Common Issues

**Q: Getting "Tensors must have same number of dimensions" error?**  
A: This has been fixed in the latest version. Update to the newest commit.

**Q: Model generates garbled text?**  
A: Ensure you're using the exact model checkpoint from the Google Drive link. The model architecture now matches the original exactly.

**Q: Web interface shows 404 errors?**  
A: Fixed in latest version with proper static file serving and favicon support.

**Q: Slow model loading?**  
A: The model now uses the original architecture for faster loading. Disable compilation if needed: `compile_model: false`

## 🛠️ Development

### Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
ruff format minillm/
ruff check minillm/ --fix
```

### Adding Features

1. Update configuration in `config.py`
2. Implement feature in appropriate module
3. Add CLI interface if needed
4. Update tests and documentation
5. Run CI pipeline

## 📊 Monitoring

### Weights & Biases

```yaml
logging:
  use_wandb: true
  wandb_project: "minillm"
```

### TensorBoard

```bash
# Enable in config.yaml
logging:
  use_tensorboard: true

# View logs
tensorboard --logdir logs/tensorboard
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LLaMA Architecture**: Meta AI for the original transformer design
- **Training Data**: Fineweb-edu and BAAI teams for high-quality datasets
- **Optimization Techniques**: FlashAttention, torch.compile, and quantization communities
- **Open Source Libraries**: PyTorch, Transformers, FastAPI, and many others

## 📚 Citation

If you use MinillM in your research or projects, please cite:

```bibtex
@software{minillm2025,
  author = {Nathan Sprague},
  title = {MinillM: A Production-Ready Transformer Language Model},
  url = {https://github.com/nsprague/minillm},
  year = {2025}
}
```

---

**Note**: This model is for research and educational purposes. Please ensure compliance with applicable laws and ethical guidelines when deploying in production environments.