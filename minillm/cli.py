"""
Command-line interface for MinillM.
Provides console entry points for chat, finetuning, and serving.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .config import load_config, Config
from .models import TransformerModel
from .tokenizer import TokenizerManager
from .generation import TextGenerator
from .utils import setup_device, setup_logging


def chat_main():
    """Main entry point for llm-chat command."""
    parser = argparse.ArgumentParser(description="Chat with MinillM")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--model-path", "-m", help="Override model path from config")
    parser.add_argument("--device", help="Override device from config (cpu, cuda, mps, auto)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument("--max-length", type=int, default=100, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with CLI args
        if args.model_path:
            config.paths.model_file = args.model_path
        if args.device:
            config.compute.device = args.device
        if args.compile:
            config.performance.compile_model = True
        if args.quantize:
            config.performance.use_quantization = True
            
        # Override generation settings
        config.generation.max_length = args.max_length
        config.generation.temperature = args.temperature
        config.generation.top_p = args.top_p
        
        # Setup logging
        setup_logging(config.logging)
        
        # Initialize chat
        chat = ChatInterface(config)
        chat.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def finetune_main():
    """Main entry point for llm-finetune command."""
    parser = argparse.ArgumentParser(description="Finetune MinillM")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--data-path", "-d", required=True, help="Training data path")
    parser.add_argument("--output-dir", "-o", default="./checkpoints", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with CLI args
        if args.wandb:
            config.logging.use_wandb = True
            
        # Setup logging
        setup_logging(config.logging)
        
        # Initialize trainer
        from .training import Trainer
        trainer = Trainer(config, args.data_path, args.output_dir)
        
        # Start training
        trainer.train(resume_from=args.resume)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def serve_main():
    """Main entry point for llm-serve command."""
    parser = argparse.ArgumentParser(description="Serve MinillM via web API")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override server config
        config.server.host = args.host
        config.server.port = args.port
        config.server.workers = args.workers
        
        # Start server
        from .server import start_server
        start_server(config)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


class ChatInterface:
    """Interactive chat interface for MinillM."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Setup device
        self.device = setup_device(config.compute.device)
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = TokenizerManager(config.paths, config.tokens)
        if not self.tokenizer.validate_tokenizer():
            raise RuntimeError("Tokenizer validation failed")
        
        # Load model
        print("Loading model...")
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from transformer_model_llama_june2025 import TransformerModel as OriginalTransformerModel
        
        self.model = OriginalTransformerModel(
            ntokens=config.model.vocab_size,
            max_seq_len=config.model.max_seq_len,
            emsize=-1,
            nhead=config.model.n_heads,
            nlayers=config.model.n_layers,
            ffn_dim=config.model.ffn_dim,
            dim=config.model.dim,
            batch_size=config.model.max_batch_size,
            device=str(self.device)
        ).to(self.device)
        
        self.model.eval()
        
        # Load state dict
        state_dict = torch.load(config.paths.model_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Initialize generator
        self.generator = TextGenerator(self.model, self.tokenizer, config.generation)
        
        # Conversation history
        self.conversation = []
        
        print("Ready! Type your message (press Enter on empty line to start new conversation)")
        print("Type 'quit' or 'exit' to quit\n")
    
    def _apply_optimizations(self):
        """Apply performance optimizations to the model."""
        # Set to evaluation mode
        self.model.eval()
        
        # Quantization
        if self.config.performance.use_quantization:
            print("Applying quantization...")
            try:
                from .optimization import quantize_model
                self.model = quantize_model(self.model, self.config.performance.quantization_bits)
            except ImportError:
                print("Warning: Quantization not available, skipping...")
        
        # Torch compile
        if self.config.performance.compile_model:
            print("Compiling model...")
            try:
                self.model.compile_model()
            except Exception as e:
                print(f"Warning: Model compilation failed: {e}")
    
    def run(self):
        """Run the interactive chat loop."""
        try:
            while True:
                # Get user input
                user_input = input(">> ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input == "":
                    # Start new conversation
                    self.conversation = []
                    self.model.clear_kv_cache()
                    print("Started new conversation\n")
                    continue
                
                # Add user message to conversation
                self.conversation.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Generate response
                print("Thinking...", end="", flush=True)
                response = self.generator.generate_response(self.conversation)
                print(f"\r{response}\n")
                
                # Add assistant response to conversation
                self.conversation.append({
                    'role': 'assistant',
                    'content': response
                })
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            print(f"\nError during chat: {e}")


if __name__ == "__main__":
    # Determine which command to run based on script name
    script_name = Path(sys.argv[0]).name
    
    if script_name == "llm-chat" or "chat" in script_name:
        chat_main()
    elif script_name == "llm-finetune" or "finetune" in script_name:
        finetune_main()
    elif script_name == "llm-serve" or "serve" in script_name:
        serve_main()
    else:
        print("Unknown command. Use llm-chat, llm-finetune, or llm-serve")
        sys.exit(1)