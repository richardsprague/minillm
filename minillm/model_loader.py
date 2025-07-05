"""
Model loader system for MinillM.
Supports local models, Hugging Face models, and API-based models.
"""

import os
import sys
import torch
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod

from .config import ModelSourceConfig, Config
from .utils import setup_device


class ModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = setup_device(config.compute.device)
    
    @abstractmethod
    def load_model(self, model_source: ModelSourceConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer. Returns (model, tokenizer)."""
        pass
    
    @abstractmethod
    def generate(self, model: Any, tokenizer: Any, prompt: str, **kwargs) -> str:
        """Generate text using the loaded model."""
        pass


class LocalModelLoader(ModelLoader):
    """Loader for local MinillM models."""
    
    def load_model(self, model_source: ModelSourceConfig) -> Tuple[Any, Any]:
        """Load local MinillM model."""
        if model_source.type != "local":
            raise ValueError(f"Expected local model, got {model_source.type}")
        
        # Import the original transformer model
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from transformer_model_llama_june2025 import TransformerModel as OriginalTransformerModel
        
        # Load tokenizer
        from .tokenizer import TokenizerManager
        tokenizer = TokenizerManager(self.config.paths, self.config.tokens)
        
        # Create model
        model = OriginalTransformerModel(
            ntokens=self.config.model.vocab_size,
            max_seq_len=self.config.model.max_seq_len,
            emsize=-1,
            nhead=self.config.model.n_heads,
            nlayers=self.config.model.n_layers,
            ffn_dim=self.config.model.ffn_dim,
            dim=self.config.model.dim,
            batch_size=self.config.model.max_batch_size,
            device=str(self.device)
        ).to(self.device)
        
        # Load weights
        if model_source.path:
            state_dict = torch.load(model_source.path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        
        # Apply optimizations
        if self.config.performance.compile_model:
            model = torch.compile(model)
        
        return model, tokenizer
    
    def generate(self, model: Any, tokenizer: Any, prompt: str, **kwargs) -> str:
        """Generate text using local model."""
        from .generation import TextGenerator
        
        # Create generator
        generator = TextGenerator(model, tokenizer, self.config.generation)
        
        # Generate response
        conversation = [{'role': 'user', 'content': prompt}]
        return generator.generate_response(conversation, **kwargs)


class HuggingFaceModelLoader(ModelLoader):
    """Loader for Hugging Face models."""
    
    def load_model(self, model_source: ModelSourceConfig) -> Tuple[Any, Any]:
        """Load Hugging Face model."""
        if model_source.type != "huggingface":
            raise ValueError(f"Expected huggingface model, got {model_source.type}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers package is required for Hugging Face models")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_source.name,
            trust_remote_code=model_source.trust_remote_code
        )
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_source.name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None,
            trust_remote_code=model_source.trust_remote_code
        )
        
        if self.device.type != 'cuda':
            model = model.to(self.device)
        
        model.eval()
        
        return model, tokenizer
    
    def generate(self, model: Any, tokenizer: Any, prompt: str, **kwargs) -> str:
        """Generate text using Hugging Face model."""
        # Prepare input
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.size(1) + kwargs.get('max_length', 100),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                do_sample=kwargs.get('temperature', 0.7) > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=kwargs.get('repetition_penalty', 1.1)
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True)
        return response.strip()


class OpenAIModelLoader(ModelLoader):
    """Loader for OpenAI API models."""
    
    def load_model(self, model_source: ModelSourceConfig) -> Tuple[Any, Any]:
        """Load OpenAI API client."""
        if model_source.type != "openai":
            raise ValueError(f"Expected openai model, got {model_source.type}")
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI models")
        
        # Create client
        client = OpenAI(
            api_key=model_source.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=model_source.api_base
        )
        
        # Return client as both model and tokenizer
        return client, client
    
    def generate(self, model: Any, tokenizer: Any, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            response = model.chat.completions.create(
                model=self.config.model_source.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_length', 100),
                top_p=kwargs.get('top_p', 0.9),
                frequency_penalty=kwargs.get('repetition_penalty', 1.1) - 1.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class ModelManager:
    """Manages multiple model loaders."""
    
    def __init__(self, config: Config):
        self.config = config
        self.loaders = {
            "local": LocalModelLoader(config),
            "huggingface": HuggingFaceModelLoader(config),
            "openai": OpenAIModelLoader(config)
        }
        self.current_model = None
        self.current_tokenizer = None
        self.current_loader = None
        self.current_source = None
    
    def load_model(self, model_source: ModelSourceConfig) -> Tuple[Any, Any]:
        """Load a model from the specified source."""
        if model_source.type not in self.loaders:
            raise ValueError(f"Unsupported model type: {model_source.type}")
        
        loader = self.loaders[model_source.type]
        model, tokenizer = loader.load_model(model_source)
        
        # Update current state
        self.current_model = model
        self.current_tokenizer = tokenizer
        self.current_loader = loader
        self.current_source = model_source
        
        return model, tokenizer
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the current model."""
        if self.current_model is None or self.current_loader is None:
            raise RuntimeError("No model loaded")
        
        return self.current_loader.generate(
            self.current_model,
            self.current_tokenizer,
            prompt,
            **kwargs
        )
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        return [
            {
                "name": model.name,
                "type": model.type,
                "description": f"{model.type.title()} model: {model.name}"
            }
            for model in self.config.available_models
        ]
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        for model_source in self.config.available_models:
            if model_source.name == model_name:
                try:
                    self.load_model(model_source)
                    return True
                except Exception as e:
                    print(f"Failed to switch to model {model_name}: {e}")
                    return False
        
        print(f"Model {model_name} not found in available models")
        return False
    
    def switch_model_source(self, model_source: 'ModelSourceConfig') -> bool:
        """Switch to a model using a ModelSourceConfig."""
        try:
            self.load_model(model_source)
            return True
        except Exception as e:
            print(f"Failed to switch to model {model_source.name}: {e}")
            return False
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.current_source is None:
            return {"name": "None", "type": "None", "loaded": False}
        
        return {
            "name": self.current_source.name,
            "type": self.current_source.type,
            "loaded": True,
            "path": self.current_source.path if self.current_source.type == "local" else None
        }