"""
Configuration management for MinillM using Pydantic.
Provides type-safe configuration loading and validation.
"""

from typing import Optional, List, Literal
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    
    dim: int = Field(default=512, description="Hidden dimension")
    ffn_dim: int = Field(default=1536, description="Feed-forward network dimension")
    n_layers: int = Field(default=16, description="Number of transformer layers")
    n_heads: int = Field(default=16, description="Number of attention heads")
    n_kv_heads: int = Field(default=16, description="Number of key-value heads for GQA")
    vocab_size: int = Field(default=50000, description="Vocabulary size")
    max_seq_len: int = Field(default=128, description="Maximum sequence length")
    norm_eps: float = Field(default=1e-5, description="RMS normalization epsilon")
    rope_theta: float = Field(default=50000.0, description="RoPE base frequency")
    max_batch_size: int = Field(default=4, description="Maximum batch size for KV cache")
    
    @property
    def head_dim(self) -> int:
        """Calculate head dimension."""
        return self.dim // self.n_heads
    
    @property
    def n_kv_head_rep(self) -> int:
        """Calculate KV head repetition for GQA."""
        return self.n_heads // self.n_kv_heads
    
    @validator('n_heads')
    def validate_n_heads(cls, v, values):
        """Ensure dim is divisible by n_heads."""
        if 'dim' in values and values['dim'] % v != 0:
            raise ValueError(f"dim ({values['dim']}) must be divisible by n_heads ({v})")
        return v
    
    @validator('n_kv_heads')
    def validate_n_kv_heads(cls, v, values):
        """Ensure n_heads is divisible by n_kv_heads."""
        if 'n_heads' in values and values['n_heads'] % v != 0:
            raise ValueError(f"n_heads ({values['n_heads']}) must be divisible by n_kv_heads ({v})")
        return v


class PathsConfig(BaseModel):
    """File path configuration."""
    
    model_file: str = Field(description="Path to model checkpoint file")
    tokenizer_dir: str = Field(description="Directory containing tokenizer files")
    vocab_file: str = Field(default="tokenizer_50k_2025-vocab.json", description="Vocabulary file name")
    merges_file: str = Field(default="tokenizer_50k_2025-merges.txt", description="Merges file name")
    
    @property
    def vocab_path(self) -> str:
        """Full path to vocabulary file."""
        return str(Path(self.tokenizer_dir) / self.vocab_file)
    
    @property
    def merges_path(self) -> str:
        """Full path to merges file."""
        return str(Path(self.tokenizer_dir) / self.merges_file)


class TokensConfig(BaseModel):
    """Special token configuration."""
    
    pad_token: int = Field(default=0, description="Padding token ID")
    question_end_token: int = Field(default=1, description="Question end token ID")
    answer_end_token: int = Field(default=2, description="Answer end token ID")
    think_start_token: int = Field(default=7, description="Think start token ID")
    think_end_token: int = Field(default=8, description="Think end token ID")


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    batch_size: int = Field(default=1, description="Training batch size")
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    weight_decay: float = Field(default=0.1, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping norm")
    warmup_steps: int = Field(default=1000, description="Warmup steps")
    max_steps: int = Field(default=100000, description="Maximum training steps")
    save_steps: int = Field(default=5000, description="Save checkpoint every N steps")
    eval_steps: int = Field(default=1000, description="Evaluate every N steps")
    logging_steps: int = Field(default=100, description="Log every N steps")
    dataloader_num_workers: int = Field(default=4, description="Dataloader workers")
    
    # Optimization
    optimizer: Literal["adamw", "adam", "sgd"] = Field(default="adamw", description="Optimizer type")
    scheduler: Literal["cosine", "linear", "constant"] = Field(default="cosine", description="Learning rate scheduler")
    gradient_accumulation_steps: int = Field(default=8, description="Gradient accumulation steps")
    gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")
    mixed_precision: Literal["no", "fp16", "bf16"] = Field(default="bf16", description="Mixed precision training")
    
    # Regularization
    dropout: float = Field(default=0.01, description="Dropout probability")
    attention_dropout: float = Field(default=0.0, description="Attention dropout probability")


class GenerationConfig(BaseModel):
    """Text generation configuration."""
    
    max_length: int = Field(default=100, description="Maximum generation length")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p nucleus sampling")
    top_k: int = Field(default=50, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Use sampling vs greedy decoding")


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""
    
    compile_model: bool = Field(default=True, description="Use torch.compile")
    use_flash_attention: bool = Field(default=False, description="Use FlashAttention")
    use_quantization: bool = Field(default=False, description="Use quantization")
    quantization_bits: int = Field(default=8, description="Quantization bits (4 or 8)")
    use_gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")


class ComputeConfig(BaseModel):
    """Compute and device configuration."""
    
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(default="auto", description="Device to use")
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = Field(default="auto", description="Data type")
    num_threads: Optional[int] = Field(default=None, description="Number of threads (None for auto)")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: str = Field(default="minillm", description="W&B project name")
    use_tensorboard: bool = Field(default=False, description="Use TensorBoard")
    log_dir: str = Field(default="./logs", description="Logging directory")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    
    eval_datasets: List[str] = Field(default_factory=list, description="Evaluation datasets")
    eval_batch_size: int = Field(default=8, description="Evaluation batch size")


class ServerConfig(BaseModel):
    """Web server configuration."""
    
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    max_concurrent_requests: int = Field(default=10, description="Max concurrent requests")


class Config(BaseModel):
    """Complete MinillM configuration."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathsConfig
    tokens: TokensConfig = Field(default_factory=TokensConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file with validation."""
    try:
        return Config.from_yaml(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")