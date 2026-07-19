"""
Configuration for the QWEN-3 Style Transformer model.
"""

from pydantic import BaseModel

class QWEN3_MINI_Config(BaseModel):
    """
    Configuration for a model smaller than the 0.6B model.

    Will be used for prototyping and first training runs.

    Currently the values are just place holders, these will be changed later!
    """
    vocab_size: int = 151_643 # This is the size of the tokenizer vocabulary.
    context_length: int = 256 # Reduced massively for tiny stories.
    embedding_dim: int = 1024 # Dimension of the token embeddings (used throughout the model really).
    fnn_hidden_dim: int = 3072 # This hidden dim for the FNN.
    num_q_heads: int = 16 # Number of Q heads.
    num_kv_heads: int = 8 # 8 KV heads used for 16 Q heads.
    head_dim: int = 128 # Dimension of the heads.
    num_layers: int = 12 # Number of layers.
    tie_embeddings: bool = True # Use tied embeddings

class QWEN3_06B_Config(BaseModel):
    """
    Configuration for the QWEN-3 Style Transformer model.
    """
    vocab_size: int = 151_643 # This is the size of the tokenizer vocabulary.
    context_length: int = 40_960 # This might need to be adjusted becuase we are training from scratch.
    embedding_dim: int = 1024 # Dimension of the token embeddings (used throughout the model really).
    fnn_hidden_dim: int = 3072 # This hidden dim for the FNN.
    num_q_heads: int = 16 # Number of Q heads.
    num_kv_heads: int = 8 # 8 KV heads used for 16 Q heads.
    head_dim: int = 128 # Dimension of the heads.
    num_layers: int = 28 # Number of layers.
    tie_embeddings: bool = True # Use tied embeddings

class MiniTrainerConfig(BaseModel):
    """
    Configuration for the tiny training run.
    """
    total_iterations: int = 600
    gradient_accumulation_steps: int = 4
    warmup_iters: int = 100
    learning_rate: float = 6e-4 # used in Kaparthy's nanogpt.
    beta1: float = 0.9
    beta2: float = 0.95
    eval_interval: int = 100
    resume: bool = False
    wandb_log: bool = True
    root_save_path: str = "/data/mini"
    device: str = "cuda"
    log_interval: int = 50
    wandb_entity: str = "camwheeler135-university-of-edinburgh"
    wandb_project: str = "cuda-transformer"

class StandardTrainerConfig(BaseModel):
    """
    Configuration for training.
    """
    total_iterations: int = 600000 
    wandb_log: bool = True
    gradient_accumulation_steps: int = 4
    warmup_iters: int = 1000
    learning_rate: float = 6e-4 # used in Kaparthy's nanogpt.
    beta1: float = 0.9
    beta2: float = 0.95
    eval_interval: int = 100
    resume: bool = False
    root_save_path: str = "/data/standard"
    device: str = "cuda"
    log_interval: int = 50
    wandb_entity: str = "camwheeler135-university-of-edinburgh"
    wandb_project: str = "cuda-transformer"


