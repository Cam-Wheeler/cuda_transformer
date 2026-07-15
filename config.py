"""
Configuration for the QWEN-3 Style Transformer model.
"""

from pydantic import BaseModel

class QWEN3Config(BaseModel):
    """
    Configuration for the QWEN-3 Style Transformer model.
    """
    vocab_size: int = 151_936 # This is the size of the tokenizer vocabulary.
    context_length: int = 40_960 # This might need to be adjusted becuase we are training from scratch.
    embedding_dim: int = 1024 # Dimension of the token embeddings (used throughout the model really).
    fnn_hidden_dim: int = 3072 # This hidden dim for the FNN.
    num_q_heads: int = 16 # Number of Q heads.
    num_kv_heads: int = 8 # 8 KV heads used for 16 Q heads.
    head_dim: int = 128 # Dimension of the heads.
    num_layers: int = 28 # Number of layers.
    tie_embeddings: bool = True # Use tied embeddings

class TrainingConfig(BaseModel):
    """
    Configuration for training.
    """
    pass


