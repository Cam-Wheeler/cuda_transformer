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
    fnn_hidden_dim: int = 3071 # This hidden dim for the FNN.
    num_layers: int = 28 # Number of layers.
    num_q_heads: int = 16 # Number of Q heads.
    num_kv_heads: int = 8 # 8 KV heads used for 16 Q heads.
    head_dim: int = 128 # Dimension of the heads.
    rms_norms_eps: float = 1e-06 # Epsilon for the RMS norms.
    rope_theta: float = 1_000_000.0 # The theta for the RoPE.

class TrainingConfig(BaseModel):
    """
    Configuration for training.
    """
    pass


