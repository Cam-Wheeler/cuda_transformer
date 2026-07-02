"""
Configuration for the QWEN-3 Style Transformer model.
"""

from pydantic import BaseModel

class QWEN3Config(BaseModel):
    """
    Configuration for the QWEN-3 Style Transformer model.
    """

    num_layers: int = 28
    num_q_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 1024
    rms_norms_eps: float = 1e-06
    rope_theta: float = 1000000