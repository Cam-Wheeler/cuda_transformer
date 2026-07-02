"""
Base Pytorch implementation of QWEN-3 Style Transformer model.

Architecture Overview:
    - QWEN-3 style transformer
    - 0.6B parameters
    - Grouped Query Attention (GQA)
    - Rotary Position Embedding (RoPE)
    - RMSNorm with pre-normalisation
    - QK-Norm
    - Tie embeddings
    - Casual masking
    - We will use the QWEN-3 tokeniser to tokenise the text.

Model Architecture:
    - 28 Layers
    - Heads (Q / KV): 16/8
    - Head dim: 128
    - Hidden activation: SELU
    - Hidden dim: 1024
    - Num Q heads: 16
    - Num KV heads: 8
    - RMS Norms EPS: 1e-06
    - ROPE Theta: 1000000

This model is based on the QWEN-3 paper: https://arxiv.org/abs/2505.09388
"""

import torch
from torch import nn

# Local imports
from config import QWEN3Config


class QWEN3(nn.Module):
    
    def __init__(self, config: QWEN3Config):
        super().__init__()
        self.config = config
        pass