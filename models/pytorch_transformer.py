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

We make use of many more papers as well:

- GQA: https://arxiv.org/abs/2305.13245
- RoPE: https://arxiv.org/abs/2104.09864
- RMSNorm: https://arxiv.org/abs/1910.07467
- SWIGLU: https://arxiv.org/abs/2002.05202

I also used several other resources to help me understand and build!

- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse
"""

import torch
from torch import nn
from torch.nn import functional as F


# Local imports
from config import QWEN3Config


class QWEN3(nn.Module):
    """ A QWEN-3 style transformer model."""

    def __init__(self, config: QWEN3Config):
        super().__init__()
        self.config = config
        pass

class QWEN3Embedding(nn.Module):
    """ The embedding layer for the model."""
    pass

class QWEN3RoPE(nn.Module):
    """ RoPE embedding for the model."""
    pass

class QWEN3Block(nn.Module):
    """ 
    A single QWEN-3 decoder block.
    This block consists of:
    - RMSNorm
    - GQA self-attention (RoPE included)
    - RMSNorm
    - MLP (SWIGLU)
    """
    pass

class QWEN3RMSNorm(nn.Module):
    """ The RMSNorm for the model."""
    pass

class QWEN3GQAAttention(nn.Module):
    """ The GQA self-attention for the model."""
    pass

class QWEN3MLP(nn.Module):
    """ The MLP layer for the model. This uses SWIGLU."""
    pass

class QWEN3LMHead(nn.Module):
    """ The LM head for the model."""
    pass
