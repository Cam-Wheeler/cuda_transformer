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

    - RMSNorm 1
    - Masked GQA Attention (ROPE and QK-Norm included)
    - Res connection 1
    - RMSNorm 2
    - FFN
    - Res connection 2
    """
    pass

class QWEN3RMSNorm(nn.Module):
    """
    Root Mean Square Normalisation (RMSNorm)

    Arguments:
    - hidden_dim: The dimension of the hidden layer.
    - eps: The epsilon value for the normalisation.
    - bias: Whether to use a bias term.
    - fp32_stability: Whether to use fp32 stability (hf code uses this)
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-06, fp32_stability: bool = True):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim)) # g in the paper.
        self.epsilon = eps
        self.fp32_stability = fp32_stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_type = x.dtype
        if self.fp32_stability: # HF always cast for stability.
            x = x.to(torch.float32)
        mean_squared = x.pow(2).mean(-1, keepdim=True) # We grab the means across the rows (tokens).
        sqrt_norm = x  * torch.rsqrt(mean_squared + self.epsilon) # Now we take the rsqrt (not sqrt) and add epsilon.
        scaled = sqrt_norm * self.scale # Now we scale the input.
        return scaled.type(original_type)

class QWEN3GQAAttention(nn.Module):
    """
    Grouped Query Attention (GQA)    
    """
    pass

class QWEN3FFN(nn.Module):
    """
    Feed-Forward Network (FFN) using GLU with SiLU activation.

    Arguments:
    - embedding_dim: The dimension of the embedding (output of the transformer block)
    - hidden_dim: The dimension of the hidden layer.
    - bias: Whether to use a bias term.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, embedding_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class QWEN3LMHead(nn.Module):
    """ The LM head for the model."""
    pass
