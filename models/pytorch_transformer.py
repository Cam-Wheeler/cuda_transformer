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
    - Hidden activation: SiLU
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
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

# Local imports
from config import QWEN3Config


class QWEN3(nn.Module):
    """
    A QWEN-3 style transformer model.

    Arguments:
    - config: The configuration for the model.

    Returns:
    - x: The output tensor.

    Architecture:
    - Embedding layer
    - Transformer blocks * num_layers
    - RMSNorm
    - LM head
    """

    def __init__(self, config: QWEN3Config):
        super().__init__()
        self.config = config
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [QWEN3Block(config) for _ in range(config.num_layers)]
        )
        self.norm = QWEN3RMSNorm(config.embedding_dim)
        self.lm_head = QWEN3LMHead(config.embedding_dim, config.vocab_size)
        if config.tie_embeddings:
            self.lm_head.proj.weight = (
                self.embedding_layer.weight
            )  # Tie the embedding layer and head layer together.
        cos, sin = QWEN3RoPE.compute_rope_parameters(
            config.head_dim,
            max_context_len=config.context_length,
            # We will use the RoPE deafult for theta_base.
        )
        # Register the buffers to torch (as currently cos and sin are on the cpu).
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Arguments:
        - x: The input tensor token ids.

        Returns:
        - x: The output tensor.
        """

        x = self.embedding_layer(
            x
        )  # embed the tokens into vectors [batch_size, seq_len, embedding_dim]
        seq_len = x.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )  # Generate the causal mask for causal attention.

        for block in self.transformer_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.norm(x)  # [batch_size, seq_len, embedding_dim]
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        return logits


class QWEN3RoPE(nn.Module):
    """RoPE embedding for the model."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_rope_parameters(
        head_dim: int,
        theta_base: float = 1_000_000.0,
        max_context_len: int = 40_960,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precomputes the Cos and Sin values for RoPE.

        Called once at model initialisation then stored for reuse!

        Arguments:
        - head_dim: The dimension of the head.
        - theta_base: The base of the theta value.
        - max_context_len: The maximum context length.
        - dtype: The dtype of the tensors.

        Returns:
        - cos: The cosine values for the RoPE.
        - sin: The sine values for the RoPE.
        """
        assert head_dim % 2 == 0, (
            "Head dimension must be even for RoPE so we can compute pairs."
        )

        # Compute the thetas for each pair! theta = base ** (-2i / head_dim)
        # Again we do some reciprocal trickery to get the inverse frequencies!
        inverse_freqs = 1.0 / (
            theta_base
            ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype) / head_dim)
        ).unsqueeze(0)  # [1, head_dim // 2]
        absolute_pos = torch.arange(max_context_len, dtype=dtype).unsqueeze(
            1
        )  # [max_context_len, 1]
        angles = absolute_pos * inverse_freqs  # [max_context_len, head_dim // 2]

        # Because we are using rotate half trick.
        angles = torch.cat([angles, angles], dim=1)  # [max_context_len, head_dim]

        # Now we compute the cos and sin values for each pair!
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Actually spin the vectors!

        Arguments:
        - x: The input tensor. [batch_size, n_heads, seq_len, head_dim]
        - cos: The cosine values for the RoPE.
        - sin: The sine values for the RoPE.

        Returns:
        - x: The output tensor. [batch_size, heads, seq_len, head_dim]
        """
        batch_size, n_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, (
            "Head dimension must be even for RoPE so we can spin pairs."
        )

        # Split the head dim into two halves.
        x_1 = x[..., : head_dim // 2]
        x_2 = x[..., head_dim // 2 :]

        # Original cos and sin shapes are [max_context_len, head_dim]
        # We need to reshape them to [1, 1, max_context_len, head_dim]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        #  RoPE with half trick!
        flipped = torch.cat((-x_2, x_1), dim=-1)
        rotated = (x * cos) + (flipped * sin)

        return rotated


class QWEN3Block(nn.Module):
    """
    A single QWEN-3 decoder block.
    This block consists of:

    - RMSNorm 1
    - Masked GQA Attention (ROPE and QK-Norm included)
    - Res connection 1 addition.
    - RMSNorm 2
    - FFN
    - Res connection 2 addition.
    """

    def __init__(self, config: QWEN3Config) -> None:
        super().__init__()
        self.norm_1 = QWEN3RMSNorm(config.embedding_dim)
        self.group_query_attn = QWEN3GQAAttention(
            input_dim=config.embedding_dim,
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        )
        self.norm_2 = QWEN3RMSNorm(config.embedding_dim)
        self.ffn = QWEN3FFN(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.fnn_hidden_dim,
            bias=False,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the QWEN-3 block.

        Arguments:
        - x: The input tensor.
        - mask: The mask tensor for causal attention.
        - cos: The cosine values for the RoPE.
        - sin: The sine values for the RoPE.

        Returns:
        - x: The output tensor.
        """
        residual = x  # [batch, seq_len, embed_dim]
        x = self.norm_1(x)
        x = self.group_query_attn(x, mask, cos, sin)  # [batch_size, seq_len, embed_dim]
        x = x + residual  # Elementwise addition.

        residual = x
        x = self.norm_2(x)
        x = self.ffn(x)  # SwiGLU
        x = x + residual

        return x


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

    def __init__(
        self, hidden_dim: int, eps: float = 1e-06, fp32_stability: bool = True
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))  # g in the paper.
        self.epsilon = eps
        self.fp32_stability = fp32_stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_type = x.dtype
        if self.fp32_stability:  # HF always cast for stability.
            x = x.to(torch.float32)
        mean_squared = x.pow(2).mean(
            -1, keepdim=True
        )  # We grab the means across the rows (tokens).
        sqrt_norm = x * torch.rsqrt(
            mean_squared + self.epsilon
        )  # Now we take the rsqrt (not sqrt) and add epsilon.
        scaled = sqrt_norm * self.scale  # Now we scale the input.
        return scaled.type(original_type)


class QWEN3GQAAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    Arguments:
    - input_dim: The dimension of the input tensor.
    - num_q_heads: The number of query heads.
    - num_kv_heads: The number of key/value heads.
    - head_dim: The dimension of each head.
    - dtype: The dtype of the tensors.

    Returns:
    - x: The output tensor.
    """

    def __init__(
        self,
        input_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Setup dimensions.
        self.input_dim = (
            input_dim  # embedding dimension (the size of the token vectors).
        )
        self.head_dim = head_dim  # dimension of each Q head.
        self.num_q_heads = num_q_heads  # number of query heads.
        self.q_dim = self.num_q_heads * self.head_dim  # dimension of all Q heads.
        self.num_kv_heads = num_kv_heads  # number of key/value heads.
        self.kv_dim = self.num_kv_heads * self.head_dim  # dimension of all KV heads.
        self.kv_group_size = (
            self.num_q_heads // self.num_kv_heads
        )  # number of Q heads per KV group.
        self.dtype = dtype  # dtype of the tensors.

        # Check dimensions. (this is slightly overkill but I want to ensure I am correct).
        assert self.num_q_heads > self.num_kv_heads, (
            "Number of Q heads must be greater than number of KV heads for GQA."
        )
        assert self.q_dim > self.kv_dim, (
            "Q dimension must be larger than KV dimension, this is the whole point of GQA."
        )
        assert self.num_q_heads % self.num_kv_heads == 0, (
            "num_q_heads must be divisible by num_kv_heads for GQA grouping."
        )

        # Weights for projections.
        self.q_proj = nn.Linear(
            self.input_dim, self.q_dim, bias=False, dtype=self.dtype
        )  # Project into q space. [input_dim, q_dim]
        self.k_proj = nn.Linear(
            self.input_dim, self.kv_dim, bias=False, dtype=self.dtype
        )  # Project into kv space. [input_dim, kv_dim]
        self.v_proj = nn.Linear(
            self.input_dim, self.kv_dim, bias=False, dtype=self.dtype
        )  # Project into kv space. [input_dim, kv_dim]

        # QK-Norm
        self.q_norm = QWEN3RMSNorm(
            self.head_dim
        )  # We have a parameter for the head dimension.
        self.k_norm = QWEN3RMSNorm(self.head_dim)  # ^^^^^^^

        # RoPE
        self.rope = QWEN3RoPE()  # No parameters needed!

        # Output projection
        self.out_proj = nn.Linear(
            self.q_dim, self.input_dim, bias=False, dtype=self.dtype
        )  # [q_dim, input_dim]

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the GQA attention.

        Arguments:
        - x: The input tensor.
        - mask: The mask tensor for causal attention.
        - cos: The cosine values for the RoPE.
        - sin: The sine values for the RoPE.

        Returns:
        - x: The output tensor.
        """
        batch_size, seq_len, embedding_dim = (
            x.shape
        )  # embedding_dim not used, just named for clarity.

        # Projections
        queries = self.q_proj(x)  # [batch_size, seq_len, q_dim]
        keys = self.k_proj(x)  # [batch_size, seq_len, kv_dim]
        values = self.v_proj(x)  # [batch_size, seq_len, kv_dim]

        # Reshape the projections to [batch_size, n_heads, seq_len, head_dim]
        # This is becuase we want to separate the heads into their own dimension so we can work with them.
        # View just reshapes the tensor. Transpose swaps the dims of seq_len and n_heads.
        # original shape: (batch_size, seq_len, head_dim) -> view (batch_size, seq_len, n_heads, head_dim) -> transpose (batch_size, n_heads, seq_len, head_dim)
        queries = queries.view(
            batch_size, seq_len, self.num_q_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # QK Norm
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # RoPE, spinnnnnnnnnnnnnnn.
        queries = self.rope(queries, cos, sin)
        keys = self.rope(keys, cos, sin)

        # Expand K and V to match Q.
        # repeat_interleave will duplicate the heads like so (0, 1, 2) -> (0, 0, 1, 1, 2, 2)
        # Shapes will now match queries.
        keys = keys.repeat_interleave(self.kv_group_size, dim=1)
        values = values.repeat_interleave(self.kv_group_size, dim=1)

        # Attention babaaayyyyy!
        attention_scores = (
            queries @ keys.transpose(-2, -1)
        )  # Transpose the seq_len, head_dim tensor, output is (batch_size, n_heads, seq_len, seq_len)
        masked_scores = attention_scores.masked_fill(
            mask, -torch.inf
        )  # Mask out the future tokens.
        attention_weights = F.softmax(
            masked_scores / self.head_dim**0.5, dim=-1
        )  # Softmax over the columns (each row gets a softmax).

        # Output
        # We dot scaled and values making [batch_size, n_heads, seq_len, head_dim]
        # Transpose the n_heads and seq_len dims, then reshape to [batch_size, seq_len, q_dim] (OG shape after projections).
        attention_output = (
            (attention_weights @ values)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.q_dim)
        )

        # Output projection back to input dim, input shape to attention == output shape from attention.
        return self.out_proj(
            attention_output
        )  # [batch_size, seq_len, q_dim] -> [batch_size, seq_len, embedding_dim]


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
    """The LM head for the model."""

    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
