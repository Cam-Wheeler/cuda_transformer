import pytest
import torch

from config import QWEN3Config
from models.pytorch_transformer import (
    QWEN3,
    QWEN3Block,
    QWEN3FFN,
    QWEN3GQAAttention,
    QWEN3LMHead,
    QWEN3RMSNorm,
    QWEN3RoPE,
)


@pytest.fixture
def tiny_config():
    return QWEN3Config(
        vocab_size=32,
        context_length=16,
        embedding_dim=64,
        fnn_hidden_dim=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        num_layers=2,
        tie_embeddings=True,
    )


class TestQWEN3FFN:
    def test_forward_shape(self):
        ffn = QWEN3FFN(embedding_dim=64, hidden_dim=128)
        x = torch.randn(2, 8, 64)
        assert ffn(x).shape == (2, 8, 64)


class TestQWEN3RMSNorm:
    def test_forward_shape(self):
        rmsnorm = QWEN3RMSNorm(hidden_dim=64)
        x = torch.randn(2, 8, 64)
        assert rmsnorm(x).shape == (2, 8, 64)


class TestQWEN3RoPE:
    def test_forward_shape(self, tiny_config):
        rope = QWEN3RoPE()
        cos, sin = QWEN3RoPE.compute_rope_parameters(
            tiny_config.head_dim,
            max_context_len=tiny_config.context_length,
        )
        x = torch.randn(2, 4, 8, tiny_config.head_dim)
        assert rope(x, cos, sin).shape == x.shape


class TestQWEN3GQAAttention:
    def test_forward_shape(self, tiny_config):
        attn = QWEN3GQAAttention(
            input_dim=tiny_config.embedding_dim,
            num_q_heads=tiny_config.num_q_heads,
            num_kv_heads=tiny_config.num_kv_heads,
            head_dim=tiny_config.head_dim,
        )
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, tiny_config.embedding_dim)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        cos, sin = QWEN3RoPE.compute_rope_parameters(
            tiny_config.head_dim,
            max_context_len=tiny_config.context_length,
        )
        assert attn(x, mask, cos, sin).shape == x.shape


class TestQWEN3LMHead:
    def test_forward_shape(self, tiny_config):
        head = QWEN3LMHead(tiny_config.embedding_dim, tiny_config.vocab_size)
        x = torch.randn(2, 8, tiny_config.embedding_dim)
        assert head(x).shape == (2, 8, tiny_config.vocab_size)


class TestQWEN3Block:
    def test_forward_shape(self, tiny_config):
        block = QWEN3Block(tiny_config)
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, tiny_config.embedding_dim)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        cos, sin = QWEN3RoPE.compute_rope_parameters(
            tiny_config.head_dim,
            max_context_len=tiny_config.context_length,
        )
        assert block(x, mask, cos, sin).shape == x.shape


class TestQWEN3:
    def test_forward_shape(self, tiny_config):
        model = QWEN3(tiny_config)
        token_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        logits = model(token_ids)
        assert logits.shape == (2, 8, tiny_config.vocab_size)

    def test_tied_embeddings(self, tiny_config):
        model = QWEN3(tiny_config)
        assert id(model.lm_head.proj.weight) == id(model.embedding_layer.weight)
