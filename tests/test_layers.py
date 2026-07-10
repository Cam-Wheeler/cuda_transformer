import pytest
import torch

# Locals
from models.pytorch_transformer import QWEN3FFN

class TestQWEN3FFN:
    def test_forward_shape(self):
        ffn = QWEN3FFN(
            embedding_dim=1024,
            hidden_dim=3072
        )
        x = torch.randn(1, 1024)
        y = ffn(x)
        assert y.shape == (1, 1024)