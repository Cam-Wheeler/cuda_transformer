"""
Tests for the custom CUDA silu function vs baseline torch
Like element-wise requires CUDA to acutally run the tests.
"""

import pytest
import torch
import torch.nn.functional as F

from wrappers.training import SiLU

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for SiLU tests."
)

@pytest.fixture
def device():
    return torch.device("cuda")

@pytest.fixture
def input_tensor(device):
    x = torch.randn(2, 8, 64, dtype=torch.float32, device=device)
    return x

@pytest.fixture
def silu_op():
    return SiLU()

@requires_cuda
class TestSilu:

    def test_silu_forward(self, input_tensor, silu_op):
        """Simple forward pass"""
        x = input_tensor
        out_cuda = silu_op(x)
        out_torch = F.silu(x)
        torch.testing.assert_close(out_cuda, out_torch)

    def test_silu_backward(self, input_tensor, silu_op):
        """Simple backward pass"""
        x_cuda = input_tensor.detach().requires_grad_(True)
        x_torch = input_tensor.detach().requires_grad_(True)

        silu_op(x_cuda).sum().backward()
        F.silu(x_torch).sum().backward()

        torch.testing.assert_close(x_cuda.grad, x_torch.grad)

