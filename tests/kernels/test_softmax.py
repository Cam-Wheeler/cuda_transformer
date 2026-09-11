"""
Parity tests for custom CUDA Softmax vs torch.
Requires a CUDA GPU and a built `custom_training` extension.
"""

import pytest
import torch
import torch.nn.functional as F

from wrappers.training import Softmax

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)


@pytest.fixture
def device():
    return torch.device("cuda")

@pytest.fixture
def input_tensor(device):
    """float32 tensors on CUDA for testing."""
    return torch.randn(2, 8, 64, dtype=torch.float32, device=device)

@pytest.fixture
def softmax_op():
    return Softmax()


@requires_cuda
class TestSoftmax:

    def test_softmax_forward(self, input_tensor, softmax_op):
        """Test the forward for Softmax"""
        x = input_tensor
        out_cuda = softmax_op(x)
        out_torch = F.softmax(x, dim=-1)
        torch.testing.assert_close(out_cuda, out_torch) # testing values and shape

    def test_softmax_bwd(self, input_tensor, softmax_op):
        """Test the backward for Softmax"""
        x_cuda = input_tensor.detach().clone().requires_grad_(True)
        x_torch = input_tensor.detach().clone().requires_grad_(True)

        # Ones-valued upstream grad (from .sum()) makes softmax input grads
        # identically zero, so use a random grad_out instead.
        grad_out = torch.randn_like(x_cuda)

        softmax_op(x_cuda).backward(grad_out)
        F.softmax(x_torch, dim=-1).backward(grad_out)

        torch.testing.assert_close(x_cuda.grad, x_torch.grad)

    def test_softmax_forward_causal_mask(self, device, softmax_op):
        """Causal -inf must not NaN the row (attention scores)."""
        seq = 16
        x = torch.randn(2, seq, seq, dtype=torch.float32, device=device)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1)
        x = x.masked_fill(mask, float("-inf"))

        out_cuda = softmax_op(x)
        out_torch = F.softmax(x, dim=-1)

        assert torch.isfinite(out_cuda).all()
        torch.testing.assert_close(out_cuda, out_torch)

    def test_softmax_bwd_causal_mask(self, device, softmax_op):
        """Backward through causal -inf rows vs torch."""
        seq = 16
        x = torch.randn(2, seq, seq, dtype=torch.float32, device=device)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1)
        x = x.masked_fill(mask, float("-inf"))

        x_cuda = x.detach().clone().requires_grad_(True)
        x_torch = x.detach().clone().requires_grad_(True)
        grad_out = torch.randn_like(x_cuda)

        softmax_op(x_cuda).backward(grad_out)
        F.softmax(x_torch, dim=-1).backward(grad_out)

        torch.testing.assert_close(x_cuda.grad, x_torch.grad)

    def test_softmax_forward_tail(self, device, softmax_op):
        """N % 4 != 0 hits the scalar tail and misaligned later rows."""
        x = torch.randn(2, 4, 5, dtype=torch.float32, device=device)
        out_cuda = softmax_op(x)
        out_torch = F.softmax(x, dim=-1)
        torch.testing.assert_close(out_cuda, out_torch)

    def test_softmax_bwd_tail(self, device, softmax_op):
        """Backward through N % 4 != 0 rows vs torch."""
        x = torch.randn(2, 4, 5, dtype=torch.float32, device=device)
        x_cuda = x.detach().clone().requires_grad_(True)
        x_torch = x.detach().clone().requires_grad_(True)
        grad_out = torch.randn_like(x_cuda)

        softmax_op(x_cuda).backward(grad_out)
        F.softmax(x_torch, dim=-1).backward(grad_out)

        torch.testing.assert_close(x_cuda.grad, x_torch.grad)
