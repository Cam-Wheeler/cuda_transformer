"""
Parity tests for custom CUDA RMSNorm vs torch.
Requires a CUDA GPU and a built `custom_training` extension.
"""

import pytest
import torch

from wrappers.training import RMSNorm

# Epsilon
EPS = 1e-6

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)

# Base qwen implementation of RMSNorm.
def rmsnorm_torch(x, gamma, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


@pytest.fixture
def device():
    return torch.device("cuda")

@pytest.fixture
def input_gamma_pair(device):
    """float32 tensors on CUDA for testing."""
    x = torch.randn(2, 8, 64, dtype=torch.float32, device=device)
    gamma = torch.randn(64, dtype=torch.float32, device=device)
    return x, gamma

@pytest.fixture
def rmsnorm_op():
    return RMSNorm()


@requires_cuda
class TestRMSNorm:

    def test_rmsnorm_forward(self, input_gamma_pair, rmsnorm_op):
        """Test the forward for RMSNorm"""
        x, gamma = input_gamma_pair
        out_cuda = rmsnorm_op(x, gamma, EPS)
        out_torch = rmsnorm_torch(x, gamma, EPS)
        torch.testing.assert_close(out_cuda, out_torch) # testing values and shape

    def test_rmsnorm_bwd(self, input_gamma_pair, rmsnorm_op):
        """Test the backward for RMSNorm"""
        x, gamma = input_gamma_pair

        x_cuda = x.detach().clone().requires_grad_(True)
        gamma_cuda = gamma.detach().clone().requires_grad_(True)

        x_torch = x.detach().clone().requires_grad_(True)
        gamma_torch = gamma.detach().clone().requires_grad_(True)

        rmsnorm_op(x_cuda, gamma_cuda, EPS).sum().backward()
        rmsnorm_torch(x_torch, gamma_torch, EPS).sum().backward()

        torch.testing.assert_close(x_cuda.grad, x_torch.grad)
        torch.testing.assert_close(gamma_cuda.grad, gamma_torch.grad)
