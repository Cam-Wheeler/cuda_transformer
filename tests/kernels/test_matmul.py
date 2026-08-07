"""
Parity tests for custom CUDA matmul operations vs torch.
Requires a CUDA GPU and a built `custom_training` extension.
"""

import pytest
import torch

from wrappers.training import MatMul, BatchedMatMul


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)


@pytest.fixture
def device():
    return torch.device("cuda")

@pytest.fixture
def input_weight_pair(device):
    """float32 tensors on CUDA for testing."""
    x = torch.randn(16, 64, dtype=torch.float32, device=device)
    weight = torch.randn(64, 128, dtype=torch.float32, device=device)
    return x, weight

@pytest.fixture
def batched_input_weight_pair(device):
    """float32 tensors on CUDA for testing."""
    a = torch.randn(4, 16, 64, dtype=torch.float32, device=device)
    b = torch.randn(4, 64, 128, dtype=torch.float32, device=device)
    return a, b

@pytest.fixture
def matmul_op():
    return MatMul()

@pytest.fixture
def batched_matmul_op():
    return BatchedMatMul()

@requires_cuda
class TestMatMul:

    def test_matmul_forward(self, input_weight_pair, matmul_op):
        """Test the forward for matmul"""
        x, weight = input_weight_pair
        out_cuda = matmul_op(x, weight)
        out_torch = torch.mm(x, weight)
        torch.testing.assert_close(out_cuda, out_torch) # testing values and shape

    def test_matmul_bwd(self, input_weight_pair, matmul_op):
        """Test the backward for matmul"""
        x, weight = input_weight_pair

        x_cuda = x.detach().clone().requires_grad_(True)
        weight_cuda = weight.detach().clone().requires_grad_(True)

        x_torch = x.detach().clone().requires_grad_(True)
        weight_torch = weight.detach().clone().requires_grad_(True)

        matmul_op(x_cuda, weight_cuda).sum().backward()
        torch.mm(x_torch, weight_torch).sum().backward()

        torch.testing.assert_close(x_cuda.grad, x_torch.grad)
        torch.testing.assert_close(weight_cuda.grad, weight_torch.grad)

        
@requires_cuda
class TestBatchedMatMul:

    def test_batched_matmul_forward(self, batched_input_weight_pair, batched_matmul_op):
        """Test the forward for batched matmul"""
        a, b = batched_input_weight_pair
        out_cuda = batched_matmul_op(a, b)
        out_torch = torch.bmm(a, b)
        torch.testing.assert_close(out_cuda, out_torch) # testing values and shape

    def test_batched_matmul_backward(self, batched_input_weight_pair, batched_matmul_op):
        """Test the backward for batched matmul"""
        a, b = batched_input_weight_pair

        a_cuda = a.detach().clone().requires_grad_(True)
        b_cuda = b.detach().clone().requires_grad_(True)

        a_torch = a.detach().clone().requires_grad_(True)
        b_torch = b.detach().clone().requires_grad_(True)

        batched_matmul_op(a_cuda, b_cuda).sum().backward()
        torch.bmm(a_torch, b_torch).sum().backward()

        torch.testing.assert_close(a_cuda.grad, a_torch.grad)
        torch.testing.assert_close(b_cuda.grad, b_torch.grad)