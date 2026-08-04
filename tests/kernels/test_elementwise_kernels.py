"""
Parity tests for custom CUDA element-wise add/mul vs torch.
Requires a CUDA GPU and a built `custom_training` extension.
"""

import pytest
import torch

from wrappers.training import ElementWiseAdd, ElementWiseMultiplication

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)

@pytest.fixture
def device():
    return torch.device("cuda")

@pytest.fixture
def pair(device):
    """float32 tensors on CUDA for testing."""
    a = torch.randn(2, 8, 64, dtype=torch.float32, device=device)
    b = torch.randn(2, 8, 64, dtype=torch.float32, device=device)
    return a, b

@pytest.fixture
def broadcast_pair(device):
    """float32 tensors on CUDA for testing."""
    a = torch.randn(2, 8, 64, dtype=torch.float32, device=device)
    b = torch.randn(1, 8, 64, dtype=torch.float32, device=device)
    return a, b

@pytest.fixture
def add_op():
    return ElementWiseAdd()

@pytest.fixture
def mul_op():
    return ElementWiseMultiplication()

@requires_cuda
class TestAddition:

    def test_forward(self, pair, add_op):
        """Test the forward pass for elementwise addition."""
        a, b = pair
        out_cuda = add_op(a, b)
        out_torch = a + b
        torch.testing.assert_close(out_cuda, out_torch)

    def test_backward(self, pair, add_op):
        """Test the backward pass for elementwise addition."""
        a, b = pair

        a_cuda = a.detach().requires_grad_(True)
        b_cuda = b.detach().requires_grad_(True)

        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        add_op(a_cuda, b_cuda).sum().backward()
        (a_ref + b_ref).sum().backward()

        torch.testing.assert_close(a_cuda.grad, a_ref.grad)
        torch.testing.assert_close(b_cuda.grad, b_ref.grad)

    def test_forward_broadcast(self, broadcast_pair, add_op):
        """Test forward pass with broadcast."""
        a, b = broadcast_pair
        out_cuda = add_op(a, b)
        out_torch = a + b
        torch.testing.assert_close(out_cuda, out_torch)

    def test_backward_broadcast(self, broadcast_pair, add_op):
        """Test backward pass with broadcast."""
        a, b = broadcast_pair
        
        a_cuda = a.detach().requires_grad_(True)
        b_cuda = b.detach().requires_grad_(True)

        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        add_op(a_cuda, b_cuda).sum().backward()
        (a_ref + b_ref).sum().backward()

        torch.testing.assert_close(a_cuda.grad, a_ref.grad)
        torch.testing.assert_close(b_cuda.grad, b_ref.grad)


@requires_cuda
class TestMultiplication:

    def test_forward(self, pair, mul_op):
        """Test forward pass for elementwise multiplication."""
        a, b = pair
        out_cuda = mul_op(a, b)
        out_torch = a * b
        torch.testing.assert_close(out_cuda, out_torch)

    def test_backward(self, pair, mul_op):
        """Test backward pass for elementwise multiplication."""
        a, b = pair

        a_cuda = a.detach().requires_grad_(True)
        b_cuda = b.detach().requires_grad_(True)

        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        mul_op(a_cuda, b_cuda).sum().backward()
        (a_ref * b_ref).sum().backward()

        torch.testing.assert_close(a_cuda.grad, a_ref.grad)
        torch.testing.assert_close(b_cuda.grad, b_ref.grad)