"""
Python wrappers for the element-wise operations.
So this code is what is actually called in the model! 

It imports the module (written in C++) created by PyBind

The flow is like so:
    model --> this code --> C++ code --> CUDA kernels.

We use torch.autograd.Function to hook into autograd! Each of these functions have
forward and backward passes that autograd will utilise!
"""

import torch
from torch import nn
from torch.autograd import Function


class ElementWiseAdd(nn.Module):
    """
    nn.Module wrapper around the autograd Function for addition.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return AdditionFunction.apply(a, b)


class AdditionFunction(Function):
    """
    Custom autograd function for element-wise addition forward and backward pass.
    """

    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass.
        """

        a_broadcast, b_broadcast = torch.broadcast_tensors(a, b)

        ctx.save_for_backward(a, b)
        ctx.a_broadcast_shape = a_broadcast.shape
        ctx.b_broadcast_shape = b_broadcast.shape

        out = torch.empty_like(a_broadcast)

        import custom_training as ct
        ct.fwd_add(a_broadcast, b_broadcast, out)

        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.
        """

        a, b = ctx.saved_tensors
        a_broadcast_shape = ctx.a_broadcast_shape
        b_broadcast_shape = ctx.b_broadcast_shape

        grad_a_broadcast = torch.empty(a_broadcast_shape, dtype=a.dtype, device=a.device)
        grad_b_broadcast = torch.empty(b_broadcast_shape, dtype=b.dtype, device=b.device)

        import custom_training as ct
        ct.bwd_add(grad_out, grad_a_broadcast, grad_b_broadcast)

        if a_broadcast_shape != a.shape:
            grad_a = grad_a_broadcast.sum_to_size(a.shape)
        else:
            grad_a = grad_a_broadcast
        
        if b_broadcast_shape != b.shape:
            grad_b = grad_b_broadcast.sum_to_size(b.shape)
        else:
            grad_b = grad_b_broadcast

        return grad_a, grad_b


class ElementWiseMultiplication(nn.Module):
    """
    nn.Module wrapper around the autograd Function for multiplication.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return MultiplicationFunction.apply(a, b)


class MultiplicationFunction(Function):
    """
    Custom autograd function for element-wise multiplication forward and backward pass.
    """
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass
        """
        
        ctx.save_for_backward(a, b)

        out = torch.empty_like(a)

        import custom_training as ct
        ct.fwd_multi(a, b, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass
        """

        a, b = ctx.saved_tensors

        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(b)

        import custom_training as ct
        ct.bwd_multi(grad_out, a, b, grad_a, grad_b)

        return grad_a, grad_b