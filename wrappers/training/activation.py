"""
Python wrapper for the non-linear activation functions.

QWEN uses the SiLU activation function within the SwiGLU FFN.
"""

import torch
from torch import nn
from torch.autograd import Function


class SiLU(nn.Module):
    """
    nn.Module wrapper for the SiLU activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass.
        """
        return SiLUFunction.apply(x)

class SiLUFunction(Function):
    """
    Custom autograd function for the SiLU activation function.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass.
        """
        # Save the input for the backward pass.
        ctx.save_for_backward(x)

        out = torch.empty_like(x)
        import custom_training as ct
        ct.fwd_silu(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.
        """
        # load the input back in
        x, = ctx.saved_tensors # Returns to us a tensor, so just need to unpack.

        # Create the gradient tensor to fill.
        grad_x = torch.empty_like(x)

        import custom_training as ct
        ct.bwd_silu(grad_out, x, grad_x)
        return grad_x