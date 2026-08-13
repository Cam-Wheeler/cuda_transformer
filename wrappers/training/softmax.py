"""
Python wrapper for the Softmax.
"""

import torch
from torch import nn
from torch.autograd import Function

class Softmax(nn.Module):
    """
    nn.Module wrapper for the Softmax layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass.
        """
        return SoftmaxFunction.apply(x)

class SoftmaxFunction(Function):
    """
    Custom autograd function for the Softmax layer.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass.

        Args:
        - x: The input tensor

        Returns:
        - out: The output of softmax.
        """
        
        # Shapes
        batch_size, seq_len, n_embed = x.size()

        # Allocate for out.
        out = torch.empty(batch_size, seq_len, n_embed, dtype=x.dtype, device=x.device)

        import custom_training as ct
        ct.fwd_softmax(x, out)

        ctx.save_for_backward(out) # Save the out as we need it to compute grads.

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.

        Args:
        - grad_out: Gradient wrt output.

        Returns:
        - grad_x: Gradient wrt input.
        """

        output_probs, = ctx.saved_tensors

        # Shapes
        batch_size, seq_len, n_embed = output_probs.size()

        # Grad storage tensors
        grad_x = torch.empty(
            batch_size, seq_len, n_embed, dtype=output_probs.dtype, device=output_probs.device
        )

        import custom_training as ct
        ct.bwd_softmax(grad_out, output_probs, grad_x)

        return grad_x
