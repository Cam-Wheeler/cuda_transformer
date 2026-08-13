"""
Python wrapper for the RMSNorm layer.
"""

import torch
from torch import nn
from torch.autograd import Function

class RMSNorm(nn.Module):
    """
    nn.Module wrapper for the RMSNorm layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, eps=1e-6):
        """
        Forward pass.
        """
        return RMSNormFunction.apply(x, gamma, eps)

class RMSNormFunction(Function):
    """
    Custom autograd function for the RMSNorm layer.
    """
    @staticmethod
    def forward(ctx, x, gamma, eps):
        """
        Forward pass.

        Args:
        - x: The input tensor
        - gamma: Scale param tensor
        - eps: Epsilon parameter for RMSNorm.

        Returns:
        - out: Normalised tensor.
        """
        
        # Shapes.
        batch_size, seq_len, n_embed = x.size()
        
        # Allocate out and inv_rms_out so we can write to them in the kernel.
        out = torch.empty(batch_size, seq_len, n_embed, dtype=x.dtype, device=x.device)
        inv_rms_out = torch.empty(batch_size, seq_len, dtype=x.dtype, device=x.device)

        import custom_training as ct
        ct.fwd_rmsnorm(x, gamma, out, inv_rms_out, eps)

        # Save important things for backwards
        ctx.save_for_backward(x, gamma, inv_rms_out)
        
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass

        Args:
        - grad_out: Gradient with respect to the output.

        Returns:
        - grad_x: Gradients with respect to the input.
        - grad_gamma: Gradients with respect to the params.
        """
        x_input, gamma, inv_rms_out = ctx.saved_tensors

        # Shapes
        batch_size, seq_len, n_embed = x_input.size()
        
        grad_x = torch.empty(batch_size, seq_len, n_embed, dtype=x_input.dtype, device=x_input.device)
        grad_gamma = torch.empty(gamma.size(), dtype=gamma.dtype, device=gamma.device)

        import custom_training as ct
        ct.bwd_rmsnorm(grad_out, x_input, gamma, inv_rms_out, grad_x, grad_gamma)

        # gradients are returned, we need to return None for eps! 
        return grad_x, grad_gamma, None 



