"""
Python wrapper for the non batched matrix multiplication.

This is used in the FFN layer!
"""

import torch
from torch import nn
from torch.autograd import Function


class MatMul(nn.Module):
    """
    nn.Module wrapper for the non batched matrix multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        """
        Forward pass.
        """
        return MatMulFunction.apply(A, B)

class MatMulFunction(Function):
    """
    Custom autograd function for the non batched matrix multiplication.
    """
    @staticmethod
    def forward(ctx, A, B):
        """
        Forward pass.
        """
        ctx.save_for_backward(A, B)

        M, _ = A.shape # M, K really but we do not need the K.
        _, N = B.shape # K, N but again do not need the K.

        C = torch.empty(M, N, dtype=A.dtype, device=A.device)

        import custom_training as ct
        ct.fwd_matmul(A, B, C)

        return C

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.

        Gradient formulas:
            - grad_A = grad_C @ B^T  (gradient w.r.t. A)
            - grad_B = A^T @ grad_C  (gradient w.r.t. B)
        """
        A, B = ctx.saved_tensors

        grad_A = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        grad_B = torch.empty(B.shape, dtype=B.dtype, device=B.device)

        import custom_training as ct
        ct.bwd_matmul(grad_out, A, B, grad_A, grad_B)

        return grad_A, grad_B