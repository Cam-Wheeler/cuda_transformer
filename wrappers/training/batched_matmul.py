"""
Python wrapper for the batched matrix multiplication.

This is used in the GQA!
"""

import torch
from torch import nn
from torch.autograd import Function


class BatchedMatMul(nn.Module):
    """
    nn.Module wrapper for the batched matrix multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        """
        Forward pass.
        """
        return BatchedMatMulFunction.apply(A, B)

class BatchedMatMulFunction(Function):
    """
    Custom autograd function for the batched matrix multiplication.
    """
    @staticmethod
    def forward(ctx, A, B):
        """
        Forward pass.

        Args:
        - ctx: Context object injected by torch.
        - A: Matrix A (batch x M x K).
        - B: Matrix B (batch x K x N)

        Returns:
        - C: Result of batched matrix multiply (batch x M x N)
        """
        ctx.save_for_backward(A, B)

        # Shapes
        batch_size, M, _ = A.shape # batch, M, K
        _, _, N = B.shape # batch, K, N

        C = torch.empty(batch_size, M, N, dtype=A.dtype, device=A.device)

        import custom_training as ct
        ct.fwd_batched_matmul(A, B, C)

        return C

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.

        Gradient formulas:
            - grad_A = grad_C @ B^T  (gradient w.r.t. A)
            - grad_B = A^T @ grad_C  (gradient w.r.t. B)

        Args:
        - ctx: Context object injected by torch.
        - grad_out: Gradients with respect to the output of the matmul (batch x M x N).

        Returns:
        - grad_a: Gradients with respect to A (batch x M x K).
        - grad_b: Gradients with respect to B (batch x K x N). 
        """
        A, B = ctx.saved_tensors

        grad_a = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        grad_b = torch.empty(B.shape, dtype=B.dtype, device=B.device)

        import custom_training as ct
        ct.bwd_batched_matmul(grad_out, A, B, grad_a, grad_b)

        return grad_a, grad_b