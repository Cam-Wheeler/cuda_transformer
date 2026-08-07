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
        """
        pass
    

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.
        """
        pass