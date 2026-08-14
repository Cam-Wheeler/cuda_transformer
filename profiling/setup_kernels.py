"""Set up the kernels for profiling"""

import torch
from torch.nn import functional as F
from wrappers.training import (
    BatchedMatMul, ElementWiseAdd, ElementWiseMultiplication, 
    MatMul, RMSNorm, Softmax
)

def setup_matmul():
    """FFN up-projection shape: [B*S, E] @ [E, H]."""

    M = 4 * 256 # Batch size * context length
    K = 1024 # Embedding dim
    N = 3072 # FFN hidden dim

    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    op = MatMul()

    return {
        "cuda": lambda: op(A, B),
        "torch": lambda: torch.mm(A, B),
        "flops": 2 * M * N * K,
        "bytes": None,
        "shape": f"({M}, {K}) @ ({K}, {N})",
    }

def setup_batch_matmul():
    """GQA attention shape: [B*H, S, E] @ [B*H, E, S]. Q @ K^T."""

    BATCH = 4 # Batch size
    N_HEADS = 16 # Number of heads
    SEQ_LEN = 256 # Seq len
    HEAD_DIM = 128 # head dim

    A = torch.randn(BATCH * N_HEADS, SEQ_LEN, HEAD_DIM, device="cuda")
    B = torch.randn(BATCH * N_HEADS, HEAD_DIM, SEQ_LEN, device="cuda")
    op = BatchedMatMul()

    return {
        "cuda": lambda: op(A, B),
        "torch": lambda: torch.bmm(A, B),
        "flops": 2 * (BATCH * N_HEADS) * SEQ_LEN * SEQ_LEN * HEAD_DIM,
        "bytes": None,
        "shape": f"({BATCH * N_HEADS}, {SEQ_LEN}, {HEAD_DIM}) @ ({BATCH * N_HEADS}, {HEAD_DIM}, {SEQ_LEN})",
    }

def setup_elementwise_add():
    """
    Elementwise addition shape:
    [Batch * Seq_Len * Embedding] + [Batch * Seq_Len * Embedding]
    """

    BATCH = 4
    SEQ_LEN = 256
    EMBED_DIM = 1024

    A = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, device="cuda")
    B = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, device="cuda")
    op = ElementWiseAdd()

    return {
        "cuda": lambda: op(A, B),
        "torch": lambda: torch.add(A, B),
        "flops": None,
        "bytes": 3 * A.numel() * 4,
        "shape": f"({BATCH}, {SEQ_LEN}, {EMBED_DIM}) + ({BATCH}, {SEQ_LEN}, {EMBED_DIM})"
    }

def setup_elementwise_multi():
    """
    Elementwise multiplication shape: 
    [Batch * Seq_Len * Embedding] * [Batch * Seq_Len * Embedding]
    """

    BATCH = 4
    SEQ_LEN = 256
    EMBED_DIM = 3072 # FFN elementwise multi dim.

    A = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, device="cuda")
    B = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, device="cuda")
    op = ElementWiseMultiplication()

    return {
        "cuda": lambda: op(A, B),
        "torch": lambda: torch.mul(A, B),
        "flops": None,
        "bytes": 3 * A.numel() * 4,
        "shape": f"({BATCH}, {SEQ_LEN}, {EMBED_DIM}) * ({BATCH}, {SEQ_LEN}, {EMBED_DIM})"
    }

def setup_rms_norm():
    """RMSNorm Shape: [Batch * Seq_len * Embedding]"""

    BATCH = 4
    SEQ_LEN = 256
    EMBED_DIM = 1024
    A = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, device="cuda")
    gamma = torch.ones(EMBED_DIM, device="cuda")
    epsilon = 1e-6
    op = RMSNorm()

    return {
        "cuda": lambda: op(A, gamma, epsilon),
        "torch": lambda: F.rms_norm(A, (EMBED_DIM,), weight=gamma, eps=epsilon),
        "flops": None,
        "bytes": 2 * A.numel() * 4 + gamma.numel() * 4,
        "shape": f"({BATCH}, {SEQ_LEN}, {EMBED_DIM})"
    }

def setup_softmax():
    """Softmax shape: [batch_size, seq_len, seq_len]"""

    BATCH = 4
    HEADS = 16
    SEQ_LEN = 256
    A = torch.randn(BATCH * HEADS, SEQ_LEN, SEQ_LEN, device="cuda")
    op = Softmax()

    return {
        "cuda": lambda: op(A),
        "torch": lambda: F.softmax(A, dim=-1),
        "flops": None, 
        "bytes": 2 * A.numel() * 4,
        "shape": f"({BATCH * HEADS}, {SEQ_LEN}, {SEQ_LEN})"
    }

# Allow us to setup the kernels.
KERNELS = {
    "matmul": setup_matmul,
    "batch_matmul": setup_batch_matmul,
    "addition": setup_elementwise_add,
    "multi": setup_elementwise_multi,
    "softmax": setup_softmax,
    "rmsnorm": setup_rms_norm
}