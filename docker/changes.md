### QWEN PyTorch Training

v0.0.1 - First smoke test.

v0.0.2 - Weight initialisation.

v0.0.3 - Grad clipping and weight decay.

v0.0.4 - DDP with extended mini-run (60000 iterations)

### Kernel Testing

v0.0.1 - First tests for elementwise additions.

v0.0.2 - Refactor and tests for activation (silu).

v0.0.3 - Added tests for matmul.

v0.0.4 - Added tests for rmsnorm.

v0.0.5 - Added tests for softmax.

v0.0.6 - Testing the shared mem matmul and batched matmul.

### CUDA Smoke Tests

v0.0.1 - First smoke test for elementwise addition and multiplication.

v0.0.2 - Smoke test with V1 + silu non-linear activation.

v0.0.3 - Custom Linear in the FFN.

v0.0.4 - Custom Linear in Attention.

v0.0.5 - BMM in GQA.

v0.0.6 - RMSNorm

v0.0.7 - Softmax

### Profiling Layer 0

v0.0.1 - Matmul and Batched Matmul

v0.0.2 - Elementwise add and multi

v0.0.3 - Softmax

v0.0.4 - RMSNorm

v0.0.5 - Warp Coalescing for the MatMul and Batched MatMul.

v0.0.6 - Smem MatMul and Batched MatMul

### Profiling Layer 1

v0.0.1 - first draft of layer 1 of profiling pipeline.
