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

v0.0.7 - Testing the 1D blocktiling

v0.0.8 - Testing the 2D blocktiling.

v0.0.9 - Testing the transposed smem and vectorised loads.

v0.0.10 - Testing vectorised loads on elementwise kernels.

v0.0.11 - Testing online softmax (new test for casual mask added).

v0.0.12 - Testing shuffle reduction.

v0.0.13 - Testing vec loads for softmax.

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

v0.0.7 - Smem Batched MatMul layer 0 numbers

v0.0.8 - 1D blocktiling

v0.0.9 - 2D blocktiling

v0.0.10 - 2D blocktiling shape alterations for batched matmul.

v0.0.11 - smem transpose and vectorised loads.

v0.0.12 - Different BK on the matmul.

v0.0.13 - Elementwise vectorised loads. 

v0.0.14 - Online softmax

v0.0.15 - Shuffle reduction (softmax).

v0.0.16 - Vec loading (softmax).

### Profiling Layer 1

v0.0.1 - first draft of layer 1 of profiling pipeline.
