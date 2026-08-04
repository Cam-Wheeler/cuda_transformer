/*
Implementation for non-linear activations.

Within QWEN the only non-linear activation used is SILU within
the SwiGLU FFN.
*/

#include <math.h>

/*
Forward pass kernel for SiLU.

SiLU formula: SiLU(x) = x * (1 / (1 + e^{-x}))

@param x: Input to the silu.
@param out: Output from the silu.
@param size: The size of the tensor.
*/
__global__ void fwd_silu(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sigmoid = 1.f / (1.f + expf(-val));
        out[idx] = val * sigmoid;
    }
}

/*
Backward pass kernel for SiLU.

Derivative for SiLU:
    (1 + e^{-x} + x e^{-x}) / (1 + e^{-x})^2

@param grad_out: The gradient coming into the node.
@param x: The input for the forward pass.
@param grad_in: The gradient for the input to the node.
@param size: The size of the tensors. 
*/
__global__ void bwd_silu(const float* grad_out, const float* x, float* grad_in, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float neg_exp = expf(-val);
        float numerator = 1.f + neg_exp + val * neg_exp;
        float denominator = (1.f + neg_exp) * (1.f + neg_exp);
        grad_in[idx] = grad_out[idx] * numerator / denominator;
    }
}

/*
Kernel launch for SiLU forward.

@param x: The input tensor.
@param out: The output tensor.
@param size: The size of the tensor.
*/
__host__ void launch_silu_fwd(const float* x, float* out, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    fwd_silu<<<blocks, threads_per_block>>>(x, out, size);
}

/*
Kernel Launch for SiLU backward.

@param grad_out: The input gradients.
@param x: The input tensor.
@param grad_in: The gradient for the input tensor.
@param size: The size of the tensor.
*/
__host__ void launch_silu_bwd(const float* grad_out, const float* x, float* grad_in, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    bwd_silu<<<blocks, threads_per_block>>>(grad_out, x, grad_in, size);
}