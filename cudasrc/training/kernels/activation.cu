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
    1 + e^{-x} + x * e^{-x} / (1 + e^{-x})^{2}
*/
__global__ void bwd_silu() {

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
*/
__host__ void launch_silu_bwd() {

}