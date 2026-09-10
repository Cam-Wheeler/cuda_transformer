/*
Implementation of element-wise addition and multiplication.
Main targets are the residual connections and FFN within QWEN.

Uses vectorised float4 loads/stores (same idea as matmul) to move four
elements per memory instruction. A scalar tail handles size % 4 != 0.
*/

#include <cuda_runtime.h>

// namespace
namespace {

constexpr int kThreadsPerBlock = 256;

__device__ inline float4 add_f4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline float4 mul_f4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

// Grid sized for float4 slots, not scalars. One block when size < 4 so the
// scalar tail still runs.
inline int elementwise_blocks(int size) {
    if (size <= 0) {
        return 0;
    }
    const int nvec = size / 4;
    int blocks = (nvec + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return blocks < 1 ? 1 : blocks;
}

}

/*
Forward pass kernel for element-wise addition.
y[idx] = a[idx] + b[idx]

@param a: Input tensor a.
@param b: Input tensor b.
@param out: Output tensor
@param size: Number of elements in the tensors.
*/
__global__ void fwd_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ out,
                        int size) {
    const int nvec = size / 4;
    const int tail_start = nvec * 4;
    const int stride = gridDim.x * blockDim.x;

    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
        out4[i] = add_f4(a4[i], b4[i]);
    }

    for (int idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += stride) {
        out[idx] = a[idx] + b[idx];
    }
}

/*
Backward pass kernel for element-wise addition.
a[idx] = grad[idx], b[idx] = grad[idx]

@param grad_out: The gradient of the output.
@param grad_a: The gradient for the input tensor a.
@param grad_b: The gradient for the input tensor b.
@param size: The number of elements in the tensors.
*/
__global__ void bwd_add(const float* __restrict__ grad_out,
                        float* __restrict__ grad_a,
                        float* __restrict__ grad_b,
                        int size) {
    const int nvec = size / 4;
    const int tail_start = nvec * 4;
    const int stride = gridDim.x * blockDim.x;

    const float4* grad_out4 = reinterpret_cast<const float4*>(grad_out);
    float4* grad_a4 = reinterpret_cast<float4*>(grad_a);
    float4* grad_b4 = reinterpret_cast<float4*>(grad_b);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
        float4 g = grad_out4[i];
        grad_a4[i] = g;
        grad_b4[i] = g;
    }

    for (int idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += stride) {
        grad_a[idx] = grad_out[idx];
        grad_b[idx] = grad_out[idx];
    }
}

/*
Forward pass kernel for element-wise multiplication.
y[idx] = a[idx] * b[idx]

@param a: Input tensor a.
@param b: Input tensor b.
@param out: Output tensor.
@param size: Number of elements in the tensors.
*/
__global__ void fwd_multi(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ out,
                          int size) {
    const int nvec = size / 4;
    const int tail_start = nvec * 4;
    const int stride = gridDim.x * blockDim.x;

    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
        out4[i] = mul_f4(a4[i], b4[i]);
    }

    for (int idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += stride) {
        out[idx] = a[idx] * b[idx];
    }
}

/*
Backward pass kernel for element-wise multiplication.
Gradients computed as grad_a = grad_out * b, grad_b = grad_out * a

@param grad_out: The gradient from the output.
@param a: The input tensor a.
@param b: The input tensor b.
@param grad_a: The gradient for input a.
@param grad_b: The gradient for input b.
@param size: Number of elements in the tensor.

We are using the product rule here, so d/dx (a*b) = b, d/dy (a *b) = a
*/
__global__ void bwd_multi(const float* __restrict__ grad_out,
                          const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ grad_a,
                          float* __restrict__ grad_b,
                          int size) {
    const int nvec = size / 4;
    const int tail_start = nvec * 4;
    const int stride = gridDim.x * blockDim.x;

    const float4* grad_out4 = reinterpret_cast<const float4*>(grad_out);
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* grad_a4 = reinterpret_cast<float4*>(grad_a);
    float4* grad_b4 = reinterpret_cast<float4*>(grad_b);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
        float4 g = grad_out4[i];
        float4 va = a4[i];
        float4 vb = b4[i];
        grad_a4[i] = mul_f4(g, vb);
        grad_b4[i] = mul_f4(g, va);
    }

    for (int idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += stride) {
        grad_a[idx] = grad_out[idx] * b[idx];
        grad_b[idx] = grad_out[idx] * a[idx];
    }
}

/*
Lauch code for forward-pass of element-wise addition.
@param a: Input tensor a.
@param b: Input tensor b.
@param out: The output tensor.
@param size: The number of elements in the tensor.

Called on the CPU to launch the kernel on the GPU.
*/
__host__ void launch_fwd_add(const float* a, const float* b, float* out, int size) {
    int blocks = elementwise_blocks(size);
    if (blocks == 0) {
        return;
    }
    fwd_add<<<blocks, kThreadsPerBlock>>>(a, b, out, size);
}

/*
Launch code for the backward pass of element-wise addition.

@param grad_out: The gradient of the output.
@param grad_a: The gradient for the input tensor a.
@param grad_b: The gradient for the input tensor b.
@param size: The number of elements in the tensors.

Called on the CPU to launch the kernel on the GPU.
*/
__host__ void launch_bwd_add(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int blocks = elementwise_blocks(size);
    if (blocks == 0) {
        return;
    }
    bwd_add<<<blocks, kThreadsPerBlock>>>(grad_out, grad_a, grad_b, size);
}

/*
Launch code for the forward pass of element-wise multiplication.

@param: a: Input tensor a.
@param: b: Input tensor b.
@param: out: The output tensor.
@param: size: The number of elements in the tensor.

Called on the CPU to launch the kernel on the GPU.
*/
__host__ void launch_fwd_multi(const float* a, const float* b, float* out, int size) {
    int blocks = elementwise_blocks(size);
    if (blocks == 0) {
        return;
    }
    fwd_multi<<<blocks, kThreadsPerBlock>>>(a, b, out, size);
}

/*
Lanches code for the backward pass of element-wise multiplication.

@param grad_out: The gradient from the output.
@param a: The input tensor a.
@param b: The input tensor b.
@param grad_a: The gradient for input a.
@param grad_b: The gradient for input b.
@param size: Number of elements in the tensor.
*/
__host__ void launch_bwd_multi(const float* grad_out,
                               const float* a,
                               const float* b,
                               float* grad_a,
                               float* grad_b,
                               int size) {
    int blocks = elementwise_blocks(size);
    if (blocks == 0) {
        return;
    }
    bwd_multi<<<blocks, kThreadsPerBlock>>>(grad_out, a, b, grad_a, grad_b, size);
}
