/*
Softmax Kernel

We are computing the numerically stable softmax (online, 2-pass):
1. One scan: each thread keeps a running max and a running sum of exp.
2. Warp-shuffle reduce, then a tiny smem mailbox across warps.
3. Second scan: write exp(x - max) / sum.

Each block handles a single row. Forward launches one thread per float4
(rounded up to a warp, capped at 256). Threads stride through the row.
Shuffle helpers use blockDim.x so 32–256 threads all work.
*/

#include <cmath>
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp shuffle, then smem[nwarps] mailbox, then warp 0 shuffle.
// Idle lanes in warp 0 hold identity so the mask stays 0xffffffff.
__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    val = warp_reduce_max(val);
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();
    float out = smem[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    val = warp_reduce_sum(val);
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();
    float out = smem[0];
    __syncthreads();
    return out;
}

}

/*
Softmax forward kernel

@param x: Input logits (batch_size × seq_len × n_embed)
@param out: Output probabilities (batch_size × seq_len × n_embed)
@param batch_size: Batch size
@param seq_len: The length of the sequence
@param n_embed: The size of the embeddings (in attention this is also seq len)
*/
__global__ void fwd_softmax(const float* __restrict__ x, float* out, int batch_size, int seq_len, int n_embed) {

    // indexes
    int b_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* smem = shared_mem;

    if (b_idx < batch_size && seq_idx < seq_len) {

        float local_max = -INFINITY;
        float local_norm = 0.f;

        // Row in floats, then float4 view of that row.
        int row = b_idx * seq_len * n_embed + seq_idx * n_embed;
        int n_float4s = n_embed / 4;
        int tail = n_embed % 4;

        // Vec load pointers.
        const float4* input_vec = reinterpret_cast<const float4*>(x + row);
        float4* output_vec = reinterpret_cast<float4*>(out + row);
        float max_val = -INFINITY;

        // Causal masks fill future logits with -inf. exp(-inf - -inf) is NaN, so
        // we clamp the exponent; CUDA fmaxf(NaN, c) returns c. -inf then adds ~0.
        constexpr float kNegClamp = -80.f;

        // float4 needs 16-byte rows. If N % 4 != 0, later rows can be misaligned.
        if ((row % 4) == 0) {
            for (int i = thread_idx; i < n_float4s; i += blockDim.x) {
                float4 elements = input_vec[i];

                max_val = fmaxf(max_val, elements.x);
                max_val = fmaxf(max_val, elements.y);
                max_val = fmaxf(max_val, elements.z);
                max_val = fmaxf(max_val, elements.w);
                if (max_val > local_max) {
                    local_norm *= expf(fmaxf(local_max - max_val, kNegClamp));
                    local_max = max_val;
                }
                local_norm += expf(fmaxf(elements.x - max_val, kNegClamp));
                local_norm += expf(fmaxf(elements.y - max_val, kNegClamp));
                local_norm += expf(fmaxf(elements.z - max_val, kNegClamp));
                local_norm += expf(fmaxf(elements.w - max_val, kNegClamp));
            }

            if (tail && thread_idx < tail) { // cleanup the tail ends if required.
                float val = x[row + n_float4s * 4 + thread_idx];
                if (val > local_max) {
                    local_norm *= expf(fmaxf(local_max - val, kNegClamp));
                    local_max = val;
                }
                local_norm += expf(fmaxf(val - local_max, kNegClamp));
            }  
        } else { // fallback if the row does not fit nicely into a float4
            for (int i = thread_idx; i < n_embed; i += blockDim.x) {
                int idx = row + i;
                float x_val = x[idx];
                float new_max = fmaxf(local_max, x_val);
                local_norm = local_norm * expf(fmaxf(local_max - new_max, kNegClamp))
                                        + expf(fmaxf(x_val - new_max, kNegClamp));
                local_max = new_max;
            }
        }

        float global_row_max = block_reduce_max(local_max, smem);
        float aligned = local_norm * expf(fmaxf(local_max - global_row_max, kNegClamp));
        float global_row_sum = block_reduce_sum(aligned, smem);

        if ((row % 4) == 0) {
            for (int i = thread_idx; i < n_float4s; i += blockDim.x) {
                float4 elements = input_vec[i];
                elements.x = expf(elements.x - global_row_max) / global_row_sum;
                elements.y = expf(elements.y - global_row_max) / global_row_sum;
                elements.z = expf(elements.z - global_row_max) / global_row_sum;
                elements.w = expf(elements.w - global_row_max) / global_row_sum;
                output_vec[i] = elements;
            }
            if (tail && thread_idx < tail) {
                float val = x[row + n_float4s * 4 + thread_idx];
                out[row + n_float4s * 4 + thread_idx] = expf(val - global_row_max) / global_row_sum;
            }
        } else {
            for (int i = thread_idx; i < n_embed; i += blockDim.x) {
                int idx = row + i;
                out[idx] = expf(x[idx] - global_row_max) / global_row_sum;
            }
        }
    }
}

/*
Softmax backward kernel

@param grad_out: The gradient with respect to the outputs (batch_size x seq_len x n_embed)
@param output_probs: output_probabilities (batch_size × seq_len × n_embed)
@param grad_x: Gradients with respect to the inputs (batch_size × seq_len × n_embed)
@param batch_size: Batch size
@param seq_len: The length of the sequence
@param n_embed: The size of the embeddings (in attention this is also seq len)
*/
__global__ void bwd_softmax (
    const float* grad_out, const float* output_probs, 
    float* grad_x, int batch_size, int seq_len, int n_embed
) {

    int b_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* smem = shared_mem;

    if (b_idx < batch_size && seq_idx < seq_len) {

        float local_sum = 0.f;
        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            local_sum += output_probs[idx] * grad_out[idx];
        }
        float global_row_sum = block_reduce_sum(local_sum, smem);

        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            grad_x[idx] = output_probs[idx] * (grad_out[idx] - global_row_sum);
        }
    }
}

/*
Fwd softmax kernel launch

@param x Input logits (batch_size × seq_len × n_embed)
@param out Output probabilities (batch_size × seq_len × n_embed)
@param batch_size: Batch size
@param seq_len: The length of the sequence
@param n_embed: The size of the embeddings (in attention this is also seq len)
*/
__host__ void launch_fwd_softmax(
    const float* x, float* out, 
    int batch_size, int seq_len, int n_embed
) {

    dim3 blocks(batch_size, seq_len); // batch_size num of blocks along X and seq_len blocks along Y.
    // One thread per float4, we round up to a warp so shuffle masks stay valid.
    int n_float4s = n_embed / 4;
    int threads_per_block = n_float4s > 0 ? n_float4s : n_embed;
    threads_per_block = ((threads_per_block + 31) / 32) * 32;
    if (threads_per_block < 32) {
        threads_per_block = 32;
    }
    if (threads_per_block > 256) {
        threads_per_block = 256;
    }
    
    // Smem is one float per warp, not per thread.
    int nwarps = (threads_per_block + 31) / 32;
    size_t shared_mem = nwarps * sizeof(float);
    fwd_softmax<<<blocks, threads_per_block, shared_mem>>>(
        x, out, batch_size, seq_len, n_embed
    );
}

/*
Bwd softmax kernel launch

@param grad_out: The gradient with respect to the outputs (batch_size x seq_len x n_embed)
@param output_probs: output_probabilities (batch_size × seq_len × n_embed)
@param grad_x: Gradients with respect to the inputs (batch_size × seq_len × n_embed)
@param batch_size: Batch size
@param seq_len: The length of the sequence
@param n_embed: The size of the embeddings (in attention this is also seq len)
*/
__host__ void launch_bwd_softmax(
    const float* grad_out, const float* output_probs, 
    float* grad_x, int batch_size, int seq_len, int n_embed
) {
    dim3 blocks(batch_size, seq_len); 
    int threads_per_block = 256;
    int nwarps = (threads_per_block + 31) / 32;
    size_t shared_mem = nwarps * sizeof(float);
    bwd_softmax<<<blocks, threads_per_block, shared_mem>>>(
        grad_out, output_probs, grad_x, batch_size, seq_len, n_embed
    );
}
