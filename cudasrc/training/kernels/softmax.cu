/*
Softmax Kernel

We are computing the numerically stable softmax (online, 2-pass):
1. One scan: each thread keeps a running max and a running sum of exp.
2. Block-reduce the max, then align and block-reduce the sums.
3. Second scan: write exp(x - max) / sum.

Formula:
    softmax(x[i]) = exp(x[i] - max) / sum(exp(x - max))
    Where x is the entire vector, x[i] is an element in the vector and max is the
    max value in the vector.

Each block will handle a single row in the matrix!
*/

#include <cuda_runtime.h>
#include <math.h>

/*
Softmax forward kernel

@param x: Input logits (batch_size × seq_len × n_embed)
@param out: Output probabilities (batch_size × seq_len × n_embed)
@param batch_size: Batch size
@param seq_len: The length of the sequence
@param n_embed: The size of the embeddings (in attention this is also seq len)
*/
__global__ void fwd_softmax(const float* x, float* out, int batch_size, int seq_len, int n_embed) {

    // indexes
    int b_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;

    // Shared memory
    extern __shared__ float shared_mem[];
    float* smem = shared_mem;

    if (b_idx < batch_size && seq_idx < seq_len) {

        // Get the local max for the thread
        float local_max = -INFINITY;
        float local_norm = 0.f;

        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            float x_val = x[idx];
            if (x_val > local_max) {
                local_norm *= expf(local_max - x_val);
                local_max = x_val;
            }
            local_norm += expf(x_val - local_max);
        }

        // Write to smem for block level reduction.
        smem[thread_idx] = local_max;
        __syncthreads();

        // Tree reduce the max to find the actual max for the row
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread_idx < stride) {
                smem[thread_idx] = fmaxf(smem[thread_idx], smem[thread_idx + stride]);
            }
            __syncthreads();
        }

        // Copy the row max out of smem before we reuse the buffer for the sum.
        float global_row_max = smem[0];
        __syncthreads();

        // Align each thread's running sum onto the row max, then tree reduce.
        smem[thread_idx] = local_norm * expf(local_max - global_row_max);
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread_idx < stride) {
                smem[thread_idx] += smem[thread_idx + stride];
            }
            __syncthreads();
        }

        // Grab the global sum for the row.
        float global_row_sum = smem[0];

        // Now that we have all we need, lets do the actual normalisation.
        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            out[idx] = expf(x[idx] - global_row_max) / global_row_sum;
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
        smem[thread_idx] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread_idx < stride) {
                smem[thread_idx] += smem[thread_idx + stride];
            }
            __syncthreads();
        }

        float global_row_sum = smem[0];
        __syncthreads();

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
    int threads_per_block = 256;
    size_t shared_mem = threads_per_block * sizeof(float);
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
    size_t shared_mem = threads_per_block * sizeof(float);
    bwd_softmax<<<blocks, threads_per_block, shared_mem>>>(
        grad_out, output_probs, grad_x, batch_size, seq_len, n_embed
    );
}
