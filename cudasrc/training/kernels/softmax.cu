/*
Softmax Kernel

We are computing the numerically stable softmax! So the process is:
1. Find the max of the row.
2. Compute exp(x - max) for each element in the row.
3. Compute sum of exponentials for the row.
4. Normalise.

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

@param x Input logits (batch_size × seq_len × n_embed)
@param out Output probabilities (batch_size × seq_len × n_embed)
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
    float* row_max = shared_mem;
    float* row_sum = &shared_mem[blockDim.x]; // Where shared mem becomes row_sum not row_max

    if (b_idx < batch_size && seq_idx < seq_len) {

        // Get the local max for the thread
        float local_max = -INFINITY;
        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            local_max = fmaxf(local_max, x[idx]);
        }
        row_max[thread_idx] = local_max;
        __syncthreads();

        // Tree reduce the max to find the actual max for the row
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread_idx < stride) {
                row_max[thread_idx] = fmaxf(row_max[thread_idx], row_max[thread_idx + stride]);
            }
            __syncthreads();
        }

        // Lets grab the max for the row.
        float global_row_max = row_max[0];

        // Now that we have the max, lets get the sum of exponentials
        float local_sum = 0.f;
        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            local_sum += expf(x[idx] - global_row_max);
        }
        row_sum[thread_idx] = local_sum;
        __syncthreads();

        // Now that we have the partial sums, lets tree reduce to get the global sum.
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread_idx < stride) {
                row_sum[thread_idx] = row_sum[thread_idx] + row_sum[thread_idx + stride];
            }
            __syncthreads();
        }

        // Grab the global sum for the row.
        float global_row_sum = row_sum[0];

        // Now that we have all we need, lets do the actual normalisation.
        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            out[idx] = expf(x[idx] - global_row_max) / global_row_sum;
        }
    }
}

/*
Softmax backward kernel
*/
__global__ void bwd_softmax () {
   
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
    size_t shared_mem = 2 * threads_per_block * sizeof(float);
    fwd_softmax<<<blocks, threads_per_block, shared_mem>>>(
        x, out, batch_size, seq_len, n_embed
    );
}

/*
Bwd softmax kernel launch
*/
__host__ void launch_bwd_softmax() {

}