/*
RMSNorm Kernel.

RMSNorm: a_i = (a_i / RMS(a)) * g where:
    RMS(a) = sqrt{1/ n \sum{a_i^2}}
*/

#include <cuda_runtime.h>

/*
Forward pass for RMSNorm

RMSNorm formula: a_i = (a_i / RMS(a)) * g where:
    RMS(a) = sqrt{1/ n \sum{a_i^2}}

    So the code does the following:
    1. Sum the squares of each feature in the token vector.
    2. Take the mean of that value.
    3. Computing the inverse square root.
    4. Normalise each element in the vector by the inverse square root
    5. Multiply by gamma.

@param x: The input tensor (batch_size, seq_len, n_embed)
@param gamma: The gamma tensor (n_embed)
@param out: The output tensor (batch_size, seq_len, n_embed)
@param inv_rms_out: The inverse RMS tensor for backward pass (batch_size, seq_len)
@param batch_size: The number of sequences in the batch.
@param seq_len: The number of tokens in the sequence.
@param n_embed: The number of features in the embedding.
@param eps: The epsilon value for numerical stability.
*/
__global__ void fwd_rmsnorm(
    const float* x, const float* gamma,
    float* out, float* inv_rms_out,
    int batch_size, int seq_len, int n_embed, float eps
) {

    // Indexes
    int b_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;

    // Shared memory
    extern __shared__ float shared_mem[];
    float* sum_sq_vals = shared_mem; // Stored the partial sum of squares for the token.

    // Compute the partial sum of squares.
    if (b_idx < batch_size && seq_idx < seq_len) {
        float local_sum_sq = 0.f;
        for (int i = thread_idx; i < n_embed; i += blockDim.x) { // jump blockdim each time.
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            local_sum_sq += x[idx] * x[idx]; // square each element.
        }
        sum_sq_vals[thread_idx] = local_sum_sq; // index into smem using the thread idx.

        // We need to be sure that the threads are all done before moving to the reduce.
        __syncthreads();

        // Now we need to reduce the partial sums into a single sum
        // We are using a tree reduce here.
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread_idx < stride) {
                sum_sq_vals[thread_idx] += sum_sq_vals[thread_idx + stride];
            }
            __syncthreads(); // Ensure all are done doing their reduce before reducing more.
        }

        // Compute the mean of the sum of squares. It will be sat at idx[0] in smem.
        float total_sum_sq = sum_sq_vals[0];
        float mean_sum_sq = total_sum_sq / n_embed;
        float inv_rms = rsqrtf(mean_sum_sq + eps);

        // Save the inv_rms for backward pass.
        if (thread_idx == 0) {
            int inv_rms_idx = b_idx * seq_len + seq_idx; // one value per token per sequence per batch.
            inv_rms_out[inv_rms_idx] = inv_rms;
        }

        // Normalise
        for (int i = thread_idx; i < n_embed; i += blockDim.x) {
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            float normalised = (x[idx] * inv_rms);
            out[idx] = normalised * gamma[i];
        }
    }
}


/*
Backward pass for RMSNorm

@param
*/
__global__ void bwd_rmsnorm() {

}


/*
Kernel launch for RMSNorm forward.

@param x: The input tensor (batch_size, seq_len, n_embed)
@param gamma: The gamma tensor (n_embed)
@param out: The output tensor (batch_size, seq_len, n_embed)
@param inv_rms_out: The inverse RMS tensor for backward pass (batch_size, seq_len)
@param batch_size: The number of sequences in the batch.
@param seq_len: The number of tokens in the sequence.
@param n_embed: The number of features in the embedding.
@param eps: The epsilon value for numerical stability.
*/
__host__ void launch_fwd_rmsnorm(
    const float* x, const float* gamma,
    float* out, float* inv_rms_out,
    int batch_size, int seq_len, int n_embed, float eps
) {
    int threads_per_block = 256;
    dim3 blocks(batch_size, seq_len);
    size_t smem = threads_per_block * sizeof(float); // This will hold a partial sum of squares.
    fwd_rmsnorm<<<blocks, threads_per_block, smem>>>(
        x, gamma, out, inv_rms_out, batch_size, seq_len, n_embed, eps
    );
}

/*
Kernel launch for RMSNorm backward.
*/
__host__ void launch_bwd_rmsnorm() {

}
