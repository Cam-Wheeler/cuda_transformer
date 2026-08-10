/*
RMSNorm Kernel.

RMSNorm: a_i = (a_i / RMS(a)) * g where:
    RMS(a) = sqrt{1/ n \sum{a_i^2}}
*/

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <device_launch_parameters.h>

/*
Forward pass for RMSNorm

@param
*/
__global__ void fwd_rmsnorm(
    const float* x, const float* gamma,
    float* out, float* mean_out,
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
    if (b_idx < batch_size and seq_idx < seq_len) {
        float local_sum_sq = 0.f;
        for (int i = thread_idx; i < n_embed; i += blockDim.x) { // jump blockdim each time.
            int idx = b_idx * seq_len * n_embed + seq_idx * n_embed + i;
            local_sum_sq += x[idx] * x[idx];
        }
        sum_sq_vals[thread_idx] = local_sum_sq; // index into smem using the thread idx.

        // Now we need to reduce the partial sums into a single sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (thread_idx < stride) {
                sum_sq_vals[thread_idx] += sum_sq_vals[thread_idx + stride];
            }
        }
        __syncthreads();

        // Compute the mean of the sum of squares. It will be sat at idx[0] in smem.
        float total_sum_sq = sum_sq_vals[0];
        float mean_sum_sq = total_sum_sq / n_embed;

        // Save the mean for backward pass.
        if (thread_idx == 0) {
            int mean_idx = b_idx * seq_len + seq_idx;
            mean_out[mean_idx] = mean_sum_sq;
        }

        // Normalise
        float inv_rms = rsqrtf(mean_sum_sq + eps);
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
*/
__host__ void launch_fwd_rmsnorm() {

}

/*
Kernel launch for RMSNorm backward.
*/
__host__ void launch_bwd_rmsnorm() {

}
