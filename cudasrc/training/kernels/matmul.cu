/*
CUDA code for matrix multiply.
Within QWEN we will be using this in the attention and FFN layers.
*/

#include <cuda_runtime.h>

/*
Matmul forward pass.

Uses shared memory to reduce trips to HBM when performing matmul.

Y = A @ B:
    A is M x K
    B is K x N
    Y = M x N

@param A: Input matrix A (M x K)
@param B: Input matrix B (K x N)
@param C: Output matrix C (M x N)
@param M: The number of rows in A and C
@param N: The number of cols in B and C
@param K: The number of columns in A and rows in B.
*/
template <const int BLOCKSIZE>
__global__ void fwd_matmul(const float* A, const float* B, float* C, int M, int N, int K) {

    // Set the output tile that we need to compute for! 
    int c_rows = blockIdx.y;
    int c_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    A += c_rows * BLOCKSIZE * K; // jump rows of A (M x K) to find the tiles rows
    B += c_cols * BLOCKSIZE; // jump cols of B (K x N) to find the tiles columns
    C += c_rows * BLOCKSIZE * N + c_cols * BLOCKSIZE; // where the tile sits in the output.

    // Shared memory to load the tiles into
    __shared__ float smem_A[BLOCKSIZE * BLOCKSIZE];
    __shared__ float smem_B[BLOCKSIZE * BLOCKSIZE];

    // Thread positions within the block! In smem thread owns [thread_row, thread_col]
    int thread_row_idx = threadIdx.y;
    int thread_col_idx = threadIdx.x;

    // Global row and col this thread is going to handle. This can be out of bounds.
    int row = c_rows * BLOCKSIZE + thread_row_idx;
    int col = c_cols * BLOCKSIZE + thread_col_idx;

    // Output for this specific thread C[thread_row, thread_col]
    float thread_sum = 0.f;

    // Now we start iterating through K in tile steps computing the total as we go.
    for (int t_k_idx = 0; t_k_idx < K; t_k_idx += BLOCKSIZE) {

        // We needs to bounds check to ensure we do not have issues when 
        // our tiles do not fit into the A and B neatly.
        int a_k = t_k_idx + thread_col_idx;  // A[row, a_k]
        int b_k = t_k_idx + thread_row_idx;  // B[b_k, col]
        
        // Becuase of our indexing pattern, this is all coalseced! 
        smem_A[thread_row_idx * BLOCKSIZE + thread_col_idx] =
        (row < M && a_k < K) ? A[thread_row_idx * K + thread_col_idx] : 0.f; // if out of bounds write 0 not garbage!

        smem_B[thread_row_idx * BLOCKSIZE + thread_col_idx] =
        (b_k < K && col < N) ? B[thread_row_idx * N + thread_col_idx] : 0.f; // ^^^

        // ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BLOCKSIZE; // move right a block
        B += BLOCKSIZE * N; // move down a block

        // Use smem values to compute the rolling dot product.
        for (int dot_idx = 0; dot_idx < BLOCKSIZE; dot_idx++) {
            thread_sum += smem_A[thread_row_idx * BLOCKSIZE + dot_idx] * smem_B[dot_idx * BLOCKSIZE + thread_col_idx];
        }
        
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // write the final output
    if (row < M && col < N) {
        C[thread_row_idx * N + thread_col_idx] = thread_sum;    
    }
}

/*
Standard matmul backward pass to compute the grad with respect to matrix A.

@param grad_out: The gradients of the outputs (M x N)
@param B: Input matrix B (K x N)
@param grad_a: The gradient with respsect to the input A (M x K)
@param M: The number of rows in grad_out and grad_a
@param N: The number of columns in B and grad_out
@param K: The number of rows in B and the number of columns in grad_A
*/
template <const int BLOCKSIZE>
__global__ void bwd_matmul_a(const float* grad_out, const float* B, float* grad_a, int M, int N, int K) {

    // grad_a tile
    int grad_a_rows = blockIdx.y;
    int grad_a_cols = blockIdx.x;

    // Get the tiles in the correct positions
    grad_out += grad_a_rows * BLOCKSIZE * N; // Jump to row
    B += grad_a_cols * BLOCKSIZE * N; // Jump to col (row really)
    grad_a += grad_a_rows * BLOCKSIZE * K + grad_a_cols * BLOCKSIZE; // where we are writing to.

    // shared memory
    __shared__ float smem_G[BLOCKSIZE * BLOCKSIZE];
    __shared__ float smem_B[BLOCKSIZE * BLOCKSIZE];

    // thread idx in the block
    int thread_row_idx = threadIdx.y;
    int thread_col_idx = threadIdx.x;

    // global row/col for bounds checking
    int row = grad_a_rows * BLOCKSIZE + thread_row_idx;
    int col = grad_a_cols * BLOCKSIZE + thread_col_idx;

    float grad_sum = 0.f;

    // Now we loop through N filling up as we go!
    for (int t_n_idx = 0; t_n_idx < N; t_n_idx += BLOCKSIZE) {

        // The index that we are currently working with.
        int g_n = t_n_idx + thread_col_idx;
        int b_row = grad_a_cols * BLOCKSIZE + thread_row_idx;

        smem_G[thread_row_idx * BLOCKSIZE + thread_col_idx] = 
        (row < M && g_n < N) ? grad_out[thread_row_idx * N + thread_col_idx] : 0.f;
        
        smem_B[thread_row_idx * BLOCKSIZE + thread_col_idx] = 
        (b_row < K && g_n < N) ? B[thread_row_idx * N + thread_col_idx] : 0.f;

        __syncthreads();

        grad_out += BLOCKSIZE;
        B += BLOCKSIZE;

        // now we move through the shared memory computing the partial sums
        for (int smem_idx = 0; smem_idx < BLOCKSIZE; smem_idx++) {
            grad_sum += smem_G[thread_row_idx * BLOCKSIZE + smem_idx] * smem_B[thread_col_idx * BLOCKSIZE + smem_idx];
        }

        __syncthreads();
    }


    if (row < M && col < K) {
        grad_a[thread_row_idx * K + thread_col_idx] = grad_sum;
    }
}

/*
Standard matmul backward pass to compute the grad with respect to matrix B.

@param grad_out: The gradients of the outputs (M x N)
@param A: The input matrix A (M x K)
@param grad_b: The gradients with respect to the input B (K x N)
@param M: the number of rows in grad_out and A.
@param N: The number of columns in grad_out and B
@param K: The number of columns in A and rows in grad_B
*/
template<const int BLOCKSIZE>
__global__ void bwd_matmul_b(const float* grad_out, const float* A, float* grad_b, int M, int N, int K) {

    // grab the location that this tile is covering for grad_b
    int grad_b_rows = blockIdx.y;
    int grad_b_cols = blockIdx.x;

    // Tile to correct positions
    grad_out += grad_b_cols * BLOCKSIZE; // jump to the correct row
    A += grad_b_rows * BLOCKSIZE; // jump to the correct col (row really)
    grad_b += grad_b_rows * BLOCKSIZE * N + grad_b_cols * BLOCKSIZE;

    // Shared memory
    __shared__ float smem_G[BLOCKSIZE* BLOCKSIZE];
    __shared__ float smem_A[BLOCKSIZE * BLOCKSIZE];

    // Thread indexing within the tile.
    int thread_row_idx = threadIdx.y;
    int thread_col_idx = threadIdx.x;

    // global indexing for bounds checking
    int row = grad_b_rows * BLOCKSIZE + thread_row_idx;
    int col = grad_b_cols * BLOCKSIZE + thread_col_idx;

    float grad_sum = 0.f;

    // Iterate the tile through the matrices down M. 
    for (int t_m_idx = 0; t_m_idx < M; t_m_idx += BLOCKSIZE) {
        
        // get the index for thread to load value from grad_out and A.
        int g_m = t_m_idx + thread_row_idx;
        int a_k =  grad_b_rows * BLOCKSIZE + thread_col_idx;

        smem_G[thread_row_idx * BLOCKSIZE + thread_col_idx] = 
        (g_m < M && col < N) ? grad_out[thread_row_idx * N + thread_col_idx] : 0.f;

        smem_A[thread_row_idx * BLOCKSIZE + thread_col_idx] =
        (g_m < M && a_k < K) ? A[thread_row_idx * K + thread_col_idx] : 0.f;

        __syncthreads();

        grad_out += BLOCKSIZE * N;
        A += BLOCKSIZE * K;
        
        // Now we loop through smem and compute the partial products
        for (int smem_idx = 0; smem_idx < BLOCKSIZE; smem_idx++) {
            grad_sum += smem_G[smem_idx * BLOCKSIZE + thread_col_idx] * smem_A[smem_idx * BLOCKSIZE + thread_row_idx];
        }

        __syncthreads();
    }
    if (row < K && col < N) {
        grad_b[thread_row_idx * N + thread_col_idx] = grad_sum;
    }
}

/*
Forward pass for batched matrix multiplication used in attention.
Computes Y[batch] = A[batch] * B[batch] in parallel!

@param A: Input matrix A (batch size, M, K).
@param B: Input matrix B (batch size, K, N).
@param C: Input matrix C (out) (batch size, M, N)
@param batch_size: The size of the batch.
@param M: The number of rows in A and C.
@param N: The number of cols in B.
@param K: The number of cols in A and rows in B.
*/
__global__ void fwd_batched_matmul(const float* A, const float* B, float* C, int batch_size, int M, int N, int K) {

    // Indexing
    int batch = blockIdx.z; // batch idx (the slice in the 3D grid).
    int row = blockIdx.y * blockDim.y + threadIdx.y; // The row within the batch.
    int col = blockIdx.x * blockDim.x + threadIdx.x; // The column within the batch.

    // Bounds checking
    if (batch < batch_size && row < M && col < N) {
        float sum = 0.f;
        for (int k = 0; k < K; k++) {
            // A[batch, row, idx within row], B[batch, idx within col, col]
            sum += A[batch * M * K + row * K + k] * B[batch * K * N + k * N + col];
        }
        // Lets get that sum into C.
        C[batch * M * N + row * N + col] = sum; // C[batch, row, col]
    }
}

/*
Backward pass for batched matmul to compute the gradients for A.

@param grad_out: The gradients of the outputs (batch size, M, N)
@param B: The input matrix B (batch size, K, N)
@param grad_a: The gradients with respect to the input A (batch size, M, K)
@param batch_size: The size of the batch.
@param M: The number of rows in grad_out and grad_a.
@param N: The number of columns in B and grad_out.
@param K: The number of columns in A and rows in grad_b.
*/
__global__ void bwd_batched_matmul_a(const float* grad_out, const float* B, float* grad_a, int batch_size, int M, int N, int K) {

    int batch = blockIdx.z; // batch idx (the slice in the 3D grid).
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row idx for grad_a
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col idx for grad_a

    // Bounds checking
    if (batch < batch_size && row < M && col < K) {
        float sum = 0.f;
        // Iterate through the row we want.
        for (int n = 0; n < N; n ++) {
            sum += grad_out[batch * M * N + row * N + n] * B[batch * K * N + col * N + n];
        }
        grad_a[batch * M * K + row * K + col] = sum;
    }
}

/*
Backward pass for batched matmul to compute gradients for B.

@param grad_out: The gradients of the outputs (batch size, M, N)
@param A: The input matrix A (batch size, M, K)
@param grad_b: The gradients with respect to the input B (batch size, K, N)
@param batch_size: The size of the batch.
@param M: The number of rows in grad_out and A.
@param N: The number of columns in grad_out and B.
@param K: The number of columns in A and rows in grad_b.
*/
__global__ void bwd_batched_matmul_b(const float* grad_out, const float* A, float* grad_b, int batch_size, int M, int N, int K) {
    
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < K && col < N) {
        float sum = 0.f;
        for (int m = 0; m < M; m++) {
            // A[batch, :, row idx]
            sum += A[batch * M * K + m * K + row] * grad_out[batch * M * N + m * N + col];
        }
        grad_b[batch * K * N + row * N + col] = sum;
    }
}


/*
Kernel launch for matmul.

@param A: Input matrix A (M x K)
@param B: Input matrix B (K x N)
@param C: Output matrix C (M x N) 
@param M: The number of rows in A and C
@param N: The number of cols in B and C
@param K: The number of columns in A and rows in B.
*/
__host__ void launch_fwd_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // With 32 x 32 or 32 x 16, we are fully coalescing our warps in B and C! 
    // We are going for 16 to keep the number of active warps high is our SM can only hold 1536 threads!
    const int BLOCKSIZE = 32;
    dim3 threads_per_block(BLOCKSIZE, BLOCKSIZE); 
    dim3 blocks(
        (N + BLOCKSIZE - 1) / BLOCKSIZE,
        (M + BLOCKSIZE - 1) / BLOCKSIZE
    );
    fwd_matmul<BLOCKSIZE><<<blocks, threads_per_block>>>(A, B, C, M, N, K);
}

/*
Kernel launch for backward matmul to compute grads for A and B.

@param grad_out: The gradients of the outputs (M x N)
@param A: The input matrix A (M x K)
@param B: The input matrix B (K x N)
@param grad_a: The gradients with respect to the input A (M x K)
@param grad_b: The gradients with respect to the input B (K x N)
@param M: The number of rows in grad_out and A
@param N: The number of columns in grad_out and B
@param K: The number of columns in A and rows in grad_b
*/
__host__ void launch_bwd_matmul(
    const float* grad_out, const float* A, const float* B,
    float* grad_a, float* grad_b, int M, int N, int K
) {

    // Compute the backward for A.
    // Output is M x K
    const int BLOCKSIZE = 32;
    dim3 threads_per_block_a(BLOCKSIZE, BLOCKSIZE); // Savings on the write, not much we can do currently with the reads.
    dim3 blocks_a(
        (K + BLOCKSIZE - 1) / BLOCKSIZE,
        (M + BLOCKSIZE - 1) / BLOCKSIZE
    );
    bwd_matmul_a<BLOCKSIZE><<<blocks_a, threads_per_block_a>>>(grad_out, B, grad_a, M, N, K);

    // Compute the backward for B.
    // Output is K x N
    dim3 threads_per_block_b(BLOCKSIZE, BLOCKSIZE); // Savings on the reads from grad_out and writes to grad_b
    dim3 blocks_b(
        (N + BLOCKSIZE - 1) / BLOCKSIZE,
        (K + BLOCKSIZE - 1) / BLOCKSIZE
    );
    bwd_matmul_b<BLOCKSIZE><<<blocks_b, threads_per_block_b>>>(grad_out, A, grad_b, M, N, K);
}

/*
Kernel launch for the forward batched matmul.

@param A: The input matrix A (batch size, M, K)
@param B: The input matrix B (batch size, K, N)
@param C: The output matrix C (batch size, M, N)
@param batch_size: The size of the batch
@param M: The number of rows in A and C
@param N: The number of columns in B and C
@param K: The number of columns in A and rows in B
*/
__host__ void launch_fwd_batched_matmul(
    const float* A, const float* B, float* C, int batch_size, int M, int N, int K
) {
    // Reduce the size of the blocks to account for the "slices" in the batch dimension.
    dim3 threads_per_block(32, 16); // Move 128 bytes per warp instead of 32 and broadcast to 8. Broadcast to 16 instead of 8!
    dim3 blocks(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );
    fwd_batched_matmul<<<blocks, threads_per_block>>>(A, B, C, batch_size, M, N, K);

}

/*
Kernel launch for the backward pass for batched matmul.

@param grad_out: The gradients of the outputs (batch size, M, N)
@param A: The input matrix A (batch size, M, K)
@param B: The input matrix B (batch size, K, N)
@param grad_a: The gradients with respect to the input A (batch size, M, K)
@param grad_b: The gradients with respect to the input B (batch size, K, N)
@param batch_size: The size of the batch
@param M: The number of rows in grad_out and A
@param N: The number of columns in B and grad_out
@param K: The number of columns in A and rows in grad_b
*/
__host__ void launch_bwd_batched_matmul(
    const float* grad_out, const float* A, const float* B,
    float* grad_a, float* grad_b, int batch_size, int M, int N, int K
) {
    // Compute the backward for A.
    // Output is (batch size, M, K)
    dim3 threads_per_block_a(32, 16);
    dim3 blocks_a(
        (K + threads_per_block_a.x - 1) / threads_per_block_a.x,
        (M + threads_per_block_a.y - 1) / threads_per_block_a.y,
        batch_size
    );
    bwd_batched_matmul_a<<<blocks_a, threads_per_block_a>>>(grad_out, B, grad_a, batch_size, M, N, K);

    // Compute the backward for B.
    // Output is (batch size, K, N)
    dim3 threads_per_block_b(32, 16);
    dim3 blocks_b(
        (N + threads_per_block_b.x - 1) / threads_per_block_b.x,
        (K + threads_per_block_b.y - 1) / threads_per_block_b.y,
        batch_size
    );
    bwd_batched_matmul_b<<<blocks_b, threads_per_block_b>>>(grad_out, A, grad_b, batch_size, M, N, K);
}
