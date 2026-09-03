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
template <const int BM, const int BN, const int BK, const int TM>
__global__ void fwd_matmul(const float* A, const float* B, float* C, int M, int N, int K) {

    // Set the output tile that we need to compute for! 
    const int c_rows = blockIdx.y;
    const int c_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    A += c_rows * BM * K; // jump rows of A
    B += c_cols * BN; // jump cols of B
    C += c_rows * BM * N + c_cols * BN; // where the tile sits in the output.

    // Shared memory to load the tiles into
    __shared__ float smem_A[BM * BK]; // BM rows, BK cols
    __shared__ float smem_B[BK * BN]; // BK rows, BN cols

    // Thread positions within the block.
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Indexes for the load into smem.
    const int row_a = threadIdx.x;
    const int col_a = threadIdx.y;
    const int row_b = threadIdx.y;
    const int col_b = threadIdx.x;

    // Global index (can be OOB) from the start.
    int row = c_rows * BM + row_a;
    int col = c_cols * BN + col_b;

    // Output for this specific thread C[thread_row, thread_col]
    float thread_results[TM];
    for (int i = 0; i < TM; i++) {
        thread_results[i] = 0.f;
    }

    // Now we start iterating through K in tile steps computing the total as we go.
    for (int t_k_idx = 0; t_k_idx < K; t_k_idx += BK) {

        // Check if the thread is still in bounds as we iterate the tile.
        int a_k = t_k_idx + col_a; // A[row, a_k]
        int b_k = t_k_idx + row_b; // B[b_k, col]
        
        // Load into smem. 1 val from A 1 from B for each thread.
        smem_A[row_a * BK + col_a] =
        (row < M && a_k < K) ? A[row_a * K + col_a] : 0.f; // if out of bounds write 0 not garbage!

        smem_B[row_b * BN + col_b] =
        (b_k < K && col < N) ? B[row_b * N + col_b] : 0.f; // ^^^

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK; // move right a block
        B += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            float b_val = smem_B[dot_idx * BN + thread_col];
    
            // Now we loop through A, reusing our b_val computing the rolling dot
            // for each of the C values the thread is responsible for.
            for (int register_idx = 0; register_idx < TM; register_idx++) {
                thread_results[register_idx] += (
                    smem_A[(thread_row * TM + register_idx) * BK + dot_idx] * b_val
                );
            }
        }
    
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to C.
    for (int register_idx = 0; register_idx < TM; register_idx++) {
        int c_row = thread_row * TM + register_idx;
        if (c_rows * BM + c_row < M && col < N) {
            C[c_row * N + thread_col] = thread_results[register_idx];
        }
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
template <const int BM, const int BN, const int BK, const int TM>
__global__ void bwd_matmul_a(const float* grad_out, const float* B, float* grad_a, int M, int N, int K) {

    // Set the output tile that we need to compute for (grad_a is M x K).
    const int grad_a_rows = blockIdx.y;
    const int grad_a_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    grad_out += grad_a_rows * BM * N; // jump rows of grad_out
    B += grad_a_cols * BN * N; // jump rows of B
    grad_a += grad_a_rows * BM * K + grad_a_cols * BN; // where the tile sits in the output

    // Shared memory.
    __shared__ float smem_G[BM * BK]; // BM rows, BK cols
    __shared__ float smem_BT[BK * BN]; // BK rows, BN cols

    // Thread positions within the block.
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Indexes for the load into smem.
    // G tile is BM x BK.
    const int row_g = threadIdx.x;
    const int col_g = threadIdx.y;

    // B is K x N. row_b/col_b index into B; we store it transposed as BT.
    const int row_b = threadIdx.x;
    const int col_b = threadIdx.y;

    // Global index (can be OOB) from the start.
    int row = grad_a_rows * BM + row_g; // m, for the G load
    int col = grad_a_cols * BN + thread_col; // k, output col of grad_a

    // Output for this specific thread grad_a[thread_row, thread_col]
    float thread_results[TM];
    for (int i = 0; i < TM; i++) {
        thread_results[i] = 0.f;
    }

    // Now we start iterating through N in tile steps.
    for (int t_n_idx = 0; t_n_idx < N; t_n_idx += BK) {

        // Check if the thread is still in bounds as we iterate the tile.
        int g_n = t_n_idx + col_g; // grad_out[row, g_n]
        int b_n = t_n_idx + col_b; // B[row_b, b_n]

        // Load into smem. 1 val from G, 1 from B for each thread.
        smem_G[row_g * BK + col_g] =
            (row < M && g_n < N) ? grad_out[row_g * N + col_g] : 0.f;

        // real B[k, n] -> we store as if B^T[n, k]
        smem_BT[col_b * BN + row_b] =
            (col < K && b_n < N) ? B[row_b * N + col_b] : 0.f;

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK; // move right a block (along N)
        B += BK; // move right a block (along N)

        // Use smem values to compute the rolling dot product.
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            float b_val = smem_BT[dot_idx * BN + thread_col];

            // Now we loop through G, reusing our b_val computing the rolling dot
            // for each of the grad_a values the thread is responsible for.
            for (int register_idx = 0; register_idx < TM; register_idx++) {
                thread_results[register_idx] += (
                    smem_G[(thread_row * TM + register_idx) * BK + dot_idx] * b_val
                );
            }
        }

        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_a (stride K, not N).
    for (int register_idx = 0; register_idx < TM; register_idx++) {
        int grad_a_row = thread_row * TM + register_idx;
        if (grad_a_rows * BM + grad_a_row < M && col < K) {
            grad_a[grad_a_row * K + thread_col] = thread_results[register_idx];
        }
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
template <const int BM, const int BN, const int BK, const int TM>
__global__ void bwd_matmul_b(const float* grad_out, const float* A, float* grad_b, int M, int N, int K) {

    // Set the output tile that we need to compute for (grad_b is K x N).
    const int grad_b_rows = blockIdx.y;
    const int grad_b_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    grad_out += grad_b_cols * BN; // jump cols of grad_out
    A += grad_b_rows * BM; // jump cols of A
    grad_b += grad_b_rows * BM * N + grad_b_cols * BN; // where the tile sits in the output

    // Shared memory.
    __shared__ float smem_AT[BM * BK]; // BM rows, BK cols
    __shared__ float smem_G[BK * BN]; // BK rows, BN cols

    // Thread positions within the block.
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Indexes for the load into smem.
    // AT tile is BM x BK, same mapping as A in the forward.
    const int row_at = threadIdx.x;
    const int col_at = threadIdx.y;

    // G tile is BK x BN, same mapping as B in the forward.
    const int row_g = threadIdx.y;
    const int col_g = threadIdx.x;

    // Global index (can be OOB) from the start.
    int row = grad_b_rows * BM + row_at; // k, for the AT load / grad_b rows
    int col = grad_b_cols * BN + col_g; // n, output col of grad_b

    // Output for this specific thread grad_b[thread_row, thread_col]
    float thread_results[TM];
    for (int i = 0; i < TM; i++) {
        thread_results[i] = 0.f;
    }

    // Now we start iterating through M in tile steps (reduction axis).
    for (int t_m_idx = 0; t_m_idx < M; t_m_idx += BK) {

        // Check if the thread is still in bounds as we iterate the tile.
        int a_m = t_m_idx + col_at; // A[a_m, row]
        int g_m = t_m_idx + row_g; // grad_out[g_m, col]

        // real A[m, k] -> we store as if A^T[k, m]
        smem_AT[row_at * BK + col_at] =
            (a_m < M && row < K) ? A[col_at * K + row_at] : 0.f;

        smem_G[row_g * BN + col_g] =
            (g_m < M && col < N) ? grad_out[row_g * N + col_g] : 0.f;

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK * N; // move down a block (along M)
        A += BK * K; // move down a block (along M)

        // Use smem values to compute the rolling dot product.
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            float g_val = smem_G[dot_idx * BN + thread_col];

            // Now we loop through AT, reusing our g_val computing the rolling dot
            // for each of the grad_b values the thread is responsible for.
            for (int register_idx = 0; register_idx < TM; register_idx++) {
                thread_results[register_idx] += (
                    smem_AT[(thread_row * TM + register_idx) * BK + dot_idx] * g_val
                );
            }
        }

        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_b (stride N).
    for (int register_idx = 0; register_idx < TM; register_idx++) {
        int grad_b_row = thread_row * TM + register_idx;
        if (grad_b_rows * BM + grad_b_row < K && col < N) {
            grad_b[grad_b_row * N + thread_col] = thread_results[register_idx];
        }
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
template <const int BM, const int BN, const int BK, const int TM>
__global__ void fwd_batched_matmul(
    const float* A, const float* B, float* C,
    int batch_size, int M, int N, int K
) {

    // Set the output tile that we need to compute for!
    const int c_batch = blockIdx.z;
    const int c_rows = blockIdx.y;
    const int c_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    A += c_batch * M * K + c_rows * BM * K; // jump batch, then rows of A
    B += c_batch * K * N + c_cols * BN; // jump batch, then cols of B
    C += c_batch * M * N + c_rows * BM * N + c_cols * BN; // where the tile sits in the output

    // Shared memory to load the tiles into
    __shared__ float smem_A[BM * BK]; // BM rows, BK cols
    __shared__ float smem_B[BK * BN]; // BK rows, BN cols

    // Thread positions within the block.
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Indexes for the load into smem.
    const int row_a = threadIdx.x;
    const int col_a = threadIdx.y;
    const int row_b = threadIdx.y;
    const int col_b = threadIdx.x;

    // Global index (can be OOB) from the start.
    int row = c_rows * BM + row_a;
    int col = c_cols * BN + col_b;

    // Output for this specific thread C[thread_row, thread_col]
    float thread_results[TM];
    for (int i = 0; i < TM; i++) {
        thread_results[i] = 0.f;
    }

    // Now we start iterating through K in tile steps computing the total as we go.
    for (int t_k_idx = 0; t_k_idx < K; t_k_idx += BK) {

        // Check if the thread is still in bounds as we iterate the tile.
        int a_k = t_k_idx + col_a; // A[row, a_k]
        int b_k = t_k_idx + row_b; // B[b_k, col]

        // Load into smem. 1 val from A 1 from B for each thread.
        smem_A[row_a * BK + col_a] =
            (row < M && a_k < K) ? A[row_a * K + col_a] : 0.f;

        smem_B[row_b * BN + col_b] =
            (b_k < K && col < N) ? B[row_b * N + col_b] : 0.f;

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK; // move right a block
        B += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            float b_val = smem_B[dot_idx * BN + thread_col];

            // Now we loop through A, reusing our b_val computing the rolling dot
            // for each of the C values the thread is responsible for.
            for (int register_idx = 0; register_idx < TM; register_idx++) {
                thread_results[register_idx] += (
                    smem_A[(thread_row * TM + register_idx) * BK + dot_idx] * b_val
                );
            }
        }

        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to C.
    for (int register_idx = 0; register_idx < TM; register_idx++) {
        int c_row = thread_row * TM + register_idx;
        if (c_batch < batch_size && c_rows * BM + c_row < M && col < N) {
            C[c_row * N + thread_col] = thread_results[register_idx];
        }
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
template <const int BM, const int BN, const int BK, const int TM>
__global__ void bwd_batched_matmul_a(
    const float* grad_out, const float* B, float* grad_a,
    int batch_size, int M, int N, int K
) {

    // Set the output tile that we need to compute for (grad_a is batch x M x K).
    const int grad_a_batch = blockIdx.z;
    const int grad_a_rows = blockIdx.y;
    const int grad_a_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position.
    grad_out += grad_a_batch * M * N + grad_a_rows * BM * N; // jump batch, then rows of grad_out
    B += grad_a_batch * K * N + grad_a_cols * BN * N; // jump batch, then rows of B
    grad_a += grad_a_batch * M * K + grad_a_rows * BM * K + grad_a_cols * BN; // where the tile sits in the output

    // Shared memory.
    __shared__ float smem_G[BM * BK]; // BM rows, BK cols
    __shared__ float smem_BT[BK * BN]; // BK rows, BN cols

    // Thread positions within the block.
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Indexes for the load into smem.
    // G tile is BM x BK.
    const int row_g = threadIdx.x;
    const int col_g = threadIdx.y;

    // B is K x N. row_b/col_b index into B; we store it transposed as BT.
    const int row_b = threadIdx.x;
    const int col_b = threadIdx.y;

    // Global index (can be OOB) from the start.
    int row = grad_a_rows * BM + row_g;      // m, for the G load
    int col = grad_a_cols * BN + thread_col; // k, output col of grad_a

    // Output for this specific thread grad_a[thread_row, thread_col]
    float thread_results[TM];
    for (int i = 0; i < TM; i++) {
        thread_results[i] = 0.f;
    }

    // Now we start iterating through N in tile steps.
    for (int t_n_idx = 0; t_n_idx < N; t_n_idx += BK) {

        // Check if the thread is still in bounds as we iterate the tile.
        int g_n = t_n_idx + col_g; // grad_out[row, g_n]
        int b_n = t_n_idx + col_b; // B[row_b, b_n]

        // Load into smem. 1 val from G, 1 from B for each thread.
        smem_G[row_g * BK + col_g] =
            (row < M && g_n < N) ? grad_out[row_g * N + col_g] : 0.f;

        // real B[k, n] -> we store as if B^T[n, k]
        smem_BT[col_b * BN + row_b] =
            (col < K && b_n < N) ? B[row_b * N + col_b] : 0.f;

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK; // move right a block (along N)
        B += BK; // move right a block (along N)

        // Use smem values to compute the rolling dot product.
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            float b_val = smem_BT[dot_idx * BN + thread_col];

            // Now we loop through G, reusing our b_val computing the rolling dot
            // for each of the grad_a values the thread is responsible for.
            for (int register_idx = 0; register_idx < TM; register_idx++) {
                thread_results[register_idx] += (
                    smem_G[(thread_row * TM + register_idx) * BK + dot_idx] * b_val
                );
            }
        }

        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_a (stride K, not N).
    for (int register_idx = 0; register_idx < TM; register_idx++) {
        int grad_a_row = thread_row * TM + register_idx;
        if (grad_a_batch < batch_size && grad_a_rows * BM + grad_a_row < M && col < K) {
            grad_a[grad_a_row * K + thread_col] = thread_results[register_idx];
        }
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
template <const int BM, const int BN, const int BK, const int TM>
__global__ void bwd_batched_matmul_b(
    const float* grad_out, const float* A, float* grad_b,
    int batch_size, int M, int N, int K
) {

    // Set the output tile that we need to compute for (grad_b is batch x K x N).
    const int grad_b_batch = blockIdx.z;
    const int grad_b_rows = blockIdx.y;
    const int grad_b_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    grad_out += grad_b_batch * M * N + grad_b_cols * BN; // jump batch, then cols of grad_out
    A += grad_b_batch * M * K + grad_b_rows * BM; // jump batch, then cols of A
    grad_b += grad_b_batch * K * N + grad_b_rows * BM * N + grad_b_cols * BN; // where the tile sits in the output

    // Shared memory.
    __shared__ float smem_AT[BM * BK]; // BM rows, BK cols
    __shared__ float smem_G[BK * BN];  // BK rows, BN cols

    // Thread positions within the block.
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Indexes for the load into smem.
    // AT tile is BM x BK, same mapping as A in the forward.
    const int row_at = threadIdx.x;
    const int col_at = threadIdx.y;

    // G tile is BK x BN, same mapping as B in the forward.
    const int row_g = threadIdx.y;
    const int col_g = threadIdx.x;

    // Global index (can be OOB) from the start.
    int row = grad_b_rows * BM + row_at; // k, for the AT load / grad_b rows
    int col = grad_b_cols * BN + col_g;  // n, output col of grad_b

    // Output for this specific thread grad_b[thread_row, thread_col]
    float thread_results[TM];
    for (int i = 0; i < TM; i++) {
        thread_results[i] = 0.f;
    }

    // Now we start iterating through M in tile steps (reduction axis).
    for (int t_m_idx = 0; t_m_idx < M; t_m_idx += BK) {

        // Check if the thread is still in bounds as we iterate the tile.
        int a_m = t_m_idx + col_at; // A[a_m, row]
        int g_m = t_m_idx + row_g; // grad_out[g_m, col]

        // real A[m, k] -> we store as if A^T[k, m]
        smem_AT[row_at * BK + col_at] =
            (a_m < M && row < K) ? A[col_at * K + row_at] : 0.f;

        smem_G[row_g * BN + col_g] =
            (g_m < M && col < N) ? grad_out[row_g * N + col_g] : 0.f;

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK * N; // move down a block (along M)
        A += BK * K; // move down a block (along M)

        // Use smem values to compute the rolling dot product.
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            float g_val = smem_G[dot_idx * BN + thread_col];

            // Now we loop through AT, reusing our g_val computing the rolling dot
            // for each of the grad_b values the thread is responsible for.
            for (int register_idx = 0; register_idx < TM; register_idx++) {
                thread_results[register_idx] += (
                    smem_AT[(thread_row * TM + register_idx) * BK + dot_idx] * g_val
                );
            }
        }

        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_b (stride N).
    for (int register_idx = 0; register_idx < TM; register_idx++) {
        int grad_b_row = thread_row * TM + register_idx;
        if (grad_b_batch < batch_size && grad_b_rows * BM + grad_b_row < K && col < N) {
            grad_b[grad_b_row * N + thread_col] = thread_results[register_idx];
        }
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

    // K-Tile, Registers (number of C values per thread), num of rows, num of cols in M and N per tile.
    const int BK = 8;
    const int TM = 8;
    const int BM = 64;
    const int BN = 64;

    dim3 threads_per_block(BN, BM / TM); // 64 x 8
    dim3 blocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    fwd_matmul<BM, BN, BK, TM><<<blocks, threads_per_block>>>(A, B, C, M, N, K);
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

    // Tile sizes
    const int BK = 8;
    const int TM = 8;
    const int BM = 64;
    const int BN = 64;

    dim3 threads_per_block(BN, BM / TM); // 64 x 8

    // Compute the backward for A.
    // Output is M x K
    dim3 blocks_a(
        (K + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    bwd_matmul_a<BM, BN, BK, TM><<<blocks_a, threads_per_block>>>(
        grad_out, B, grad_a, M, N, K
    );

    // Compute the backward for B.
    // Output is K x N
    dim3 blocks_b(
        (N + BN - 1) / BN,
        (K + BM - 1) / BM
    );
    bwd_matmul_b<BM, BN, BK, TM><<<blocks_b, threads_per_block>>>(
        grad_out, A, grad_b, M, N, K
    );
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
    // Tile shapes
    const int BK = 8;
    const int TM = 8;
    const int BM = 64;
    const int BN = 64;

    dim3 threads_per_block(BN, BM / TM); // 64 x 8
    dim3 blocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM,
        batch_size
    );
    fwd_batched_matmul<BM, BN, BK, TM><<<blocks, threads_per_block>>>(
        A, B, C, batch_size, M, N, K
    );
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

    // Tile shape
    const int BK = 8;
    const int TM = 8;
    const int BM = 64;
    const int BN = 64;

    dim3 threads_per_block(BN, BM / TM); // 64 x 8

    // Compute the backward for A.
    // Output is (batch size, M, K)
    dim3 blocks_a(
        (K + BN - 1) / BN,
        (M + BM - 1) / BM,
        batch_size
    );
    bwd_batched_matmul_a<BM, BN, BK, TM><<<blocks_a, threads_per_block>>>(
        grad_out, B, grad_a, batch_size, M, N, K
    );

    // Compute the backward for B.
    // Output is (batch size, K, N)
    dim3 blocks_b(
        (N + BN - 1) / BN,
        (K + BM - 1) / BM,
        batch_size
    );
    bwd_batched_matmul_b<BM, BN, BK, TM><<<blocks_b, threads_per_block>>>(
        grad_out, A, grad_b, batch_size, M, N, K
    );
}
