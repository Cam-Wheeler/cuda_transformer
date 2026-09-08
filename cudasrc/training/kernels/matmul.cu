/*
CUDA code for matrix multiply.
Within QWEN we will be using this in the attention and FFN layers.
*/

#include <cuda_runtime.h>

/*
Matmul forward pass.

Uses shared memory to reduce trips to HBM when performing matmul.
Additional register blocking (2D) to enable further reuse of elements increasing
FLOP per byte moved from HBM.

Y = A @ B:
    A is M x K
    B is K x N
    Y = M x N


@tparam BM: The block tile size in M dim (rows).
@tparam BN: The block tile size in N dim (cols).
@tparam BK: The block tile size in K dim (reduction).
@tparam TM: The number of elements in a row a thread is responsible for.
@tparam TN: The number of elements in a column a thread is responsible for.

@param A: Input matrix A (M x K)
@param B: Input matrix B (K x N)
@param C: Output matrix C (M x N)
@param M: The number of rows in A and C
@param N: The number of cols in B and C
@param K: The number of columns in A and rows in B.
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
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

    // Thread positions within the block (maps to C).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the load into smem (mapping to A and B).
    const int smem_row_a = threadIdx.x / BK;
    const int smem_col_a = threadIdx.x % BK;
    const int smem_row_b = threadIdx.x / BN;
    const int smem_col_b = threadIdx.x % BN;

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_m[TM]; // register array for A elements.
    float reg_n[TN]; // regist arrayt for B elements.

    // Stride for loading into smem (each thread is now loading several values).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_a = num_threads / BK; // Stride for loading in A.
    const int stride_b = num_threads / BN; // Stride for loading in B.

    // Now we start iterating through K in tile steps computing the total as we go.
    for (int t_k_idx = 0; t_k_idx < K; t_k_idx += BK) {


        // Load in a tile of A into smem.
        for (int load_offset = 0; load_offset < BM; load_offset += stride_a) {
            int a_row = smem_row_a + load_offset;
            int a_k   = t_k_idx + smem_col_a;
            smem_A[a_row * BK + smem_col_a] =
                (c_rows * BM + a_row < M && a_k < K)
                    ? A[a_row * K + smem_col_a]
                    : 0.f;
        }

        // Load in a tile of B into smem.
        for (int load_offset = 0; load_offset < BK; load_offset += stride_b) {
            int b_row = smem_row_b + load_offset;
            int b_k   = t_k_idx + b_row;
            smem_B[b_row * BN + smem_col_b] =
                (b_k < K && c_cols * BN + smem_col_b < N)
                    ? B[b_row * N + smem_col_b]
                    : 0.f;
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK; // move right a block
        B += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // Registers now hold a value from A and B, not just a single B value.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_m[i] = smem_A[(thread_row * TM + i) * BK + dot_idx];
            }
            for (int j = 0; j < TN; ++j) {
                reg_n[j] = smem_B[dot_idx * BN + thread_col * TN + j];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_m[m] * reg_n[n];
                }
            }
        }
    
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to C.
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int c_row = thread_row * TM + m;
            int c_col = thread_col * TN + n;
            if (c_rows * BM + c_row < M && c_cols * BN + c_col < N) {
                C[c_row * N + c_col] = thread_results[m * TN + n];
            }
        }
    }
}

/*
Standard matmul backward pass to compute the grad with respect to matrix A.

@tparam BM: The block tile size in M dim (rows).
@tparam BN: The block tile size in N dim (cols).
@tparam BK: The block tile size in K dim (reduction).
@tparam TM: The number of elements in a row a thread is responsible for.
@tparam TN: The number of elements in a column a thread is responsible for.

@param grad_out: The gradients of the outputs (M x N)
@param B: Input matrix B (K x N)
@param grad_a: The gradient with respsect to the input A (M x K)
@param M: The number of rows in grad_out and grad_a
@param N: The number of columns in B and grad_out
@param K: The number of rows in B and the number of columns in grad_A
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void bwd_matmul_a(const float* grad_out, const float* B, float* grad_a, int M, int N, int K) {

    // Set the output tile that we need to compute for!
    const int grad_a_rows = blockIdx.y;
    const int grad_a_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    grad_out += grad_a_rows * BM * N; // jump rows of grad_out
    B += grad_a_cols * BN * N; // jump rows of B (cols of B^T)
    grad_a += grad_a_rows * BM * K + grad_a_cols * BN; // where the tile sits in the output.

    // Shared memory to load the tiles into
    __shared__ float smem_grad_out[BM * BK]; // BM rows, BK cols
    __shared__ float smem_BT[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to grad_a).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the load into smem (mapping to grad_out and B).
    const int smem_row_grad_out = threadIdx.x / BK;
    const int smem_col_grad_out = threadIdx.x % BK;
    const int smem_row_bt = threadIdx.x / BN;
    const int smem_col_bt = threadIdx.x % BN;

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_grad_out[TM]; // register array for grad_out elements.
    float reg_bt[TN]; // register array for B^T elements.

    // Stride for loading into smem (each thread is now loading several values).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_grad_out = num_threads / BK; // Stride for loading in grad_out.
    const int stride_bt = num_threads / BN; // Stride for loading in B.

    // Now we start iterating through N in tile steps computing the total as we go.
    for (int t_n_idx = 0; t_n_idx < N; t_n_idx += BK) {


        // Load in a tile of grad_out into smem.
        for (int load_offset = 0; load_offset < BM; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_n   = t_n_idx + smem_col_grad_out;
            smem_grad_out[grad_out_row * BK + smem_col_grad_out] =
                (grad_a_rows * BM + grad_out_row < M && grad_out_n < N)
                    ? grad_out[grad_out_row * N + smem_col_grad_out]
                    : 0.f;
        }

        // Load in a tile of B into smem (store as B^T so layout matches the compute).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_bt) {
            int bt_row = smem_row_bt + load_offset;
            int b_n    = t_n_idx + bt_row;
            smem_BT[bt_row * BN + smem_col_bt] =
                (b_n < N && grad_a_cols * BN + smem_col_bt < K)
                    ? B[smem_col_bt * N + bt_row]
                    : 0.f;
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK; // move right a block
        B += BK; // move right a block

        // Use smem values to compute the rolling dot product.
        // Registers now hold a value from grad_out and B^T, not just a single B value.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_grad_out[i] = smem_grad_out[(thread_row * TM + i) * BK + dot_idx];
            }
            for (int j = 0; j < TN; ++j) {
                reg_bt[j] = smem_BT[dot_idx * BN + thread_col * TN + j];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_grad_out[m] * reg_bt[n];
                }
            }
        }
    
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_a.
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int grad_a_row = thread_row * TM + m;
            int grad_a_col = thread_col * TN + n;
            if (grad_a_rows * BM + grad_a_row < M && grad_a_cols * BN + grad_a_col < K) {
                grad_a[grad_a_row * K + grad_a_col] = thread_results[m * TN + n];
            }
        }
    }
}

/*
Standard matmul backward pass to compute the grad with respect to matrix B.

@tparam BM: The block tile size in M dim (rows).
@tparam BN: The block tile size in N dim (cols).
@tparam BK: The block tile size in K dim (reduction).
@tparam TM: The number of elements in a row a thread is responsible for.
@tparam TN: The number of elements in a column a thread is responsible for.

@param grad_out: The gradients of the outputs (M x N)
@param A: The input matrix A (M x K)
@param grad_b: The gradients with respect to the input B (K x N)
@param M: the number of rows in grad_out and A.
@param N: The number of columns in grad_out and B
@param K: The number of columns in A and rows in grad_B
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void bwd_matmul_b(const float* grad_out, const float* A, float* grad_b, int M, int N, int K) {

    // Set the output tile that we need to compute for!
    const int grad_b_rows = blockIdx.y;
    const int grad_b_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    A += grad_b_rows * BM; // jump cols of A (rows of A^T)
    grad_out += grad_b_cols * BN; // jump cols of grad_out
    grad_b += grad_b_rows * BM * N + grad_b_cols * BN; // where the tile sits in the output.

    // Shared memory to load the tiles into
    __shared__ float smem_AT[BM * BK]; // BM rows, BK cols
    __shared__ float smem_grad_out[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to grad_b).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the load into smem (mapping to A and grad_out).
    const int smem_row_at = threadIdx.x / BK;
    const int smem_col_at = threadIdx.x % BK;
    const int smem_row_grad_out = threadIdx.x / BN;
    const int smem_col_grad_out = threadIdx.x % BN;

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_at[TM]; // register array for A^T elements.
    float reg_grad_out[TN]; // register array for grad_out elements.

    // Stride for loading into smem (each thread is now loading several values).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_at = num_threads / BK; // Stride for loading in A.
    const int stride_grad_out = num_threads / BN; // Stride for loading in grad_out.

    // Now we start iterating through M in tile steps computing the total as we go.
    for (int t_m_idx = 0; t_m_idx < M; t_m_idx += BK) {


        // Load in a tile of A into smem (store as A^T so layout matches the compute).
        for (int load_offset = 0; load_offset < BM; load_offset += stride_at) {
            int at_row = smem_row_at + load_offset;
            int a_m    = t_m_idx + smem_col_at;
            smem_AT[at_row * BK + smem_col_at] =
                (grad_b_rows * BM + at_row < K && a_m < M)
                    ? A[smem_col_at * K + at_row]
                    : 0.f;
        }

        // Load in a tile of grad_out into smem.
        for (int load_offset = 0; load_offset < BK; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_m   = t_m_idx + grad_out_row;
            smem_grad_out[grad_out_row * BN + smem_col_grad_out] =
                (grad_out_m < M && grad_b_cols * BN + smem_col_grad_out < N)
                    ? grad_out[grad_out_row * N + smem_col_grad_out]
                    : 0.f;
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK * K; // move down a block
        grad_out += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // Registers now hold a value from A^T and grad_out, not just a single grad_out value.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_at[i] = smem_AT[(thread_row * TM + i) * BK + dot_idx];
            }
            for (int j = 0; j < TN; ++j) {
                reg_grad_out[j] = smem_grad_out[dot_idx * BN + thread_col * TN + j];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_at[m] * reg_grad_out[n];
                }
            }
        }
    
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_b.
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int grad_b_row = thread_row * TM + m;
            int grad_b_col = thread_col * TN + n;
            if (grad_b_rows * BM + grad_b_row < K && grad_b_cols * BN + grad_b_col < N) {
                grad_b[grad_b_row * N + grad_b_col] = thread_results[m * TN + n];
            }
        }
    }
}

/*
Forward pass for batched matrix multiplication used in attention.
Computes Y[batch] = A[batch] * B[batch] in parallel!

@tparam BM: The block tile size in M dim (rows).
@tparam BN: The block tile size in N dim (cols).
@tparam BK: The block tile size in K dim (reduction).
@tparam TM: The number of elements in a row a thread is responsible for.
@tparam TN: The number of elements in a column a thread is responsible for.

@param A: Input matrix A (batch size, M, K).
@param B: Input matrix B (batch size, K, N).
@param C: Input matrix C (out) (batch size, M, N)
@param batch_size: The size of the batch.
@param M: The number of rows in A and C.
@param N: The number of cols in B.
@param K: The number of cols in A and rows in B.
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
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
    C += c_batch * M * N + c_rows * BM * N + c_cols * BN; // where the tile sits in the output.

    // Shared memory to load the tiles into
    __shared__ float smem_A[BM * BK]; // BM rows, BK cols
    __shared__ float smem_B[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to C).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the load into smem (mapping to A and B).
    const int smem_row_a = threadIdx.x / BK;
    const int smem_col_a = threadIdx.x % BK;
    const int smem_row_b = threadIdx.x / BN;
    const int smem_col_b = threadIdx.x % BN;

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_m[TM]; // register array for A elements.
    float reg_n[TN]; // regist arrayt for B elements.

    // Stride for loading into smem (each thread is now loading several values).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_a = num_threads / BK; // Stride for loading in A.
    const int stride_b = num_threads / BN; // Stride for loading in B.

    // Now we start iterating through K in tile steps computing the total as we go.
    for (int t_k_idx = 0; t_k_idx < K; t_k_idx += BK) {


        // Load in a tile of A into smem.
        for (int load_offset = 0; load_offset < BM; load_offset += stride_a) {
            int a_row = smem_row_a + load_offset;
            int a_k   = t_k_idx + smem_col_a;
            smem_A[a_row * BK + smem_col_a] =
                (c_rows * BM + a_row < M && a_k < K)
                    ? A[a_row * K + smem_col_a]
                    : 0.f;
        }

        // Load in a tile of B into smem.
        for (int load_offset = 0; load_offset < BK; load_offset += stride_b) {
            int b_row = smem_row_b + load_offset;
            int b_k   = t_k_idx + b_row;
            smem_B[b_row * BN + smem_col_b] =
                (b_k < K && c_cols * BN + smem_col_b < N)
                    ? B[b_row * N + smem_col_b]
                    : 0.f;
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK; // move right a block
        B += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // Registers now hold a value from A and B, not just a single B value.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_m[i] = smem_A[(thread_row * TM + i) * BK + dot_idx];
            }
            for (int j = 0; j < TN; ++j) {
                reg_n[j] = smem_B[dot_idx * BN + thread_col * TN + j];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_m[m] * reg_n[n];
                }
            }
        }
    
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to C.
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int c_row = thread_row * TM + m;
            int c_col = thread_col * TN + n;
            if (c_batch < batch_size && c_rows * BM + c_row < M && c_cols * BN + c_col < N) {
                C[c_row * N + c_col] = thread_results[m * TN + n];
            }
        }
    }
}

/*
Backward pass for batched matmul to compute the gradients for A.

@tparam BM: The block tile size in M dim (rows).
@tparam BN: The block tile size in N dim (cols).
@tparam BK: The block tile size in K dim (reduction).
@tparam TM: The number of elements in a row a thread is responsible for.
@tparam TN: The number of elements in a column a thread is responsible for.

@param grad_out: The gradients of the outputs (batch size, M, N)
@param B: The input matrix B (batch size, K, N)
@param grad_a: The gradients with respect to the input A (batch size, M, K)
@param batch_size: The size of the batch.
@param M: The number of rows in grad_out and grad_a.
@param N: The number of columns in B and grad_out.
@param K: The number of columns in A and rows in grad_b.
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void bwd_batched_matmul_a(
    const float* grad_out, const float* B, float* grad_a,
    int batch_size, int M, int N, int K
) {

    // Set the output tile that we need to compute for!
    const int grad_a_batch = blockIdx.z;
    const int grad_a_rows = blockIdx.y;
    const int grad_a_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    grad_out += grad_a_batch * M * N + grad_a_rows * BM * N; // jump batch, then rows of grad_out
    B += grad_a_batch * K * N + grad_a_cols * BN * N; // jump batch, then rows of B (cols of B^T)
    grad_a += grad_a_batch * M * K + grad_a_rows * BM * K + grad_a_cols * BN; // where the tile sits in the output.

    // Shared memory to load the tiles into
    __shared__ float smem_grad_out[BM * BK]; // BM rows, BK cols
    __shared__ float smem_BT[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to grad_a).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the load into smem (mapping to grad_out and B).
    const int smem_row_grad_out = threadIdx.x / BK;
    const int smem_col_grad_out = threadIdx.x % BK;
    const int smem_row_bt = threadIdx.x / BN;
    const int smem_col_bt = threadIdx.x % BN;

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_grad_out[TM]; // register array for grad_out elements.
    float reg_bt[TN]; // register array for B^T elements.

    // Stride for loading into smem (each thread is now loading several values).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_grad_out = num_threads / BK; // Stride for loading in grad_out.
    const int stride_bt = num_threads / BN; // Stride for loading in B.

    // Now we start iterating through N in tile steps computing the total as we go.
    for (int t_n_idx = 0; t_n_idx < N; t_n_idx += BK) {


        // Load in a tile of grad_out into smem.
        for (int load_offset = 0; load_offset < BM; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_n   = t_n_idx + smem_col_grad_out;
            smem_grad_out[grad_out_row * BK + smem_col_grad_out] =
                (grad_a_rows * BM + grad_out_row < M && grad_out_n < N)
                    ? grad_out[grad_out_row * N + smem_col_grad_out]
                    : 0.f;
        }

        // Load in a tile of B into smem (store as B^T so layout matches the compute).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_bt) {
            int bt_row = smem_row_bt + load_offset;
            int b_n    = t_n_idx + bt_row;
            smem_BT[bt_row * BN + smem_col_bt] =
                (b_n < N && grad_a_cols * BN + smem_col_bt < K)
                    ? B[smem_col_bt * N + bt_row]
                    : 0.f;
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK; // move right a block
        B += BK; // move right a block

        // Use smem values to compute the rolling dot product.
        // Registers now hold a value from grad_out and B^T, not just a single B value.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_grad_out[i] = smem_grad_out[(thread_row * TM + i) * BK + dot_idx];
            }
            for (int j = 0; j < TN; ++j) {
                reg_bt[j] = smem_BT[dot_idx * BN + thread_col * TN + j];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_grad_out[m] * reg_bt[n];
                }
            }
        }
    
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_a.
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int grad_a_row = thread_row * TM + m;
            int grad_a_col = thread_col * TN + n;
            if (grad_a_batch < batch_size && grad_a_rows * BM + grad_a_row < M && grad_a_cols * BN + grad_a_col < K) {
                grad_a[grad_a_row * K + grad_a_col] = thread_results[m * TN + n];
            }
        }
    }
}

/*
Backward pass for batched matmul to compute gradients for B.

@tparam BM: The block tile size in M dim (rows).
@tparam BN: The block tile size in N dim (cols).
@tparam BK: The block tile size in K dim (reduction).
@tparam TM: The number of elements in a row a thread is responsible for.
@tparam TN: The number of elements in a column a thread is responsible for.

@param grad_out: The gradients of the outputs (batch size, M, N)
@param A: The input matrix A (batch size, M, K)
@param grad_b: The gradients with respect to the input B (batch size, K, N)
@param batch_size: The size of the batch.
@param M: The number of rows in grad_out and A.
@param N: The number of columns in grad_out and B.
@param K: The number of columns in A and rows in grad_b.
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void bwd_batched_matmul_b(
    const float* grad_out, const float* A, float* grad_b,
    int batch_size, int M, int N, int K
) {

    // Set the output tile that we need to compute for!
    const int grad_b_batch = blockIdx.z;
    const int grad_b_rows = blockIdx.y;
    const int grad_b_cols = blockIdx.x;

    // Update the pointers so the tile is in the correct position to start looping.
    A += grad_b_batch * M * K + grad_b_rows * BM; // jump batch, then cols of A (rows of A^T)
    grad_out += grad_b_batch * M * N + grad_b_cols * BN; // jump batch, then cols of grad_out
    grad_b += grad_b_batch * K * N + grad_b_rows * BM * N + grad_b_cols * BN; // where the tile sits in the output.

    // Shared memory to load the tiles into
    __shared__ float smem_AT[BM * BK]; // BM rows, BK cols
    __shared__ float smem_grad_out[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to grad_b).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the load into smem (mapping to A and grad_out).
    const int smem_row_at = threadIdx.x / BK;
    const int smem_col_at = threadIdx.x % BK;
    const int smem_row_grad_out = threadIdx.x / BN;
    const int smem_col_grad_out = threadIdx.x % BN;

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_at[TM]; // register array for A^T elements.
    float reg_grad_out[TN]; // register array for grad_out elements.

    // Stride for loading into smem (each thread is now loading several values).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_at = num_threads / BK; // Stride for loading in A.
    const int stride_grad_out = num_threads / BN; // Stride for loading in grad_out.

    // Now we start iterating through M in tile steps computing the total as we go.
    for (int t_m_idx = 0; t_m_idx < M; t_m_idx += BK) {


        // Load in a tile of A into smem (store as A^T so layout matches the compute).
        for (int load_offset = 0; load_offset < BM; load_offset += stride_at) {
            int at_row = smem_row_at + load_offset;
            int a_m    = t_m_idx + smem_col_at;
            smem_AT[at_row * BK + smem_col_at] =
                (grad_b_rows * BM + at_row < K && a_m < M)
                    ? A[smem_col_at * K + at_row]
                    : 0.f;
        }

        // Load in a tile of grad_out into smem.
        for (int load_offset = 0; load_offset < BK; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_m   = t_m_idx + grad_out_row;
            smem_grad_out[grad_out_row * BN + smem_col_grad_out] =
                (grad_out_m < M && grad_b_cols * BN + smem_col_grad_out < N)
                    ? grad_out[grad_out_row * N + smem_col_grad_out]
                    : 0.f;
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK * K; // move down a block
        grad_out += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // Registers now hold a value from A^T and grad_out, not just a single grad_out value.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_at[i] = smem_AT[(thread_row * TM + i) * BK + dot_idx];
            }
            for (int j = 0; j < TN; ++j) {
                reg_grad_out[j] = smem_grad_out[dot_idx * BN + thread_col * TN + j];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_at[m] * reg_grad_out[n];
                }
            }
        }
    
        // Ensure all the threads are done working.
        __syncthreads();
    }

    // Write to grad_b.
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int grad_b_row = thread_row * TM + m;
            int grad_b_col = thread_col * TN + n;
            if (grad_b_batch < batch_size && grad_b_rows * BM + grad_b_row < K && grad_b_cols * BN + grad_b_col < N) {
                grad_b[grad_b_row * N + grad_b_col] = thread_results[m * TN + n];
            }
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
    const int TN = 8;
    const int BM = 128;
    const int BN = 128;

    dim3 threads_per_block((BM * BN) / (TM * TN));  // 256, 1D thread dim.
    dim3 blocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    fwd_matmul<BM, BN, BK, TM, TN><<<blocks, threads_per_block>>>(A, B, C, M, N, K);
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

    // K-Tile, Registers (number of values per thread), num of rows, num of cols in M and N per tile.
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int BM = 128;
    const int BN = 128;

    dim3 threads_per_block((BM * BN) / (TM * TN));  // 256, 1D thread dim.

    // Compute the backward for A.
    // Output is M x K
    dim3 blocks_a(
        (K + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    bwd_matmul_a<BM, BN, BK, TM, TN><<<blocks_a, threads_per_block>>>(
        grad_out, B, grad_a, M, N, K
    );

    // Compute the backward for B.
    // Output is K x N
    dim3 blocks_b(
        (N + BN - 1) / BN,
        (K + BM - 1) / BM
    );
    bwd_matmul_b<BM, BN, BK, TM, TN><<<blocks_b, threads_per_block>>>(
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

    // K-Tile, Registers (number of C values per thread), num of rows, num of cols in M and N per tile.
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int BM = 128;
    const int BN = 128;

    dim3 threads_per_block((BM * BN) / (TM * TN));  // 256, 1D thread dim.
    dim3 blocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM,
        batch_size
    );
    fwd_batched_matmul<BM, BN, BK, TM, TN><<<blocks, threads_per_block>>>(
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

    // K-Tile, Registers (number of values per thread), num of rows, num of cols in M and N per tile.
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int BM = 128;
    const int BN = 128;

    dim3 threads_per_block((BM * BN) / (TM * TN));  // 256, 1D thread dim.

    // Compute the backward for A.
    // Output is (batch size, M, K)
    dim3 blocks_a(
        (K + BN - 1) / BN,
        (M + BM - 1) / BM,
        batch_size
    );
    bwd_batched_matmul_a<BM, BN, BK, TM, TN><<<blocks_a, threads_per_block>>>(
        grad_out, B, grad_a, batch_size, M, N, K
    );

    // Compute the backward for B.
    // Output is (batch size, K, N)
    dim3 blocks_b(
        (N + BN - 1) / BN,
        (K + BM - 1) / BM,
        batch_size
    );
    bwd_batched_matmul_b<BM, BN, BK, TM, TN><<<blocks_b, threads_per_block>>>(
        grad_out, A, grad_b, batch_size, M, N, K
    );
}
