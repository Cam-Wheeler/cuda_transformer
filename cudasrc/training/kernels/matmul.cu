/*
CUDA code for matrix multiply.
Within QWEN we will be using this in the attention and FFN layers.

Uses shared memory to reduce trips to HBM when performing matmul.
Additional register blocking (2D) to enable further reuse of elements increasing
FLOP per byte moved from HBM. 

We also use vectorised loads to decrease the number of fetch instructions.
*/

#include <cuda_runtime.h>

/*
Matmul forward pass.

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

    // Shared memory to load the tiles into.
    // smem_A is now stored transposed (BK x BM) so the inner loop can issue wide SMEM loads.
    __shared__ float smem_A[BK * BM]; // BK rows, BM cols (transposed)
    __shared__ float smem_B[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to C).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the vectorised load into smem (mapping to A and B).
    // col is a float4 slot along the contiguous dim, not a single float.
    const int smem_row_a = threadIdx.x / (BK / 4);
    const int smem_col_a = threadIdx.x % (BK / 4);
    const int smem_row_b = threadIdx.x / (BN / 4);
    const int smem_col_b = threadIdx.x % (BN / 4);

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_m[TM]; // register array for A elements.
    float reg_n[TN]; // regist arrayt for B elements.

    // Stride for loading into smem (each thread is now loading a float4).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_a = num_threads / (BK / 4); // Stride for loading in A.
    const int stride_b = num_threads / (BN / 4); // Stride for loading in B.

    // Now we start iterating through K in tile steps computing the total as we go.
    for (int t_k_idx = 0; t_k_idx < K; t_k_idx += BK) {


        // Load in a tile of A into smem (vectorised along K, store transposed).
        for (int load_offset = 0; load_offset < BM; load_offset += stride_a) {
            int a_row = smem_row_a + load_offset;
            int a_k = t_k_idx + smem_col_a * 4;
            bool can_vectorize_a =
                (c_rows * BM + a_row < M) &&
                (a_k + 4 <= K) &&
                ((a_row * K + smem_col_a * 4) % 4 == 0);
            if (can_vectorize_a) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[a_row * K + smem_col_a * 4]
                )[0];
                smem_A[(smem_col_a * 4 + 0) * BM + a_row] = tmp.x;
                smem_A[(smem_col_a * 4 + 1) * BM + a_row] = tmp.y;
                smem_A[(smem_col_a * 4 + 2) * BM + a_row] = tmp.z;
                smem_A[(smem_col_a * 4 + 3) * BM + a_row] = tmp.w;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int a_k_v = a_k + v;
                    smem_A[(smem_col_a * 4 + v) * BM + a_row] =
                        (c_rows * BM + a_row < M && a_k_v < K)
                            ? A[a_row * K + smem_col_a * 4 + v]
                            : 0.f;
                }
            }
        }
        // Load in a tile of B into smem (vectorised along N).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_b) {
            int b_row = smem_row_b + load_offset;
            int b_k = t_k_idx + b_row;
            int b_n = c_cols * BN + smem_col_b * 4;
            bool can_vectorize_b =
                (b_k < K) &&
                (b_n + 4 <= N) &&
                ((b_row * N + smem_col_b * 4) % 4 == 0);
            if (can_vectorize_b) {
                reinterpret_cast<float4*>(
                    &smem_B[b_row * BN + smem_col_b * 4]
                )[0] =
                    reinterpret_cast<const float4*>(
                        &B[b_row * N + smem_col_b * 4]
                    )[0];
            } else {
                for (int v = 0; v < 4; ++v) {
                    int b_n_v = b_n + v;
                    smem_B[b_row * BN + smem_col_b * 4 + v] =
                        (b_k < K && b_n_v < N)
                            ? B[b_row * N + smem_col_b * 4 + v]
                            : 0.f;
                }
            }
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK; // move right a block
        B += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // smem_A is K-major so reg_m walks a contiguous row.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_m[i] = smem_A[dot_idx * BM + thread_row * TM + i];
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

    // Write to C (vectorised along N when the float4 is in-bounds).
    for (int m = 0; m < TM; ++m) {
        int c_row = thread_row * TM + m;
        bool row_ok = (c_rows * BM + c_row < M);
        for (int n = 0; n < TN; n += 4) {
            int c_col = thread_col * TN + n;
            bool can_vectorize_c =
                row_ok &&
                (c_cols * BN + c_col + 4 <= N) &&
                ((c_row * N + c_col) % 4 == 0);
            if (can_vectorize_c) {
                float4 tmp;
                tmp.x = thread_results[m * TN + n + 0];
                tmp.y = thread_results[m * TN + n + 1];
                tmp.z = thread_results[m * TN + n + 2];
                tmp.w = thread_results[m * TN + n + 3];
                reinterpret_cast<float4*>(&C[c_row * N + c_col])[0] = tmp;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int c_col_v = c_col + v;
                    if (row_ok && c_cols * BN + c_col_v < N) {
                        C[c_row * N + c_col_v] = thread_results[m * TN + n + v];
                    }
                }
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

    // Shared memory to load the tiles into.
    // smem_grad_out is stored transposed (BK x BM) so the inner loop can issue wide SMEM loads.
    __shared__ float smem_grad_out[BK * BM]; // BK rows, BM cols (transposed)
    __shared__ float smem_BT[BK * BN]; // BK rows, BN cols (B^T)

    // Thread positions within the block (maps to grad_a).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the vectorised load into smem.
    // col is a float4 slot along N (contiguous in grad_out and in B).
    const int smem_row_grad_out = threadIdx.x / (BK / 4);
    const int smem_col_grad_out = threadIdx.x % (BK / 4);
    const int smem_row_bt = threadIdx.x / (BK / 4);
    const int smem_col_bt = threadIdx.x % (BK / 4);

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_grad_out[TM]; // register array for grad_out elements.
    float reg_bt[TN]; // register array for B^T elements.

    // Stride for loading into smem (each thread is now loading a float4).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_grad_out = num_threads / (BK / 4); // Stride for loading in grad_out.
    const int stride_bt = num_threads / (BK / 4); // Stride for loading in B.

    // Now we start iterating through N in tile steps computing the total as we go.
    for (int t_n_idx = 0; t_n_idx < N; t_n_idx += BK) {


        // Load in a tile of grad_out into smem (vectorised along N, store transposed).
        for (int load_offset = 0; load_offset < BM; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_n = t_n_idx + smem_col_grad_out * 4;
            bool can_vectorize_grad_out =
                (grad_a_rows * BM + grad_out_row < M) &&
                (grad_out_n + 4 <= N) &&
                ((grad_out_row * N + smem_col_grad_out * 4) % 4 == 0);
            if (can_vectorize_grad_out) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &grad_out[grad_out_row * N + smem_col_grad_out * 4]
                )[0];
                smem_grad_out[(smem_col_grad_out * 4 + 0) * BM + grad_out_row] = tmp.x;
                smem_grad_out[(smem_col_grad_out * 4 + 1) * BM + grad_out_row] = tmp.y;
                smem_grad_out[(smem_col_grad_out * 4 + 2) * BM + grad_out_row] = tmp.z;
                smem_grad_out[(smem_col_grad_out * 4 + 3) * BM + grad_out_row] = tmp.w;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_out_n_v = grad_out_n + v;
                    smem_grad_out[(smem_col_grad_out * 4 + v) * BM + grad_out_row] =
                        (grad_a_rows * BM + grad_out_row < M && grad_out_n_v < N)
                            ? grad_out[grad_out_row * N + smem_col_grad_out * 4 + v]
                            : 0.f;
                }
            }
        }
        // Load in a tile of B into smem (vectorised along N, scatter into B^T layout).
        for (int load_offset = 0; load_offset < BN; load_offset += stride_bt) {
            int bt_k = smem_row_bt + load_offset;
            int b_n = t_n_idx + smem_col_bt * 4;
            bool can_vectorize_b =
                (b_n + 4 <= N) &&
                (grad_a_cols * BN + bt_k < K) &&
                ((bt_k * N + smem_col_bt * 4) % 4 == 0);
            if (can_vectorize_b) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &B[bt_k * N + smem_col_bt * 4]
                )[0];
                smem_BT[(smem_col_bt * 4 + 0) * BN + bt_k] = tmp.x;
                smem_BT[(smem_col_bt * 4 + 1) * BN + bt_k] = tmp.y;
                smem_BT[(smem_col_bt * 4 + 2) * BN + bt_k] = tmp.z;
                smem_BT[(smem_col_bt * 4 + 3) * BN + bt_k] = tmp.w;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int b_n_v = b_n + v;
                    smem_BT[(smem_col_bt * 4 + v) * BN + bt_k] =
                        (b_n_v < N && grad_a_cols * BN + bt_k < K)
                            ? B[bt_k * N + smem_col_bt * 4 + v]
                            : 0.f;
                }
            }
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK; // move right a block
        B += BK; // move right a block

        // Use smem values to compute the rolling dot product.
        // smem_grad_out is N-major so reg_grad_out walks a contiguous row.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_grad_out[i] = smem_grad_out[dot_idx * BM + thread_row * TM + i];
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


    // Write to grad_a (vectorised along K when the float4 is in-bounds).
    for (int m = 0; m < TM; ++m) {
        int grad_a_row = thread_row * TM + m;
        bool row_ok = (grad_a_rows * BM + grad_a_row < M);
        for (int n = 0; n < TN; n += 4) {
            int grad_a_col = thread_col * TN + n;
            bool can_vectorize_grad_a =
                row_ok &&
                (grad_a_cols * BN + grad_a_col + 4 <= K) &&
                ((grad_a_row * K + grad_a_col) % 4 == 0);
            if (can_vectorize_grad_a) {
                float4 tmp;
                tmp.x = thread_results[m * TN + n + 0];
                tmp.y = thread_results[m * TN + n + 1];
                tmp.z = thread_results[m * TN + n + 2];
                tmp.w = thread_results[m * TN + n + 3];
                reinterpret_cast<float4*>(&grad_a[grad_a_row * K + grad_a_col])[0] = tmp;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_a_col_v = grad_a_col + v;
                    if (row_ok && grad_a_cols * BN + grad_a_col_v < K) {
                        grad_a[grad_a_row * K + grad_a_col_v] = thread_results[m * TN + n + v];
                    }
                }
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

    // Shared memory to load the tiles into.
    // smem_AT is stored as BK x BM so the inner loop can issue wide SMEM loads along K.
    __shared__ float smem_AT[BK * BM]; // BK rows, BM cols
    __shared__ float smem_grad_out[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to grad_b).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the vectorised load into smem.
    // A: float4 slot along K (contiguous). grad_out: float4 slot along N (contiguous).
    const int smem_row_at = threadIdx.x / (BM / 4);
    const int smem_col_at = threadIdx.x % (BM / 4);
    const int smem_row_grad_out = threadIdx.x / (BN / 4);
    const int smem_col_grad_out = threadIdx.x % (BN / 4);

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_at[TM]; // register array for A^T elements.
    float reg_grad_out[TN]; // register array for grad_out elements.

    // Stride for loading into smem (each thread is now loading a float4).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_at = num_threads / (BM / 4); // Stride for loading in A.
    const int stride_grad_out = num_threads / (BN / 4); // Stride for loading in grad_out.

    // Now we start iterating through M in tile steps computing the total as we go.
    for (int t_m_idx = 0; t_m_idx < M; t_m_idx += BK) {


        // Load in a tile of A into smem (vectorised along K, store M-major).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_at) {
            int at_m = smem_row_at + load_offset;
            int a_k = grad_b_rows * BM + smem_col_at * 4;
            bool can_vectorize_a =
                (t_m_idx + at_m < M) &&
                (a_k + 4 <= K) &&
                ((at_m * K + smem_col_at * 4) % 4 == 0);

            if (can_vectorize_a) {
                reinterpret_cast<float4*>(
                    &smem_AT[at_m * BM + smem_col_at * 4]
                )[0] =
                    reinterpret_cast<const float4*>(
                        &A[at_m * K + smem_col_at * 4]
                    )[0];
            } else {
                for (int v = 0; v < 4; ++v) {
                    int a_k_v = a_k + v;
                    smem_AT[at_m * BM + smem_col_at * 4 + v] =
                        (t_m_idx + at_m < M && a_k_v < K)
                            ? A[at_m * K + smem_col_at * 4 + v]
                            : 0.f;
                }
            }
        }

        // Load in a tile of grad_out into smem (vectorised along N).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_m = t_m_idx + grad_out_row;
            int grad_out_n = grad_b_cols * BN + smem_col_grad_out * 4;
            bool can_vectorize_grad_out =
                (grad_out_m < M) &&
                (grad_out_n + 4 <= N) &&
                ((grad_out_row * N + smem_col_grad_out * 4) % 4 == 0);

            if (can_vectorize_grad_out) {
                reinterpret_cast<float4*>(
                    &smem_grad_out[grad_out_row * BN + smem_col_grad_out * 4]
                )[0] =
                    reinterpret_cast<const float4*>(
                        &grad_out[grad_out_row * N + smem_col_grad_out * 4]
                    )[0];
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_out_n_v = grad_out_n + v;
                    smem_grad_out[grad_out_row * BN + smem_col_grad_out * 4 + v] =
                        (grad_out_m < M && grad_out_n_v < N)
                            ? grad_out[grad_out_row * N + smem_col_grad_out * 4 + v]
                            : 0.f;
                }
            }
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK * K; // move down a block
        grad_out += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // smem_AT is M-major so reg_at walks a contiguous row along K.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_at[i] = smem_AT[dot_idx * BM + thread_row * TM + i];
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

    // Write to grad_b (vectorised along N when the float4 is in-bounds).
    for (int m = 0; m < TM; ++m) {
        int grad_b_row = thread_row * TM + m;
        bool row_ok = (grad_b_rows * BM + grad_b_row < K);
        for (int n = 0; n < TN; n += 4) {
            int grad_b_col = thread_col * TN + n;
            bool can_vectorize_grad_b =
                row_ok &&
                (grad_b_cols * BN + grad_b_col + 4 <= N) &&
                ((grad_b_row * N + grad_b_col) % 4 == 0);
            if (can_vectorize_grad_b) {
                float4 tmp;
                tmp.x = thread_results[m * TN + n + 0];
                tmp.y = thread_results[m * TN + n + 1];
                tmp.z = thread_results[m * TN + n + 2];
                tmp.w = thread_results[m * TN + n + 3];
                reinterpret_cast<float4*>(&grad_b[grad_b_row * N + grad_b_col])[0] = tmp;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_b_col_v = grad_b_col + v;
                    if (row_ok && grad_b_cols * BN + grad_b_col_v < N) {
                        grad_b[grad_b_row * N + grad_b_col_v] = thread_results[m * TN + n + v];
                    }
                }
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

    // Shared memory to load the tiles into.
    // smem_A is now stored transposed (BK x BM) so the inner loop can issue wide SMEM loads.
    __shared__ float smem_A[BK * BM]; // BK rows, BM cols (transposed)
    __shared__ float smem_B[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to C).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the vectorised load into smem (mapping to A and B).
    // col is a float4 slot along the contiguous dim, not a single float.
    const int smem_row_a = threadIdx.x / (BK / 4);
    const int smem_col_a = threadIdx.x % (BK / 4);
    const int smem_row_b = threadIdx.x / (BN / 4);
    const int smem_col_b = threadIdx.x % (BN / 4);

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_m[TM]; // register array for A elements.
    float reg_n[TN]; // regist arrayt for B elements.

    // Stride for loading into smem (each thread is now loading a float4).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_a = num_threads / (BK / 4); // Stride for loading in A.
    const int stride_b = num_threads / (BN / 4); // Stride for loading in B.

    // Now we start iterating through K in tile steps computing the total as we go.
    for (int t_k_idx = 0; t_k_idx < K; t_k_idx += BK) {


        // Load in a tile of A into smem (vectorised along K, store transposed).
        for (int load_offset = 0; load_offset < BM; load_offset += stride_a) {
            int a_row = smem_row_a + load_offset;
            int a_k = t_k_idx + smem_col_a * 4;
            bool can_vectorize_a =
                (c_rows * BM + a_row < M) &&
                (a_k + 4 <= K) &&
                ((c_batch * M * K + a_row * K + smem_col_a * 4) % 4 == 0);
            if (can_vectorize_a) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[a_row * K + smem_col_a * 4]
                )[0];
                smem_A[(smem_col_a * 4 + 0) * BM + a_row] = tmp.x;
                smem_A[(smem_col_a * 4 + 1) * BM + a_row] = tmp.y;
                smem_A[(smem_col_a * 4 + 2) * BM + a_row] = tmp.z;
                smem_A[(smem_col_a * 4 + 3) * BM + a_row] = tmp.w;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int a_k_v = a_k + v;
                    smem_A[(smem_col_a * 4 + v) * BM + a_row] =
                        (c_rows * BM + a_row < M && a_k_v < K)
                            ? A[a_row * K + smem_col_a * 4 + v]
                            : 0.f;
                }
            }
        }
        // Load in a tile of B into smem (vectorised along N).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_b) {
            int b_row = smem_row_b + load_offset;
            int b_k = t_k_idx + b_row;
            int b_n = c_cols * BN + smem_col_b * 4;
            bool can_vectorize_b =
                (b_k < K) &&
                (b_n + 4 <= N) &&
                ((c_batch * K * N + b_row * N + smem_col_b * 4) % 4 == 0);
            if (can_vectorize_b) {
                reinterpret_cast<float4*>(
                    &smem_B[b_row * BN + smem_col_b * 4]
                )[0] =
                    reinterpret_cast<const float4*>(
                        &B[b_row * N + smem_col_b * 4]
                    )[0];
            } else {
                for (int v = 0; v < 4; ++v) {
                    int b_n_v = b_n + v;
                    smem_B[b_row * BN + smem_col_b * 4 + v] =
                        (b_k < K && b_n_v < N)
                            ? B[b_row * N + smem_col_b * 4 + v]
                            : 0.f;
                }
            }
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK; // move right a block
        B += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // smem_A is K-major so reg_m walks a contiguous row.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_m[i] = smem_A[dot_idx * BM + thread_row * TM + i];
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

    // Write to C (vectorised along N when the float4 is in-bounds).
    for (int m = 0; m < TM; ++m) {
        int c_row = thread_row * TM + m;
        bool row_ok = (c_batch < batch_size && c_rows * BM + c_row < M);
        for (int n = 0; n < TN; n += 4) {
            int c_col = thread_col * TN + n;
            bool can_vectorize_c =
                row_ok &&
                (c_cols * BN + c_col + 4 <= N) &&
                ((c_batch * M * N + c_row * N + c_col) % 4 == 0);
            if (can_vectorize_c) {
                float4 tmp;
                tmp.x = thread_results[m * TN + n + 0];
                tmp.y = thread_results[m * TN + n + 1];
                tmp.z = thread_results[m * TN + n + 2];
                tmp.w = thread_results[m * TN + n + 3];
                reinterpret_cast<float4*>(&C[c_row * N + c_col])[0] = tmp;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int c_col_v = c_col + v;
                    if (row_ok && c_cols * BN + c_col_v < N) {
                        C[c_row * N + c_col_v] = thread_results[m * TN + n + v];
                    }
                }
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

    // Shared memory to load the tiles into.
    // smem_grad_out is stored transposed (BK x BM) so the inner loop can issue wide SMEM loads.
    __shared__ float smem_grad_out[BK * BM]; // BK rows, BM cols (transposed)
    __shared__ float smem_BT[BK * BN]; // BK rows, BN cols (B^T)

    // Thread positions within the block (maps to grad_a).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the vectorised load into smem.
    // col is a float4 slot along N (contiguous in grad_out and in B).
    const int smem_row_grad_out = threadIdx.x / (BK / 4);
    const int smem_col_grad_out = threadIdx.x % (BK / 4);
    const int smem_row_bt = threadIdx.x / (BK / 4);
    const int smem_col_bt = threadIdx.x % (BK / 4);

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_grad_out[TM]; // register array for grad_out elements.
    float reg_bt[TN]; // register array for B^T elements.

    // Stride for loading into smem (each thread is now loading a float4).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_grad_out = num_threads / (BK / 4); // Stride for loading in grad_out.
    const int stride_bt = num_threads / (BK / 4); // Stride for loading in B.

    // Now we start iterating through N in tile steps computing the total as we go.
    for (int t_n_idx = 0; t_n_idx < N; t_n_idx += BK) {


        // Load in a tile of grad_out into smem (vectorised along N, store transposed).
        for (int load_offset = 0; load_offset < BM; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_n = t_n_idx + smem_col_grad_out * 4;
            bool can_vectorize_grad_out =
                (grad_a_rows * BM + grad_out_row < M) &&
                (grad_out_n + 4 <= N) &&
                ((grad_a_batch * M * N + grad_out_row * N + smem_col_grad_out * 4) % 4 == 0);
            if (can_vectorize_grad_out) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &grad_out[grad_out_row * N + smem_col_grad_out * 4]
                )[0];
                smem_grad_out[(smem_col_grad_out * 4 + 0) * BM + grad_out_row] = tmp.x;
                smem_grad_out[(smem_col_grad_out * 4 + 1) * BM + grad_out_row] = tmp.y;
                smem_grad_out[(smem_col_grad_out * 4 + 2) * BM + grad_out_row] = tmp.z;
                smem_grad_out[(smem_col_grad_out * 4 + 3) * BM + grad_out_row] = tmp.w;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_out_n_v = grad_out_n + v;
                    smem_grad_out[(smem_col_grad_out * 4 + v) * BM + grad_out_row] =
                        (grad_a_rows * BM + grad_out_row < M && grad_out_n_v < N)
                            ? grad_out[grad_out_row * N + smem_col_grad_out * 4 + v]
                            : 0.f;
                }
            }
        }
        // Load in a tile of B into smem (vectorised along N, scatter into B^T layout).
        for (int load_offset = 0; load_offset < BN; load_offset += stride_bt) {
            int bt_k = smem_row_bt + load_offset;
            int b_n = t_n_idx + smem_col_bt * 4;
            bool can_vectorize_b =
                (b_n + 4 <= N) &&
                (grad_a_cols * BN + bt_k < K) &&
                ((grad_a_batch * K * N + bt_k * N + smem_col_bt * 4) % 4 == 0);
            if (can_vectorize_b) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &B[bt_k * N + smem_col_bt * 4]
                )[0];
                smem_BT[(smem_col_bt * 4 + 0) * BN + bt_k] = tmp.x;
                smem_BT[(smem_col_bt * 4 + 1) * BN + bt_k] = tmp.y;
                smem_BT[(smem_col_bt * 4 + 2) * BN + bt_k] = tmp.z;
                smem_BT[(smem_col_bt * 4 + 3) * BN + bt_k] = tmp.w;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int b_n_v = b_n + v;
                    smem_BT[(smem_col_bt * 4 + v) * BN + bt_k] =
                        (b_n_v < N && grad_a_cols * BN + bt_k < K)
                            ? B[bt_k * N + smem_col_bt * 4 + v]
                            : 0.f;
                }
            }
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        grad_out += BK; // move right a block
        B += BK; // move right a block

        // Use smem values to compute the rolling dot product.
        // smem_grad_out is N-major so reg_grad_out walks a contiguous row.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_grad_out[i] = smem_grad_out[dot_idx * BM + thread_row * TM + i];
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

    // Write to grad_a (vectorised along K when the float4 is in-bounds).
    for (int m = 0; m < TM; ++m) {
        int grad_a_row = thread_row * TM + m;
        bool row_ok = (grad_a_batch < batch_size && grad_a_rows * BM + grad_a_row < M);
        for (int n = 0; n < TN; n += 4) {
            int grad_a_col = thread_col * TN + n;
            bool can_vectorize_grad_a =
                row_ok &&
                (grad_a_cols * BN + grad_a_col + 4 <= K) &&
                ((grad_a_batch * M * K + grad_a_row * K + grad_a_col) % 4 == 0);
            if (can_vectorize_grad_a) {
                float4 tmp;
                tmp.x = thread_results[m * TN + n + 0];
                tmp.y = thread_results[m * TN + n + 1];
                tmp.z = thread_results[m * TN + n + 2];
                tmp.w = thread_results[m * TN + n + 3];
                reinterpret_cast<float4*>(&grad_a[grad_a_row * K + grad_a_col])[0] = tmp;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_a_col_v = grad_a_col + v;
                    if (row_ok && grad_a_cols * BN + grad_a_col_v < K) {
                        grad_a[grad_a_row * K + grad_a_col_v] = thread_results[m * TN + n + v];
                    }
                }
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

    // Shared memory to load the tiles into.
    // smem_AT is stored as BK x BM so the inner loop can issue wide SMEM loads along K.
    __shared__ float smem_AT[BK * BM]; // BK rows, BM cols
    __shared__ float smem_grad_out[BK * BN]; // BK rows, BN cols

    // Thread positions within the block (maps to grad_b).
    const int thread_idx = threadIdx.x;
    const int thread_row = thread_idx / (BN / TN);
    const int thread_col = thread_idx % (BN / TN);

    // Indexes for the vectorised load into smem.
    // A: float4 slot along K (contiguous). grad_out: float4 slot along N (contiguous).
    const int smem_row_at = threadIdx.x / (BM / 4);
    const int smem_col_at = threadIdx.x % (BM / 4);
    const int smem_row_grad_out = threadIdx.x / (BN / 4);
    const int smem_col_grad_out = threadIdx.x % (BN / 4);

    // Outputs for this specific thread [TM * TN] values
    float thread_results[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        thread_results[i] = 0.f; // init to 0.
    }
    float reg_at[TM]; // register array for A^T elements.
    float reg_grad_out[TN]; // register array for grad_out elements.

    // Stride for loading into smem (each thread is now loading a float4).
    const int num_threads = (BM * BN) / (TM * TN);
    const int stride_at = num_threads / (BM / 4); // Stride for loading in A.
    const int stride_grad_out = num_threads / (BN / 4); // Stride for loading in grad_out.

    // Now we start iterating through M in tile steps computing the total as we go.
    for (int t_m_idx = 0; t_m_idx < M; t_m_idx += BK) {


        // Load in a tile of A into smem (vectorised along K, store M-major).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_at) {
            int at_m = smem_row_at + load_offset;
            int a_k = grad_b_rows * BM + smem_col_at * 4;
            bool can_vectorize_a =
                (t_m_idx + at_m < M) &&
                (a_k + 4 <= K) &&
                ((grad_b_batch * M * K + at_m * K + smem_col_at * 4) % 4 == 0);

            if (can_vectorize_a) {
                reinterpret_cast<float4*>(
                    &smem_AT[at_m * BM + smem_col_at * 4]
                )[0] =
                    reinterpret_cast<const float4*>(
                        &A[at_m * K + smem_col_at * 4]
                    )[0];
            } else {
                for (int v = 0; v < 4; ++v) {
                    int a_k_v = a_k + v;
                    smem_AT[at_m * BM + smem_col_at * 4 + v] =
                        (t_m_idx + at_m < M && a_k_v < K)
                            ? A[at_m * K + smem_col_at * 4 + v]
                            : 0.f;
                }
            }
        }

        // Load in a tile of grad_out into smem (vectorised along N).
        for (int load_offset = 0; load_offset < BK; load_offset += stride_grad_out) {
            int grad_out_row = smem_row_grad_out + load_offset;
            int grad_out_m = t_m_idx + grad_out_row;
            int grad_out_n = grad_b_cols * BN + smem_col_grad_out * 4;
            bool can_vectorize_grad_out =
                (grad_out_m < M) &&
                (grad_out_n + 4 <= N) &&
                ((grad_b_batch * M * N + grad_out_row * N + smem_col_grad_out * 4) % 4 == 0);

            if (can_vectorize_grad_out) {
                reinterpret_cast<float4*>(
                    &smem_grad_out[grad_out_row * BN + smem_col_grad_out * 4]
                )[0] =
                    reinterpret_cast<const float4*>(
                        &grad_out[grad_out_row * N + smem_col_grad_out * 4]
                    )[0];
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_out_n_v = grad_out_n + v;
                    smem_grad_out[grad_out_row * BN + smem_col_grad_out * 4 + v] =
                        (grad_out_m < M && grad_out_n_v < N)
                            ? grad_out[grad_out_row * N + smem_col_grad_out * 4 + v]
                            : 0.f;
                }
            }
        }

        // Ensure all threads are done loading
        __syncthreads();

        // Shift the tile pointers for the next loop.
        A += BK * K; // move down a block
        grad_out += BK * N; // move down a block

        // Use smem values to compute the rolling dot product.
        // smem_AT is M-major so reg_at walks a contiguous row along K.
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                reg_at[i] = smem_AT[dot_idx * BM + thread_row * TM + i];
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

    // Write to grad_b (vectorised along N when the float4 is in-bounds).
    for (int m = 0; m < TM; ++m) {
        int grad_b_row = thread_row * TM + m;
        bool row_ok = (grad_b_batch < batch_size && grad_b_rows * BM + grad_b_row < K);
        for (int n = 0; n < TN; n += 4) {
            int grad_b_col = thread_col * TN + n;
            bool can_vectorize_grad_b =
                row_ok &&
                (grad_b_cols * BN + grad_b_col + 4 <= N) &&
                ((grad_b_batch * K * N + grad_b_row * N + grad_b_col) % 4 == 0);
            if (can_vectorize_grad_b) {
                float4 tmp;
                tmp.x = thread_results[m * TN + n + 0];
                tmp.y = thread_results[m * TN + n + 1];
                tmp.z = thread_results[m * TN + n + 2];
                tmp.w = thread_results[m * TN + n + 3];
                reinterpret_cast<float4*>(&grad_b[grad_b_row * N + grad_b_col])[0] = tmp;
            } else {
                for (int v = 0; v < 4; ++v) {
                    int grad_b_col_v = grad_b_col + v;
                    if (row_ok && grad_b_cols * BN + grad_b_col_v < N) {
                        grad_b[grad_b_row * N + grad_b_col_v] = thread_results[m * TN + n + v];
                    }
                }
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
    const int BK = 16;
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

    // Attention QK is skinny-K (K=head_dim=128) and 256x256 per batch.
    // The FFN 128x128 / 8x8 tile under-occupies that shape; use a smaller
    // MN tile, fewer accumulators, and a larger K-step.
    const int BK = 32;
    const int TM = 4;
    const int TN = 4;
    const int BM = 64;
    const int BN = 64;

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

    // Match the forward batched tile: 64x64 / 4x4 with BK=32.
    const int BK = 32;
    const int TM = 4;
    const int TN = 4;
    const int BM = 64;
    const int BN = 64;

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
