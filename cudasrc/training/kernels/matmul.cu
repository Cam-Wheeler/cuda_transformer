/*
CUDA code for matrix multiply.
Within QWEN we will be using this in the attention and FFN layers.
*/

#include <cuda_runtime.h>

/*
Standard matmul forward pass.

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
__global__ void fwd_matmul(const float* A, const float* B, float* C, int M, int N, int K) {

    // Grab the row and col this thread is responsible for! 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check.
    if (row < M && col < N) {
        float sum = 0.f;
        // Iterate across the row and down the column.
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
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
__global__ void bwd_matmul_a(const float* grad_out, const float* B, float* grad_a, int M, int N, int K) {

    // Get the row and col
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row idx for grad_a
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col idx for grad_a

    // Bounds checking
    if (row < M && col < K) {
        float sum = 0.f;
        // We iterate through n as N is the shared column dim between grad_out and B.
        for (int n = 0; n < N; n++) {
            // cols here are not really "columns" its just the row that we want! 
            sum += grad_out[row * N + n] * B[col * N + n];
        }
        // Add the sum to the grad_A matrix.
        grad_a[row * K + col] = sum;
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
__global__ void bwd_matmul_b(const float* grad_out, const float* A, float* grad_b, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; // row idx for grad_b
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col idx for grad_b

    if (row < K && col < N) {
        float sum = 0.f;
        // Because that we have M x N and M x K we will end up with K x N
        for (int m = 0; m < M; m++) {
            sum += A[m * K + row] * grad_out[m * N + col];
        } 
        grad_b[row * N + col] = sum;
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
    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );
    fwd_matmul<<<blocks, threads_per_block>>>(A, B, C, M, N, K);
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
    dim3 threads_per_block_a(16, 16);
    dim3 blocks_a(
        (K + threads_per_block_a.x - 1) / threads_per_block_a.x,
        (M + threads_per_block_a.y - 1) / threads_per_block_a.y
    );
    bwd_matmul_a<<<blocks_a, threads_per_block_a>>>(grad_out, B, grad_a, M, N, K);

    // Compute the backward for B.
    // Output is K x N
    dim3 threads_per_block_b(16, 16);
    dim3 blocks_b(
        (N + threads_per_block_b.x - 1) / threads_per_block_b.x,
        (K + threads_per_block_b.y - 1) / threads_per_block_b.y
    );
    bwd_matmul_b<<<blocks_b, threads_per_block_b>>>(grad_out, A, grad_b, M, N, K);
}
