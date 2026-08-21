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
    dim3 threads_per_block(32, 16); 
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
    dim3 threads_per_block_a(32, 16); // Savings on the write, not much we can do currently with the reads.
    dim3 blocks_a(
        (K + threads_per_block_a.x - 1) / threads_per_block_a.x,
        (M + threads_per_block_a.y - 1) / threads_per_block_a.y
    );
    bwd_matmul_a<<<blocks_a, threads_per_block_a>>>(grad_out, B, grad_a, M, N, K);

    // Compute the backward for B.
    // Output is K x N
    dim3 threads_per_block_b(32, 16); // Savings on the reads from grad_out and writes to grad_b
    dim3 blocks_b(
        (N + threads_per_block_b.x - 1) / threads_per_block_b.x,
        (K + threads_per_block_b.y - 1) / threads_per_block_b.y
    );
    bwd_matmul_b<<<blocks_b, threads_per_block_b>>>(grad_out, A, grad_b, M, N, K);
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
