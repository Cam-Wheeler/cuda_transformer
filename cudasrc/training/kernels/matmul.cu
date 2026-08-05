/*
CUDA code for matrix multiply.
Within QWEN we will be using this in the attention and FFN layers.
*/


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
Standard matul backward pass to compute the grad with respect to matrix A.
*/
__global__ void bwd_matmul_a() {

}

/*
Standard matmul backward pass to compute the grad with respect to matrix B.
*/
__global__ void bwd_matmul_b() {

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
    dim3 threads_per_block = (16, 16)
    dim3 blocks = (N + threads_per_block.x - 1) / threads_per_block., (M + threads_per_block.y - 1) / threads_per_block.y;
    fwd_matmul<<<blocks, threads_per_block>>>(A, B, C, M, N, K);
}