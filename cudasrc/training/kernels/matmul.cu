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
__global__ void fwd_matmul() {

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