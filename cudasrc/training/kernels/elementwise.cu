/*
Implementation of element-wise addition and multiplication.
Main targets are the residual connections and FFN within QWEN.
*/

/*
Forward pass kernel for element-wise addition.
y[idx] = a[idx] + b[idx]

@param a: Input tensor a.
@param b: Input tensor b.
@param out: Output tensor
@param size: Number of elements in the tensors.
*/
__global__ void fwd_add(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // compute the overall index!
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

/*
Backward pass kernel for element-wise addition.
a[idx] = grad[idx], b[idx] = grad[idx]

@param grad_out: The gradient of the output.
@param grad_a: The gradient for the input tensor a.
@param grad_b: The gradient for the input tensor b.
@param size: The number of elements in the tensors.
*/
__global__ void bwd_add(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] = grad_out[idx];
        grad_b[idx] = grad_out[idx];
    }
}


/*
Forward pass kernel for element-wise multiplication.
y[idx] = a[idx] * b[idx]

@param a: Input tensor a.
@param b: Input tensor b.
@param out: Output tensor.
@param size: Number of elements in the tensors.
*/
__global__ void fwd_multi(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

/*
Backward pass kernel for element-wise multiplication.
Gradients computed as grad_a = grad_out * b, grad_b = grad_out * a

@param grad_out: The gradient from the output.
@param a: The input tensor a.
@param b: The input tensor b.
@param grad_a: The gradient for input a.
@param grad_b: The gradient for input b.
@param size: Number of elements in the tensor.

We are using the product rule here, so d/dx (a*b) = b, d/dy (a *b) = a
*/
__global__ void bwd_multi(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] = grad_out[idx] * b[idx];
        grad_b[idx] = grad_out[idx] * a[idx];
    }
}

/*
Lauch code for forward-pass of element-wise addition.
@param a: Input tensor a.
@param b: Input tensor b.
@param out: The output tensor.
@param size: The number of elements in the tensor.

Called on the CPU to launch the kernel on the GPU.
*/
__host__ void launch_fwd_add(const float* a, const float* b, float* out, int size) {
    int threads_per_block = 256; // We typically launch threads with multiples of 32.
    // ensure we have enough blocks of 256 threads to run the entire op with ceil division.
    int blocks = (size + threads_per_block - 1) / threads_per_block; 
    fwd_add<<<blocks, threads_per_block>>>(a, b, out, size);
}

/*
Launch code for the backward pass of element-wise addition.

@param grad_out: The gradient of the output.
@param grad_a: The gradient for the input tensor a.
@param grad_b: The gradient for the input tensor b.
@param size: The number of elements in the tensors.

Called on the CPU to launch the kernel on the GPU.
*/
__host__ void launch_bwd_add(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    bwd_add<<<blocks, threads_per_block>>>(grad_out, grad_a, grad_b, size);
}

/*
Launch code for the forward pass of element-wise multiplication.

@param: a: Input tensor a.
@param: b: Input tensor b.
@param: out: The output tensor.
@param: size: The number of elements in the tensor.

Called on the CPU to launch the kernel on the GPU.
*/
__host__ void launch_fwd_multi(const float* a, const float* b, float* out, int size) {
    int threads_per_block = 256; 
    int blocks = (size + threads_per_block - 1) / threads_per_block; 
    fwd_multi<<<blocks, threads_per_block>>>(a, b, out, size);
}

/*
Lanches code for the backward pass of element-wise multiplication.

@param grad_out: The gradient from the output.
@param a: The input tensor a.
@param b: The input tensor b.
@param grad_a: The gradient for input a.
@param grad_b: The gradient for input b.
@param size: Number of elements in the tensor.
*/
__host__ void launch_bwd_multi(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block; 
    bwd_multi<<<blocks, threads_per_block>>>(grad_out, a, b, grad_a, grad_b, size);
}