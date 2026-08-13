/*
Code for calling the CUDA code from C++ Pytorch and binding to Python.

- Element wise operations (add and multiply) (V1).
- Activation functions (SiLU) (V1).
- Matrix multi (single and batched) (V1).
- RMSNorm (V1)
- Softmax (V1)

The functions do their own input validation and convert the tensors
into pointers so we can pass them over to CUDA.
*/

#include "c10/core/Device.h"
#include "pybind11/detail/common.h"
#include <torch/extension.h>

// Prototypes for the CUDA launch functions.
void launch_fwd_add(const float* a, const float* b, float* out, int size);
void launch_bwd_add(const float* grad_out, float* grad_a, float* grad_b, int size);
void launch_fwd_multi(const float* a, const float* b, float* out, int size);
void launch_bwd_multi(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size);
void launch_silu_fwd(const float* x, float* out, int size);
void launch_silu_bwd(const float* grad_out, const float* x, float* grad_in, int size);
void launch_fwd_matmul(const float* A, const float* B, float* C, int M, int N, int K);
void launch_bwd_matmul(const float* grad_out, const float* A, const float* B, float* grad_a, float* grad_b, int M, int N, int K);
void launch_fwd_batched_matmul(const float* A, const float* B, float* C, int batch_size, int M, int N, int K);
void launch_bwd_batched_matmul(const float* grad_out, const float* A, const float* B, float* grad_a, float* grad_b, int batch_size, int M, int N, int K);
void launch_fwd_rmsnorm(const float* x, const float* gamma, float* out, float* inv_rms_out, int batch_size, int seq_len, int n_embed, float eps);
void launch_bwd_rmsnorm(const float* grad_out, const float* x, const float* gamma, const float* inv_rms, float* grad_x, float* grad_gamma, int batch_size, int seq_len, int n_embed);
void launch_fwd_softmax(const float* x, float* out, int batch_size, int seq_len, int n_embed);
void launch_bwd_softmax(const float* grad_out, const float* output_probs, float* grad_x, int batch_size, int seq_len, int n_embed);

// Pytorch C++ binding to the CUDA code.

/*
Element-wise addition forward pass.
*/
void fwd_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    // Contiguous
    a = a.contiguous();
    b = b.contiguous();

    // CUDA kernel launch!
    launch_fwd_add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
}

/*
Element-wise addition backward pass.
*/
void bwd_add(torch::Tensor grad_out, torch::Tensor grad_a, torch::Tensor grad_b) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(grad_a.is_contiguous(), "grad_a must be contiguous");
    TORCH_CHECK(grad_b.is_contiguous(), "grad_b must be contiguous");
    TORCH_CHECK(grad_out.numel() == grad_a.numel() && grad_out.numel() == grad_b.numel(), "tensor sizes must match");

    // Contiguous for the reads. Writes should be contiguous already.
    grad_out = grad_out.contiguous();

    // CUDA kernel launch!
    launch_bwd_add(grad_out.data_ptr<float>(),
            grad_a.data_ptr<float>(),
            grad_b.data_ptr<float>(),
            grad_out.numel());
}

/*
Element-wise multiplication forward pass.
*/
void fwd_multi(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    // Contiguous
    a = a.contiguous();
    b = b.contiguous();

    // CUDA kernel launch!
    launch_fwd_multi(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
}

/*
Element-wise multiplication backward pass.
*/
void bwd_multi(torch::Tensor grad_out, torch::Tensor a, torch::Tensor b, torch::Tensor grad_a, torch::Tensor grad_b) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(grad_a.is_contiguous(), "grad_a must be contiguous");
    TORCH_CHECK(grad_b.is_contiguous(), "grad_b must be contiguous");
    TORCH_CHECK(grad_out.numel() == a.numel() && grad_out.numel() == b.numel() && 
                grad_out.numel() == grad_a.numel() && grad_out.numel() == grad_b.numel(),
                "tensor sizes must match");

    // Contiguous for the reads. Writes should be contiguous already.
    grad_out = grad_out.contiguous();
    a = a.contiguous();
    b = b.contiguous();

    // CUDA kernel launch!
    launch_bwd_multi(grad_out.data_ptr<float>(),
                     a.data_ptr<float>(),
                     b.data_ptr<float>(),
                     grad_a.data_ptr<float>(),
                     grad_b.data_ptr<float>(),
                     grad_out.numel());
}

/*
SiLU Forward pass.
*/
void fwd_silu(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.device().is_cuda(), "input to SiLU must be a CUDA tensor.");
    TORCH_CHECK(out.device().is_cuda(), "out for SiLU must be a CUDA tensor.");
    TORCH_CHECK(out.is_contiguous(), "out for SiLU must be passed in as contiguous.");
    TORCH_CHECK(x.numel() == out.numel(), "tensor sizes to SilU must match.");

    // Contiguous
    x = x.contiguous();

    // Launch the kernel
    launch_silu_fwd(x.data_ptr<float>(), out.data_ptr<float>() , x.numel());
}

/*
SiLU Backward pass.
*/
void bwd_silu(torch::Tensor grad_out, torch::Tensor x, torch::Tensor grad_in) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(grad_in.device().is_cuda(), "grad_in must be a CUDA tensor");
    TORCH_CHECK(grad_in.is_contiguous(), "grad_in must be contiguous");
    TORCH_CHECK(grad_out.numel() == x.numel() && grad_out.numel() == grad_in.numel(),
                "tensor sizes to SilU must match.");

    // Contiguous
    grad_out = grad_out.contiguous();
    x = x.contiguous();

    // Launch the kernel
    launch_silu_bwd(grad_out.data_ptr<float>(),
                    x.data_ptr<float>(),
                    grad_in.data_ptr<float>(),
                    grad_out.numel());
}

/*
Matmul forward pass.
*/
void fwd_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "A, B, and C must be 2D tensors");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0) && A.size(0) == C.size(0) && B.size(1) == C.size(1), 
                "tensor sizes must match");

    // Contiguous for the reads. Writes should be contiguous already.
    A = A.contiguous();
    B = B.contiguous();

    // Sizes for the kernel launch.
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    // CUDA kernel launch!
    launch_fwd_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

/*
Matmul backward pass.
*/
void bwd_matmul(
    torch::Tensor grad_out, torch::Tensor A, torch::Tensor B, 
    torch::Tensor grad_a, torch::Tensor grad_b
) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(grad_a.is_contiguous(), "grad_a must be contiguous");
    TORCH_CHECK(grad_b.is_contiguous(), "grad_b must be contiguous");
    TORCH_CHECK(grad_out.dim() == 2 && A.dim() == 2 && B.dim() == 2 &&
                grad_a.dim() == 2 && grad_b.dim() == 2,
                "grad_out, A, B, grad_a, grad_b must be 2D");
    TORCH_CHECK(grad_out.size(0) == A.size(0) && grad_out.size(0) == grad_a.size(0),
                "M mismatch");
    TORCH_CHECK(grad_out.size(1) == B.size(1) && grad_out.size(1) == grad_b.size(1),
                "N mismatch");
    TORCH_CHECK(A.size(1) == B.size(0) && A.size(1) == grad_a.size(1) &&
                B.size(0) == grad_b.size(0),
                "K mismatch");

    // Contiguous for the reads. Writes should be contiguous already.
    grad_out = grad_out.contiguous();
    A = A.contiguous();
    B = B.contiguous();

    // Sizes for the kernel launch.
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    // CUDA kernel launch!
    launch_bwd_matmul(
        grad_out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
        grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), M, N, K
    );
}

/*
Batched matmul forward pass.
*/
void fwd_batched_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && C.dim() == 3, "A, B, and C must be 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == C.size(0), "batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "A's K must match B's K");
    TORCH_CHECK(A.size(1) == C.size(1) && B.size(2) == C.size(2), "output shape must be (batch, M, N)");

    // Contiguous for the reads. Writes should be contiguous already.
    A = A.contiguous();
    B = B.contiguous();

    // Sizes for the kernel launch.
    int batch_size = A.size(0);
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);

    // CUDA kernel launch!
    launch_fwd_batched_matmul(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, M, N, K
    );
}

/*
Batched matmul backward pass.
*/
void bwd_batched_matmul(torch::Tensor grad_out, torch::Tensor A, torch::Tensor B,
    torch::Tensor grad_a, torch::Tensor grad_b
) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(grad_a.is_contiguous(), "grad_a must be contiguous");
    TORCH_CHECK(grad_b.is_contiguous(), "grad_b must be contiguous");
    TORCH_CHECK(grad_out.dim() == 3 && A.dim() == 3 && B.dim() == 3 &&
                grad_a.dim() == 3 && grad_b.dim() == 3,
                "grad_out, A, B, grad_a, grad_b must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == grad_out.size(0) &&
                A.size(0) == grad_a.size(0) && A.size(0) == grad_b.size(0),
                "batch sizes must match");
    TORCH_CHECK(grad_out.size(1) == A.size(1) && grad_out.size(1) == grad_a.size(1),
                "M mismatch");
    TORCH_CHECK(grad_out.size(2) == B.size(2) && grad_out.size(2) == grad_b.size(2),
                "N mismatch");
    TORCH_CHECK(A.size(2) == B.size(1) && A.size(2) == grad_a.size(2) &&
                B.size(1) == grad_b.size(1),
                "K mismatch");

    // Contiguous for the reads. Writes should be contiguous already.
    grad_out = grad_out.contiguous();
    A = A.contiguous();
    B = B.contiguous();

    // Sizes for the kernel launch.
    int batch_size = A.size(0);
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);

    // CUDA kernel launch!
    launch_bwd_batched_matmul(
        grad_out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
        grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), batch_size, M, N, K
    );
}

/*
RMSNorm forward pass.
*/
void fwd_rmsnorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor out, torch::Tensor inv_rms_out, float eps) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(inv_rms_out.device().is_cuda(), "inv_rms_out must be a CUDA tensor");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(inv_rms_out.is_contiguous(), "inv_rms_out must be contiguous");
    TORCH_CHECK(x.dim() == 3 && gamma.dim() == 1 && out.dim() == 3 && inv_rms_out.dim() == 2,
                "x must be 3D, gamma must be 1D, out must be 3D, inv_rms_out must be 2D");
    TORCH_CHECK(gamma.size(0) == x.size(2), "gamma must match the embedding dimension of x");
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == x.size(1) && out.size(2) == x.size(2),
                "out must match the shape of x");
    TORCH_CHECK(inv_rms_out.size(0) == x.size(0) && inv_rms_out.size(1) == x.size(1),
                "inv_rms_out must match the shape of batch and sequence length of x");
    TORCH_CHECK(eps > 0, "eps must be greater than 0");

    // contiguous for the reads, writes should already be cont
    x = x.contiguous();
    gamma = gamma.contiguous();

    // Sizes
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embed = x.size(2);

    launch_fwd_rmsnorm(
        x.data_ptr<float>(), gamma.data_ptr<float>(), out.data_ptr<float>(),
        inv_rms_out.data_ptr<float>(), batch_size, seq_len, n_embed, eps);
}

/*
RMSNorm backward pass.
*/
void bwd_rmsnorm(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor gamma,
    torch::Tensor inv_rms, torch::Tensor grad_x, torch::Tensor grad_gamma
) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(inv_rms.device().is_cuda(), "inv_rms must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_gamma.device().is_cuda(), "grad_gamma must be a CUDA tensor");
    TORCH_CHECK(grad_x.is_contiguous(), "grad_x must be contiguous");
    TORCH_CHECK(grad_gamma.is_contiguous(), "grad_gamma must be contiguous");
    TORCH_CHECK(grad_out.dim() == 3 && x.dim() == 3 && gamma.dim() == 1 && 
                inv_rms.dim() == 2 && grad_x.dim() == 3 && grad_gamma.dim() == 1,
                "grad_out, x, grad_x must be 3D, gamma, grad_gamma must be 1D, inv_rms must be 2D");
    TORCH_CHECK(grad_out.size(0) == x.size(0) && grad_out.size(1) == x.size(1) && grad_out.size(2) == x.size(2),
                "grad_out must match the shape of x");
    TORCH_CHECK(x.size(0) == grad_x.size(0) && x.size(1) == grad_x.size(1) && x.size(2) == grad_x.size(2),
                "grad_x must match the shape of x");
    TORCH_CHECK(gamma.size(0) == grad_gamma.size(0),
                "grad_gamma must match the shape of gamma");
    TORCH_CHECK(inv_rms.size(0) == x.size(0) && inv_rms.size(1) == x.size(1),
                "inv_rms must match the shape of x in batch and sequence length");
    TORCH_CHECK(gamma.size(0) == x.size(2), "gamma must match the embedding dimension of x");

    // contiguous for reads.
    grad_out = grad_out.contiguous();
    x = x.contiguous();
    gamma = gamma.contiguous();
    inv_rms = inv_rms.contiguous();

    // Sizes
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embed = x.size(2);

    launch_bwd_rmsnorm(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), gamma.data_ptr<float>(),
        inv_rms.data_ptr<float>(), grad_x.data_ptr<float>(), grad_gamma.data_ptr<float>(),
        batch_size, seq_len, n_embed
    );
}

/*
Softmax forward pass.
*/
void fwd_softmax(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(x.dim() == 3 && out.dim() == 3, "x and out must be 3D tensors");
    TORCH_CHECK(x.size(0) == out.size(0) && x.size(1) == out.size(1) && x.size(2) == out.size(2),
                "x and out must have the same shape");

    // Contig for reads, writes will be contig already.
    x = x.contiguous();

    // Sizes
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embed = x.size(2);

    launch_fwd_softmax(
        x.data_ptr<float>(), out.data_ptr<float>(), batch_size, seq_len, n_embed
    );

}

/*
Softmax backward pass.
*/
void bwd_softmax(torch::Tensor grad_out, torch::Tensor output_probs, torch::Tensor grad_x) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(output_probs.device().is_cuda(), "output_probs must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_x.is_contiguous(), "grad_x must be contiguous");
    TORCH_CHECK(grad_out.dim() == 3 && output_probs.dim() == 3 && grad_x.dim() == 3,
                "grad_out, output_probs, grad_x must be 3D tensors");
    TORCH_CHECK(grad_out.size(0) == output_probs.size(0) && grad_out.size(1) == output_probs.size(1) && grad_out.size(2) == output_probs.size(2),
                "grad_out must match the shape of output_probs");
    TORCH_CHECK(grad_x.size(0) == grad_out.size(0) && grad_x.size(1) == grad_out.size(1) && grad_x.size(2) == grad_out.size(2),
                "grad_x must match the shape of grad_out");

    // Contig for reads, writes contig already.
    grad_out = grad_out.contiguous();
    output_probs = output_probs.contiguous();

    // Sizes
    int batch_size = grad_out.size(0);
    int seq_len = grad_out.size(1);
    int n_embed = grad_out.size(2);

    launch_bwd_softmax(
        grad_out.data_ptr<float>(), output_probs.data_ptr<float>(),
        grad_x.data_ptr<float>(), batch_size, seq_len, n_embed
    );
}


/*
Now we bind to python!

Very roughly:
1. The preprocessor expands the PYBIND11_MODULE macro.
2. Compiled down into machine code by the compiler generating the .so file 
   full of inits (and the rest of our code).
3. We import the .so file in Python. Python callables are registered by the inits.
   The callables are then hooked to the wrappers around our C++ functions that Pybind generates.
4. Wrappers handle the conversion from calling C++ and returning results.
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd_add", &fwd_add, "Element-wise addition forward");
    m.def("bwd_add", &bwd_add, "Element-wise addition backward");
    m.def("fwd_multi", &fwd_multi, "Element-wise multiplication forward");
    m.def("bwd_multi", &bwd_multi, "Element-wise multiplication backward");
    m.def("fwd_silu", &fwd_silu, "SiLU forward");
    m.def("bwd_silu", &bwd_silu, "SiLU backward");
    m.def("fwd_matmul", &fwd_matmul, "Matmul forward");
    m.def("bwd_matmul", &bwd_matmul, "Matmul backward");
    m.def("fwd_batched_matmul", &fwd_batched_matmul, "Batched matmul forward");
    m.def("bwd_batched_matmul", &bwd_batched_matmul, "Batched matmul backward");
    m.def("fwd_rmsnorm", &fwd_rmsnorm, "RMSNorm forward");
    m.def("bwd_rmsnorm", &bwd_rmsnorm, "RMSNorm backward");
    m.def("fwd_softmax", &fwd_softmax, "Softmax forward");
    m.def("bwd_softmax", &bwd_softmax, "Softmax backward");
}