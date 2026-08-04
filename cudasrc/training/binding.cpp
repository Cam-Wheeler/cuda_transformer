/*
Code for calling the CUDA code from C++ Pytorch and binding to Python.

- Element wise operations (add and multiply) (V1).
- Activation functions (SiLU) (V1).

The functions do their own input validation and convert the tensors
into pointers so we can pass them over to CUDA.
*/

#include "c10/core/Device.h"
#include "pybind11/detail/common.h"
#include <torch/extension.h>

// Declarations for the CUDA launch functions.
void launch_fwd_add(const float* a, const float* b, float* out, int size);
void launch_bwd_add(const float* grad_out, float* grad_a, float* grad_b, int size);
void launch_fwd_multi(const float* a, const float* b, float* out, int size);
void launch_bwd_multi(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size);
void launch_silu_fwd(const float* x, float* out, int size);
void launch_silu_bwd(const float* grad_out, const float* x, float* grad_in, int size);

// Pytorch C++ binding to the CUDA code.

/*
Element-wise addition forward pass.
*/
void fwd_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    // Contiguous
    a = a.contiguous();
    b = b.contiguous();
    out = out.contiguous();

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
    TORCH_CHECK(grad_out.numel() == grad_a.numel() && grad_out.numel() == grad_b.numel(), "tensor sizes must match");

    // Contiguous
    grad_out = grad_out.contiguous();
    grad_a = grad_a.contiguous();
    grad_b = grad_b.contiguous();

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
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    // Contiguous
    a = a.contiguous();
    b = b.contiguous();
    out = out.contiguous();

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
    TORCH_CHECK(grad_out.numel() == a.numel() && grad_out.numel() == b.numel() && 
                grad_out.numel() == grad_a.numel() && grad_out.numel() == grad_b.numel(),
                "tensor sizes must match");

    // Contiguous
    grad_out = grad_out.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    grad_a = grad_a.contiguous();
    grad_b = grad_b.contiguous();

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
    TORCH_CHECK(out.device().is_cuda(), "out for SiLU must be a CUDA tensor.")
    TORCH_CHECK(x.numel() == out.numel(), "tensor sizes to SilU must match.");

    // Contiguous
    x = x.contiguous();
    out = out.contiguous();

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
    TORCH_CHECK(grad_out.numel() == x.numel() && grad_out.numel() == grad_in.numel(),
                "tensor sizes to SilU must match.");

    // Contiguous
    grad_out = grad_out.contiguous();
    x = x.contiguous();
    grad_in = grad_in.contiguous();

    // Launch the kernel
    launch_silu_bwd(grad_out.data_ptr<float>(),
                    x.data_ptr<float>(),
                    grad_in.data_ptr<float>(),
                    grad_out.numel());
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
}