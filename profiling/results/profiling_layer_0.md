### Results from Layer 0 Profiling

Params:
- Warmup: 50 iterations
- Profile Iterations: 250 iterations

---

- Matmul Profile:

```python
kernel:   matmul  (1024, 1024) @ (1024, 3072)
CUDA:     2.189 ± 0.089 ms
Torch:    0.418 ± 0.009 ms
slowdown: 5.2x
CUDA:     2.94 TFLOPS
Torch:    15.42 TFLOPS

```

- Matmul Profile (coalesced):

```python
kernel:   matmul  (1024, 1024) @ (1024, 3072)
CUDA:     2.151 ± 0.138 ms
Torch:    0.420 ± 0.007 ms
slowdown: 5.1x
CUDA:     3.00 TFLOPS
Torch:    15.32 TFLOPS

```

- Matmul Profile (smem):

```python
kernel:   matmul  (1024, 1024) @ (1024, 3072)
CUDA:     1.407 ± 0.129 ms
Torch:    0.419 ± 0.005 ms
slowdown: 3.4x
CUDA:     4.58 TFLOPS
Torch:    15.37 TFLOPS

```

- Batched Matmul Profile:

```bash
kernel:   batch_matmul  (64, 256, 128) @ (64, 128, 256)
CUDA:     0.780 ± 0.007 ms
Torch:    0.126 ± 0.022 ms
slowdown: 6.2x
CUDA:     1.38 TFLOPS
Torch:    8.49 TFLOPS

```

- Batched Matmul Profile (coalesced):

```bash
kernel:   batch_matmul  (64, 256, 128) @ (64, 128, 256)
CUDA:     0.682 ± 0.082 ms
Torch:    0.126 ± 0.006 ms
slowdown: 5.4x
CUDA:     1.57 TFLOPS
Torch:    8.55 TFLOPS

```

- Elementwise Add Profile:

```python
kernel:   addition  (4, 256, 1024) + (4, 256, 1024)
CUDA:     0.085 ± 0.012 ms
Torch:    0.030 ± 0.007 ms
slowdown: 2.8x
CUDA:     148.68 GB/s
Torch:    414.82 GB/s

```

- Elementwise Multi Profile:

```python
kernel:   multi  (4, 256, 3072) * (4, 256, 3072)
CUDA:     0.078 ± 0.015 ms
Torch:    0.045 ± 0.006 ms
slowdown: 1.7x
CUDA:     484.25 GB/s
Torch:    836.92 GB/s

```

- Softmax Profile:

```python
kernel:   softmax  (64, 256, 256)
CUDA:     0.127 ± 0.005 ms
Torch:    0.040 ± 0.004 ms
slowdown: 3.1x
CUDA:     263.70 GB/s
Torch:    830.61 GB/s

```

- RMSNorm Profile:

```python
kernel:   rmsnorm  (4, 256, 1024)
CUDA:     0.063 ± 0.009 ms
Torch:    0.038 ± 0.010 ms
slowdown: 1.6x
CUDA:     132.60 GB/s
Torch:    218.04 GB/s
```