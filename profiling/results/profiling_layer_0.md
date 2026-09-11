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

- Matmul Profile (1D blocktiling):

```python
kernel:   matmul  (1024, 1024) @ (1024, 3072)
CUDA:     1.262 ± 0.064 ms
Torch:    0.419 ± 0.005 ms
slowdown: 3.0x
CUDA:     5.10 TFLOPS
Torch:    15.38 TFLOPS

```

- Matmul Profile (2D blocktiling):

```bash
kernel:   matmul  (1024, 1024) @ (1024, 3072)
CUDA:     0.797 ± 0.006 ms
Torch:    0.421 ± 0.005 ms
slowdown: 1.9x
CUDA:     8.08 TFLOPS
Torch:    15.30 TFLOPS

```

- Matmul Profile (vectorised loads):

```bash
kernel:   matmul  (1024, 1024) @ (1024, 3072)
CUDA:     0.703 ± 0.029 ms
Torch:    0.425 ± 0.018 ms
slowdown: 1.7x
CUDA:     9.16 TFLOPS
Torch:    15.14 TFLOPS

```

- Matmul Profile (vectorised loads, BK=16):

Launch tile changed from `BM=128 BN=128 TM=8 TN=8 BK=8` to `BM=128 BN=128 TM=8 TN=8 BK=16`. Problem shape is unchanged.

```bash
kernel:   matmul (BK @ 16) (1024, 1024) @ (1024, 3072)
CUDA:     0.686 ± 0.070 ms
Torch:    0.432 ± 0.012 ms
slowdown: 1.6x
CUDA:     9.39 TFLOPS
Torch:    14.92 TFLOPS

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

- Batched Matmul Profile (smem):

```bash
kernel:   batch_matmul  (64, 256, 128) @ (64, 128, 256)
CUDA:     0.327 ± 0.006 ms
Torch:    0.130 ± 0.004 ms
slowdown: 2.5x
CUDA:     3.28 TFLOPS
Torch:    8.25 TFLOPS

```

- Batched Matmul Profile (1D blocktiling):

```bash
kernel:   batch_matmul  (64, 256, 128) @ (64, 128, 256)
CUDA:     0.276 ± 0.007 ms
Torch:    0.149 ± 0.006 ms
slowdown: 1.8x
CUDA:     3.90 TFLOPS
Torch:    7.21 TFLOPS

```

- Batched Matmul Profile (2D blocktiling):

```bash
kernel:   batch_matmul  (64, 256, 128) @ (64, 128, 256)
CUDA:     0.269 ± 0.008 ms
Torch:    0.148 ± 0.006 ms
slowdown: 1.8x
CUDA:     3.99 TFLOPS
Torch:    7.26 TFLOPS

```

- Batched Matmul Profile (2D blocktiling, QK tile):

Launch tile changed from `BM=128 BN=128 TM=8 TN=8 BK=8` to `BM=64 BN=64 TM=4 TN=4 BK=32`. Problem shape is unchanged.

```bash
kernel:   batch_matmul  (64, 256, 128) @ (64, 128, 256)
CUDA:     0.160 ± 0.006 ms
Torch:    0.148 ± 0.006 ms
slowdown: 1.1x
CUDA:     6.71 TFLOPS
Torch:    7.23 TFLOPS

```

- Batched Matmul Profile (vectorised loads, QK tile):

```bash
kernel:   batch_matmul  (64, 256, 128) @ (64, 128, 256)
CUDA:     0.159 ± 0.007 ms
Torch:    0.150 ± 0.008 ms
slowdown: 1.1x
CUDA:     6.77 TFLOPS
Torch:    7.17 TFLOPS

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

- Elementwise Add Profile (vectorised loads):

```python
kernel:   addition  (4, 256, 1024) + (4, 256, 1024)
CUDA:     0.085 ± 0.013 ms
Torch:    0.034 ± 0.008 ms
slowdown: 2.5x
CUDA:     147.50 GB/s
Torch:    366.33 GB/s

```

- Elementwise Multi Profile (vectorised loads):

```python
kernel:   multi  (4, 256, 3072) * (4, 256, 3072)
CUDA:     0.075 ± 0.012 ms
Torch:    0.044 ± 0.007 ms
slowdown: 1.7x
CUDA:     506.07 GB/s
Torch:    849.15 GB/s

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

- Softmax Profile (online max and norm):

```python
kernel:   softmax  (64, 256, 256)
CUDA:     0.130 ± 0.010 ms
Torch:    0.044 ± 0.018 ms
slowdown: 2.9x
CUDA:     258.43 GB/s
Torch:    760.99 GB/s

```

- Softmax Profile (shuffle reduce):

```python
kernel:   softmax  (64, 256, 256)
CUDA:     0.102 ± 0.016 ms
Torch:    0.043 ± 0.012 ms
slowdown: 2.4x
CUDA:     329.13 GB/s
Torch:    780.28 GB/s

```

- Softmax Profile (vectorised loads):

```python
kernel:   softmax  (64, 256, 256)
CUDA:     0.069 ± 0.009 ms
Torch:    0.040 ± 0.006 ms
slowdown: 1.7x
CUDA:     483.13 GB/s
Torch:    841.16 GB/s

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