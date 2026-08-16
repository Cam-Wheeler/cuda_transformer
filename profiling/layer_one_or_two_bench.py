"""
Layers 1 and 2 of the profiling framework.

Layer 1 (nsys) and layer 2 (ncu) share this harness: warmup, then a
profiler-API capture around the CUDA kernel launches.
"""

import torch
from argparse import ArgumentParser
from setup_kernels import KERNELS

def benchmark_kernel(kernel, warmup, iterations, nvtx_name):
    """Warm up, then capture kernel launches for nsys / ncu."""
    for _ in range(warmup):
        kernel()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range(nvtx_name):
        for _ in range(iterations):
            kernel()
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()

def main():
    parser = ArgumentParser()
    parser.add_argument("--kernel", choices=KERNELS.keys(), required=True)
    parser.add_argument("--layer", choices=["1", "2"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Layer 1 / Layer 2 profiling.")

    bench = KERNELS[args.kernel]()
    if args.layer == "1":
        warmup = 5
        iterations = 25
    else:
        warmup = 5
        iterations = 5  # ncu replays each launch; keep this small.

    with torch.no_grad():
        benchmark_kernel(bench["cuda"], warmup, iterations, f"layer{args.layer}")

if __name__ == "__main__":
    main()
