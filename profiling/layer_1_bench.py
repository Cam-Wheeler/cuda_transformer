"""
Layer 1 of the profiling framework.

Compare our CUDA kernels against their PyTorch siblings using CUDA events.
"""

import torch
from argparse import ArgumentParser
from setup_kernels import KERNELS

def benchmark_kernel(kernel, warmup=5, iterations=25): # change from layer 0.
    """
    Kernel for nsys profiling.
    """
    # warmup
    for _ in range(warmup):
        kernel()
    torch.cuda.synchronize()

    # Now lets profile.
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range("layer1"):    
        for _ in range(iterations):
            kernel()
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()

def main():
    parser = ArgumentParser()
    parser.add_argument("--kernel", choices=KERNELS.keys(), required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Layer 1 profiling.")


    bench = KERNELS[args.kernel]()

    with torch.no_grad():
        benchmark_kernel(bench["cuda"])

if __name__ == "__main__":
    main()