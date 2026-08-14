"""
Layer 0 of the profiling framework.

Compare our CUDA kernels against their PyTorch siblings using CUDA events.
Reports latency (ms), throughput (TFLOPS or GB/s), and slowdown.
"""
import torch
from argparse import ArgumentParser
from setup_kernels import KERNELS

def benchmark_kernel(kernel, warmup=50, iterations=250):
    """
    Time a callable with CUDA events.

    Returns mean latency in ms and the per-iteration samples.
    """
    # Warm the kernel up.
    for _ in range(warmup):
        kernel()

    # Now we actually profile.
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        kernel()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Stats
    mean_ms = sum(times) / len(times)
    return mean_ms, times

def _std(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5

def _throughput(mean_ms, flops, nbytes):
    if flops is not None:
        return flops / mean_ms / 1e9, "TFLOPS"
    if nbytes is not None:
        return nbytes / mean_ms / 1e6, "GB/s"
    return None, None

def main():
    parser = ArgumentParser()
    parser.add_argument("--kernel", choices=KERNELS.keys(), required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Layer 0 timing.")

    bench = KERNELS[args.kernel]()

    with torch.no_grad():
        cuda_ms, cuda_times = benchmark_kernel(bench["cuda"])
        torch_ms, torch_times = benchmark_kernel(bench["torch"])

    cuda_tp, unit = _throughput(cuda_ms, bench["flops"], bench["bytes"])
    torch_tp, _ = _throughput(torch_ms, bench["flops"], bench["bytes"])

    print(f"kernel:   {args.kernel}  {bench['shape']}")
    print(f"CUDA:     {cuda_ms:.3f} ± {_std(cuda_times):.3f} ms")
    print(f"Torch:    {torch_ms:.3f} ± {_std(torch_times):.3f} ms")
    print(f"slowdown: {cuda_ms / torch_ms:.1f}x")
    if cuda_tp is not None:
        print(f"CUDA:     {cuda_tp:.2f} {unit}")
        print(f"Torch:    {torch_tp:.2f} {unit}")


if __name__ == "__main__":
    main()
