### Profiling

##### Context: 

We have our base CUDA kernels written. They are numerically correct (we can be certain as we have tested) but they are SLOW. VERY SLOW. So we want to optimise them! 

Having read some awesome books like AI Systems Performance Engineering by Chris Fregly. We know already that in our optimisation loops having evidence is the key thing. We do not optimise purely on what we "think" is happening. We optimise by collecting evidence as that what is actually happening and make decisions on how we can improve! So this is what we are doing here! 

##### Optimisation Process:

Our optimisation process has 3 layers:
    
    Layer 0 — CUDA events vs PyTorch
        Q: How much slower is our kernel than the PyTorch sibling?
        What we will collect: latency (ms), throughput (TFLOPS for GEMM, GB/s for elementwise softmax / RMSNorm), slowdown (naive_ms / torch_ms), spread after warmup.
        What is this layer for: It allows us to reason on what is worth deeper profiling (slowdown × time-in-a-step).
        What this layer does not let us know: Why the kernel is slow; whether we timed the kernel itself or the launch pattern of the kernel?
    
    Layer 1 — Nsight Systems (nsys)
        Q: Is the GPU busy on our kernel, or is it busy on kernel launches / copies / idle?
        What we will collect: kernel time + launch count, GPU utilisation, CUDA API vs kernel time, memcpy/memset, idle gaps.
        What is this layer for: We will pick the slowest kernels from layer 0 to learn more about; detect launch-bound vs kernel-bound.
        What this layer does not let us know: coalescing, cache hit rate, memory vs compute bound, TFLOPS.
    
    Layer 2 — Nsight Compute (ncu) on those kernels
        Q: While this kernel runs, what is the SM bound on / waiting on?
        What we will collect:
            A. SOL: compute, memory, DRAM, L1/L2 throughput (% peak) → memory ≫ compute | compute ≫ memory | both low (latency-bound).
            B. Warp stalls: Long Scoreboard, Barrier, Not Selected, Pipe Busy.
            C. Access: sectors/request, L1/L2 hit rate.
            D. Occupancy only if B says Not Selected or block size looks wrong.
        What is this layer for: This will allow us to generate a specific one-line diagnosis to then use to target optimisations (tile vs coalesce vs fuse vs atomics).

##### Iteration Process

Based on the evidence that we collect form our optimisation process, we will look at form a hypothesis as to what optimisation will be best. We will then (hopefully) write the code for that optimisation and then reprofile!