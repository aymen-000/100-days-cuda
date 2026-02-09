# Results — Naive MatMul Baseline

This document reports the performance of of different matrix multiplication kernels measured on the Tansera platform using an NVIDIA H100.

| Kernel | Optimizations | Platform | GPU | Data type | Achieved GFLOPS |
|---|---:|---|---|---:|---:|
| Naive MatMul | none (no tiling, no shared memory, no loop unrolling) | Tansera | NVIDIA H100 | FP32 | 960.00 GFLOPS |
| Global Mem Coalesce | global memory access coalescing (coalesced loads/stores) | Tansera | NVIDIA H100 | FP32 | 3297.88 GFLOPS |

