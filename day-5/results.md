# Results — Naive MatMul Baseline

This document reports the performance of of different matrix multiplication kernels measured on the Tansera platform using an NVIDIA H100.

| Kernel | Optimizations | Platform | GPU | Data type | Achieved GFLOPS |
|---|---:|---|---|---:|---:|
| Naive MatMul | none (no tiling, no shared memory, no loop unrolling) | Tansera | NVIDIA H100 | FP32 | 960.00 GFLOPS |
| Global Mem Coalesce | global memory access coalescing (coalesced loads/stores) | Tansera | NVIDIA H100 | FP32 | 5806.04 GFLOPS |
| Shared Mem  | SMEM access with tiling  | Tansera | NVIDIA H100 | FP32 | 8435.38 GFLOPS |
| 1D Blocktiling  | One thread calculate many values of C (TM in code)  | Tansera | NVIDIA H100 | FP32 | 15821.90 GFLOPS |
| 2D Blocktiling  | One thread calculate 2D values of C (TM*TM in code)  | Tansera | NVIDIA H100 | FP32 | 11409.70 GFLOPS |


