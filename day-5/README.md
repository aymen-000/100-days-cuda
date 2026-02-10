# Results — Matrix Multiplication Benchmarks

Performance measurements for several matrix multiplication kernels, collected on the Tansera platform using an NVIDIA H100 (FP32).

| Kernel               | Optimizations / Notes                                                      | Platform | GPU        | Data type | Achieved GFLOPS |
|---                    |---                                                                          |---       |---         |---        |---              |
| Naive MatMul         | No tiling, no shared memory, no loop unrolling                              | Tansera  | NVIDIA H100| FP32      | 960.00 GFLOPS   |
| Global Mem Coalesce  | Coalesced global memory loads/stores                                        | Tansera  | NVIDIA H100| FP32      | 5806.04 GFLOPS  |
| Shared Mem           | Shared-memory tiling (SMEM)                                                  | Tansera  | NVIDIA H100| FP32      | 8435.38 GFLOPS  |
| 1D Block Tiling      | Each thread computes multiple C values along one dimension (TM in code)     | Tansera  | NVIDIA H100| FP32      | 15821.90 GFLOPS |
| 2D Block Tiling      | Each thread computes a 2D tile of C (TM × TM in code)                       | Tansera  | NVIDIA H100| FP32      | 11409.70 GFLOPS |
| Kernel Vectorize     | Vectorized memory accesses (load rows/columns as vectors)                   | Tansera  | NVIDIA H100| FP32      | 20738.77 GFLOPS |

Notes
- Results are reported in GFLOPS (giga floating-point operations per second).
- Measurements were collected on the Tansera platform using an NVIDIA H100; matrix sizes, compiler flags, and run details should be recorded alongside each measurement for reproducibility.

Credit
- Adapted from: https://siboehm.com/articles/22/CUDA-MMM

