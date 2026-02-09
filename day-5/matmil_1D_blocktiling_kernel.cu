#include <cuda_runtime.h>
#include <cassert>

#define BLOCKSIZE 32
#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void matmul_kernel_naive(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();
        
        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float tmpB = Bs[dotIdx * BN + threadCol];

            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}

extern "C" void solution(
    const float* input_a,
    const float* input_b,
    float* output_c,
    int m, int n, int k
)
{
    float *d_a, *d_b, *d_c;
    
    size_t size_a = m * k * sizeof(float);
    size_t size_b = k * n * sizeof(float);
    size_t size_c = m * n * sizeof(float);
    
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    
    cudaMemcpy(d_a, input_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, input_b, size_b, cudaMemcpyHostToDevice);
    
    dim3 block((BM * BN) / TM);
    dim3 grid(
        (n + BN - 1) / BN,
        (m + BM - 1) / BM
    );
    
    matmul_kernel_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_c, d_c, size_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}