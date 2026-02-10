#include <cuda_runtime.h>
#include <cassert>

#define BLOCKSIZE 32
#define BK 8
#define TM 8
#define TN 8

template<int BM, int BN>
__global__ void matmul_kernel_naive(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Original pointers for loading data
    const float* A_block = A + cRow * BM * K;
    const float* B_block = B + cCol * BN;
    float* C_block = C + cRow * BM * N + cCol * BN;

    float threadResults[TM * TN] = {0.0f};

    // register caches for As and Bs
    float regM[TM];
    float regN[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load tile from A into shared memory
        // Each thread loads multiple elements with a stride
        for (int loadIdx = threadIdx.x; loadIdx < BM * BK; loadIdx += blockDim.x) {
            int innerRowA = loadIdx / BK;
            int innerColA = loadIdx % BK;
            
            int globalRowA = cRow * BM + innerRowA;
            int globalColA = bkIdx + innerColA;
            
            if (globalRowA < M && globalColA < K) {
                As[innerRowA * BK + innerColA] = A_block[innerRowA * K + innerColA];
            } else {
                As[innerRowA * BK + innerColA] = 0.0f;
            }
        }
        
        // Load tile from B into shared memory
        for (int loadIdx = threadIdx.x; loadIdx < BK * BN; loadIdx += blockDim.x) {
            int innerRowB = loadIdx / BN;
            int innerColB = loadIdx % BN;
            
            int globalRowB = bkIdx + innerRowB;
            int globalColB = cCol * BN + innerColB;
            
            if (globalRowB < K && globalColB < N) {
                Bs[innerRowB * BN + innerColB] = B_block[innerRowB * N + innerColB];
            } else {
                Bs[innerRowB * BN + innerColB] = 0.0f;
            }
        }

        __syncthreads();
        
        // Move pointers for next tile
        A_block += BK;
        B_block += BK * N;

        // Compute dot products
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // Load values into registers
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            // Compute outer product
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory with bounds checking
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int globalRow = cRow * BM + threadRow * TM + resIdxM;
            int globalCol = cCol * BN + threadCol * TN + resIdxN;
            
            if (globalRow < M && globalCol < N) {
                C_block[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                    threadResults[resIdxM * TN + resIdxN];
            }
        }
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
    
    if (m >= 128 && n >= 128) {
        const int BM = 128;
        const int BN = 128;
        
        dim3 block((BM * BN) / (TM * TN));
        dim3 grid(
            (n + BN - 1) / BN,
            (m + BM - 1) / BM
        );
        
        matmul_kernel_naive<BM, BN><<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    } else {
        const int BM = 64;
        const int BN = 64;
        
        dim3 block((BM * BN) / (TM * TN));
        dim3 grid(
            (n + BN - 1) / BN,
            (m + BM - 1) / BM
        );
        
        matmul_kernel_naive<BM, BN><<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_c, d_c, size_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}