
#include <cuda_runtime.h>

#define BLOCKSIZE 32

__global__ void matmul_kernel_naive(
    const float* A,
    const float* B,
    float* C,
    int m, int n, int k
)
{
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;
    
    //allocate buffers in shared memory
    __shared__ float As[BLOCKSIZE*BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE*BLOCKSIZE];
    
    const int row = threadIdx.x / BLOCKSIZE;
    const int col = threadIdx.x % BLOCKSIZE;
    
    // Calculate global row and column indices
    const int globalRow = cRow * BLOCKSIZE + row;
    const int globalCol = cCol * BLOCKSIZE + col;
    
    float tmp = 0.0;
    
    for(int i = 0; i < k; i += BLOCKSIZE)
    {
        // Load tiles into shared memory with bounds checking
        if (globalRow < m && (i + col) < k) {
            As[row * BLOCKSIZE + col] = A[globalRow * k + i + col];
        } else {
            As[row * BLOCKSIZE + col] = 0.0;
        }
        
        if ((i + row) < k && globalCol < n) {
            Bs[row * BLOCKSIZE + col] = B[(i + row) * n + globalCol];
        } else {
            Bs[row * BLOCKSIZE + col] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int j = 0; j < BLOCKSIZE; ++j) {
            tmp += As[row * BLOCKSIZE + j] * Bs[j * BLOCKSIZE + col];
        }
        
        __syncthreads();
    }
    
    // Write result with bounds checking
    if (globalRow < m && globalCol < n) {
        C[globalRow * n + globalCol] = tmp;
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
    
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(
        (m + BLOCKSIZE - 1) / BLOCKSIZE,
        (n + BLOCKSIZE - 1) / BLOCKSIZE
    );
    
    matmul_kernel_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_c, d_c, size_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}