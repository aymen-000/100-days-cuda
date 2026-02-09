#include <cuda_runtime.h>
#define BLOCKSIZE 32

__global__ void matmul_kernel_global_mem_coalesce(
    const float* A,
    const float* B,
    float* C,
    int m, int n, int k
) {
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < m && y < n) {
    float tmp = 0.0;
    for (int i = 0; i < k; ++i) {
        tmp += A[x * k + i] * B[i * n + y];
    }
    C[x * n + y] = tmp ;}

}
extern "C" void solution(
    const float* input_a,
    const float* input_b,
    float* output_c,
    int m, int n, int k
) {
    float *d_a, *d_b, *d_c;

    size_t size_a = m * k * sizeof(float);
    size_t size_b = k * n * sizeof(float);
    size_t size_c = m * n * sizeof(float);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, input_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, input_b, size_b, cudaMemcpyHostToDevice);

    dim3 block(BLOCKSIZE*BLOCKSIZE);
    dim3 grid(
        (m + BLOCKSIZE- 1) / BLOCKSIZE,
        (n + BLOCKSIZE - 1) / BLOCKSIZE
    );

    matmul_kernel_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();

    cudaMemcpy(output_c, d_c, size_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
