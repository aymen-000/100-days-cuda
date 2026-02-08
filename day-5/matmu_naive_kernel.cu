#include <cuda_runtime.h>

__global__ void matmul_kernel_naive(
    const float* A,
    const float* B,
    float* C,
    int m, int n, int k
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
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

    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );

    matmul_kernel_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();

    cudaMemcpy(output_c, d_c, size_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
