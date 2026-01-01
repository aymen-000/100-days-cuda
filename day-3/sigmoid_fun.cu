#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

__global__ void sigmoid_kernel(const float* input , float* output , size_t n , size_t m) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < n && col < m) { // avoid out-of-bounds
        output[row * m + col] = 1.0f / (1.0f + expf(-input[row * m + col]));
    }

}
// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    // float *d_input, *d_output;
    // size_t size = n * m * sizeof(float);

    //cudaMalloc(&d_input, size);
    // cudaMalloc(&d_output, size);

    // cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    sigmoid_kernel<<<grid, block>>>(input, output, n, m);
    cudaDeviceSynchronize();

    //cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    //cudaFree(d_input);
    //cudaFree(d_output);
}
