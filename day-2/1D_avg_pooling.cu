#include <cuda_runtime.h>
#include <stdio.h>

/*
    CUDA kernel for 1D Average Pooling
*/
__global__ void avg_pool1d_kernel(
    const float* input,
    int kernel_size,
    int stride,
    int padding,
    float* output,
    int H,
    int H_output
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < H_output) {
        float sum = 0.0f;
        int count = 0;

        for (int j = 0; j < kernel_size; j++) {
            int idx = i * stride + j - padding;
            if (idx >= 0 && idx < H) {
                sum += input[idx];
                count++;
            }
        }

        output[i] = (count > 0) ? sum / count : 0.0f;
    }
}


extern "C" void solution(
    const float* input,
    int kernel_size,
    int stride,
    int padding,
    float* output,
    size_t H
) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;

    size_t size_input  = H * sizeof(float);
    size_t size_output = H_out * sizeof(float);

    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_input, size_input);
    cudaMalloc((void**)&d_output, size_output);

    // Copy input to device
    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (H_out + blockSize - 1) / blockSize;

    avg_pool1d_kernel<<<numBlocks, blockSize>>>(
        d_input,
        kernel_size,
        stride,
        padding,
        d_output,
        (int)H,
        H_out
    );


    // Copy result back to host
    cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
