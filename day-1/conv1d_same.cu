#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* A , const float* B , float* C , int N , int M ) {
    int i = threadIdx.x + blockIdx.x*blockDim.x; 

    if (i <N) {
        float sum = 0.0f ; 
        int half = M/2 ; 
        for (size_t j =0 ; j <M , j++) {
            int index = i+j-half ; 
            if (index >= 0 && index < N) {
                sum += A[index] * B[j] ; 
            } 
            // else we have 0 since we are using zero-padding 
        }
        C[i] = sum ;
    }
}

extern "C" void solution(const float* A, const* float B , float* C , int N , int M) {
    float *da , *db , *dc ; 
    size_t size_A , size_B , size_C  = N* sizeof(float) ;
    //allocate  memoary on device
    cudaMalloc((void**)&da , size_A) ; 
    cudaMalloc((void**)&db , size_B) ;
    cudaMalloc((void**)&dc , size_C) ;
    // copy data from host to device
    cudaMemcpy(da , A , size_A , cudaMemcpyHostToDevice) ;
    cudaMemcpy(db , B , size_B , cudaMemcpyHostToDevice) ;
    // launch kernel
    int blockSize = 256 ;
    int numBlocks = (N + blockSize - 1) / blockSize ;
    conv1d_kernel<<<numBlocks , blockSize>>>(da , db , dc , N , M) ;
    // copy result from device to host
    cudaMemcpy(C , dc , size_C , cudaMemcpyDeviceToHost) ;
    // free device memory
    cudaFree(da) ; 
    cudaFree(db) ;
    cudaFree(dc) ;  
}