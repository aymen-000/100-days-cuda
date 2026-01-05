#include <stdio.h>
#include <cuda_runtime.h>

/** HYPERPARAMETERS **/
#define RESX 1024
#define RESY 1024
int resolutionX = RESX;
int resolutionY = RESY;
double rectangleWidth = 0.00000005;
double rectangleCenterX = -1.86216646;
double rectangleCenterY = 0.0;
int iterationsMax = 10000;

/* ERROR CHECKING MACRO */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* SAVE IMAGE */
void savePPM(const char *filename, int *matrix)
{
    FILE *image = fopen(filename, "w");
    if (!image) {
        perror("Error opening file");
        return;
    }
    fprintf(image, "P3\n%d %d\n255\n", resolutionX, resolutionY);
    
    for (int y = 0; y < resolutionY; y++) {
        for (int x = 0; x < resolutionX; x++) {
            int val = matrix[y * resolutionX + x];
            if (val == 0) {
                fprintf(image, "0 0 0\n"); /
            } else {
                int r = (val * 5) % 256;
                int g = (val * 7) % 256;
                int b = (val * 11) % 256;
                fprintf(image, "%d %d %d\n", r, g, b);
            }
        }
    }
    fclose(image);
}

/* MANDELBROT TEST */
__device__ int isInSet(double x, double y, int iterations)
{
    double x0 = 0.0, y0 = 0.0;
    int counter = 0;
    
    for (int i = 0; i < iterations; i++) {
        double x_sq = x0 * x0;
        double y_sq = y0 * y0;
        
        if (x_sq + y_sq >= 4.0) {
            return counter;
        }
        
        double currentY = 2.0 * x0 * y0 + y;
        double currentX = x_sq - y_sq + x;
        
        x0 = currentX;
        y0 = currentY;
        counter++;
    }
    
    return 0; // Point is in the set
}

/* CUDA KERNEL */
__global__ void mandelbrotKernel(int *result, double cxStart, double cyStart,
                                  double step, int width, int height, int maxIter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    double cx = cxStart + x * step;
    double cy = cyStart - y * step;
    
    result[y * width + x] = isInSet(cx, cy, maxIter);
}

int main()
{
    size_t imageSize = sizeof(int) * resolutionX * resolutionY;
    int *h_result = (int *)malloc(imageSize);
    int *d_result;
    
    CUDA_CHECK(cudaMalloc((void **)&d_result, imageSize));
    
    double step = rectangleWidth / (double)resolutionX;
    double cxStart = rectangleCenterX - rectangleWidth / 2.0;
    double cyStart = rectangleCenterY + rectangleWidth * resolutionY / resolutionX / 2.0;
    
    dim3 blockSize(16, 16);
    dim3 gridSize((resolutionX + blockSize.x - 1) / blockSize.x,
                  (resolutionY + blockSize.y - 1) / blockSize.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    mandelbrotKernel<<<gridSize, blockSize>>>(d_result, cxStart, cyStart, 
                                               step, resolutionX, resolutionY, iterationsMax);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    
    CUDA_CHECK(cudaMemcpy(h_result, d_result, imageSize, cudaMemcpyDeviceToHost));
    
    savePPM("mandelbrot_cuda.ppm", h_result);
    printf("GPU Mandelbrot generated in %.3f ms\n", elapsed);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_result);
    free(h_result);
    
    return 0;
}