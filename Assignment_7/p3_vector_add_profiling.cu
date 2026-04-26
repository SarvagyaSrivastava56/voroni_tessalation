#include <stdio.h>

#define N 1000000

// 1.1 Statically defined global variables (device memory)
// Note: Size declared at compile time, no cudaMalloc needed for these.
__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vectorAdd() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main() {
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Copy to statically defined device memory
    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

    // 1.2 Record timing data of the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Retrieve results
    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

    printf("1.2 Kernel Execution Time: %f ms\n\n", milliseconds);

    // 1.3 Query device properties & calculate theoretical bandwidth
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    int memoryClockRate;
    // In CUDA 13.x, memoryClockRate is obtained via cudaDeviceGetAttribute
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, deviceId);
    
    // memoryClockRate is in kilohertz (kHz), memoryBusWidth is in bits
    // theoreticalBW = memoryClockRate * memoryBusWidth
    // multiply by two (DDR is double pumped)
    double kilobits_per_second = (double)memoryClockRate * (double)prop.memoryBusWidth * 2.0;
    
    // Convert to Gb/s (Gigabits per second)
    double gigabits_per_second = kilobits_per_second / 1000000.0;
    
    // Convert to GB/s (Gigabytes per second)
    double theoreticalBW = gigabits_per_second / 8.0;

    printf("1.3 Theoretical Memory Bandwidth: %f GB/s\n", theoreticalBW);

    // 1.4 Calculate measured bandwidth
    // Vector add reads 2 arrays (d_A, d_B), writes 1 array (d_C). 
    // Data processed per array = N * sizeof(float)
    double rBytes = 2.0 * N * sizeof(float);
    double wBytes = 1.0 * N * sizeof(float);
    
    // Convert time to seconds
    double t_seconds = milliseconds / 1000.0;
    
    // Calculate measured bandwidth in bytes/s, then convert to GB/s (using 1e9 for Giga)
    double measuredBW = ((rBytes + wBytes) / 1e9) / t_seconds;

    printf("1.4 Measured Memory Bandwidth: %f GB/s\n", measuredBW);

    // Validate result
    bool correct = true;
    for(int i = 0; i < N; i++) {
        if(h_C[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    printf("\nVector Add Validation: %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
