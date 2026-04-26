#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1000

// --- GPU CUDA MERGE SORT (Bottom-Up) ---

__device__ void merge_gpu(int* source, int* dest, int start, int middle, int end) {
    int i = start;
    int j = middle;
    int k = start;
    
    while (i < middle && j < end) {
        if (source[i] <= source[j]) {
            dest[k++] = source[i++];
        } else {
            dest[k++] = source[j++];
        }
    }
    while (i < middle) {
        dest[k++] = source[i++];
    }
    while (j < end) {
        dest[k++] = source[j++];
    }
}

__global__ void mergeSortKernel(int* source, int* dest, int width, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = idx * 2 * width;
    
    if (start < n) {
        int middle = min(start + width, n);
        int end = min(start + 2 * width, n);
        merge_gpu(source, dest, start, middle, end);
    }
}

void cudaMergeSort(int* h_arr, int n, float *elapsed_time) {
    int *d_A, *d_B;
    cudaMalloc((void**)&d_A, n * sizeof(int));
    cudaMalloc((void**)&d_B, n * sizeof(int));
    
    cudaMemcpy(d_A, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 threads(256);
    cudaEventRecord(start);
    
    for (int width = 1; width < n; width *= 2) {
        dim3 blocks((n / (2 * width)) + 1);
        mergeSortKernel<<<blocks, threads>>>(d_A, d_B, width, n);
        cudaDeviceSynchronize();
        
        // Swap pointers for next iteration
        int* temp = d_A;
        d_A = d_B;
        d_B = temp;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time, start, stop);
    
    cudaMemcpy(h_arr, d_A, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// --- CPU PIPELINED / PARALLEL MERGE SORT (OpenMP) ---

void merge_cpu(int* arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));
    
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    
    free(L);
    free(R);
}

void mergeSortCPUPipelined(int* arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        
        // Pipelining using standard OpenMP sections (Compatible with MSVC /openmp)
        #pragma omp parallel sections if(r-l > 200)
        {
            #pragma omp section
            mergeSortCPUPipelined(arr, l, m);
            #pragma omp section
            mergeSortCPUPipelined(arr, m + 1, r);
        }
        
        merge_cpu(arr, l, m, r);
    }
}

int main() {
    int *h_arr_cpu = (int*)malloc(N * sizeof(int));
    int *h_arr_gpu = (int*)malloc(N * sizeof(int));
    
    // Initialize arrays
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        int val = rand() % 10000;
        h_arr_cpu[i] = val;
        h_arr_gpu[i] = val;
    }
    
    // a. CPU Pipelined Parallelization Timing (OpenMP Tasks)
    // Note: Compile with -Xcompiler /openmp
    double start_cpu = omp_get_wtime();
    mergeSortCPUPipelined(h_arr_cpu, 0, N - 1);
    double end_cpu = omp_get_wtime();
    double cpu_time_ms = (end_cpu - start_cpu) * 1000.0;
    
    // b. GPU CUDA Parallelization Timing
    float gpu_time_ms = 0;
    cudaMergeSort(h_arr_gpu, N, &gpu_time_ms);
    
    // c. Compare performance
    printf("--- Performance Comparison (n=%d) ---\n", N);
    printf("CPU (Pipelined/OpenMP) Time: %f ms\n", cpu_time_ms);
    printf("GPU (CUDA) Time          : %f ms\n", gpu_time_ms);
    printf("Speedup (CPU / GPU)      : %fx\n", cpu_time_ms / (gpu_time_ms == 0 ? 1.0 : gpu_time_ms));
    
    // Verify
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_arr_cpu[i] != h_arr_gpu[i]) correct = false;
        if (i < N - 1 && h_arr_gpu[i] > h_arr_gpu[i+1]) correct = false;
    }
    printf("\nSort Validation: %s\n", correct ? "PASSED" : "FAILED");
    
    free(h_arr_cpu);
    free(h_arr_gpu);
    return 0;
}
