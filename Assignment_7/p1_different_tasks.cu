#include <stdio.h>

#define N 1024

__global__ void different_tasks_kernel(int *d_in, int *d_out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Step 7: Create the kernel for adding the first N integers
    if (tid == 0) {
        // Task a: Iterative approach
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += d_in[i];
        }
        d_out[0] = sum;
    } else if (tid == 1) {
        // Task b: Direct formula
        d_out[1] = n * (n + 1) / 2;
    }
}

int main() {
    // Step 1: Define the value of N (Already defined as macro N = 1024)

    // Step 2: Create two arrays: One for input and another for output
    int h_in[N];
    int h_out[2] = {0, 0}; // 2 outputs: one for thread 0, one for thread 1
    
    // Step 4: Fill the array with first N integers
    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1; // integers from 1 to N
    }
    
    // Step 3: Allocate memory on device for the data
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, 2 * sizeof(int));
    
    // Step 5: Copy the data from host to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Step 6: Define block and grid sizes
    dim3 grid(1);
    dim3 block(2);
    
    // Step 7: Call it from host
    different_tasks_kernel<<<grid, block>>>(d_in, d_out, N);
    
    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Task A (Iterative Sum by Thread 0): %d\n", h_out[0]);
    printf("Task B (Formula Sum by Thread 1)  : %d\n", h_out[1]);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
