#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

// Shared memory reduction kernel
__global__ void sumReduction(float* d_in, float* d_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? d_in[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// Run one reduction and return kernel time in ms
float runReduction(float* d_in, int n, int threadsPerBlock) {
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    float* d_intermediate;
    cudaMalloc(&d_intermediate, blocksPerGrid * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sumReduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_intermediate, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy back and accumulate to verify correctness
    std::vector<float> h_intermediate(blocksPerGrid);
    cudaMemcpy(h_intermediate.data(), d_intermediate, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float result = 0;
    for (float v : h_intermediate) result += v;

    cudaFree(d_intermediate);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print result once to verify (suppress after first call if needed)
    (void)result;
    return gpu_time;
}

int main() {
    // ── Fixed problem size (Strong Scaling: this never changes) ──
    const int n = 1 << 20; // 1,048,576 elements
    size_t size = n * sizeof(float);

    std::vector<float> h_in(n, 1.0f);

    // ── CPU Baseline ──
    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_sum = 0;
    for (int i = 0; i < n; i++) cpu_sum += h_in[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Upload input once ──
    float* d_in;
    cudaMalloc(&d_in, size);
    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

    // ── Strong Scaling: sweep threadsPerBlock (32 → 1024) ──
    // These are valid CUDA block sizes (must be multiples of warp size = 32)
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);

    // Warm-up run (GPU needs to JIT/initialize on first call)
    runReduction(d_in, n, 256);

    std::cout << "\n====== Strong Scaling Analysis ======\n";
    std::cout << "Fixed Problem Size: " << n << " floats ("
              << (size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "CPU Baseline:       " << cpu_sum << "  (" << cpu_time << " ms)\n\n";

    std::cout << std::left
              << std::setw(18) << "Threads/Block"
              << std::setw(16) << "Blocks/Grid"
              << std::setw(16) << "GPU Time (ms)"
              << std::setw(14) << "Speedup"
              << std::setw(16) << "Efficiency"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    // Use baseline (smallest config) as reference for relative GPU speedup
    float base_gpu_time = -1.0f;

    for (int k = 0; k < numConfigs; k++) {
        int tpb = blockSizes[k];
        int bpg = (n + tpb - 1) / tpb;

        // Average over 5 runs for stability
        float total = 0;
        for (int r = 0; r < 5; r++) total += runReduction(d_in, n, tpb);
        float avg_time = total / 5.0f;

        if (base_gpu_time < 0) base_gpu_time = avg_time; // set reference

        double speedup_vs_cpu    = cpu_time / avg_time;
        double speedup_vs_base   = base_gpu_time / avg_time; // relative GPU scaling
        // Efficiency = speedup / (tpb / base_tpb) — how well extra threads are used
        double parallelism_ratio = (double)tpb / blockSizes[0];
        double efficiency        = (speedup_vs_base / parallelism_ratio) * 100.0;

        std::cout << std::left
                  << std::setw(18) << tpb
                  << std::setw(16) << bpg
                  << std::setw(16) << std::fixed << std::setprecision(4) << avg_time
                  << std::setw(14) << std::setprecision(3) << speedup_vs_cpu << "x (vs CPU)"
                  << std::setw(16) << std::setprecision(1) << efficiency << "%"
                  << "\n";
    }

    std::cout << std::string(78, '-') << "\n";
    std::cout << "\nNote: Speedup = CPU time / GPU kernel time.\n";
    std::cout << "      Efficiency = (relative GPU speedup / thread ratio) x 100\n";
    std::cout << "      100% efficiency = perfect linear scaling with more threads.\n";

    cudaFree(d_in);
    return 0;
}