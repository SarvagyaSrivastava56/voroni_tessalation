#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>

template<typename T>
std::string toStr(T v) { std::ostringstream ss; ss << v; return ss.str(); }

// ─────────────────────────────────────────────────────────────────────────────
// Matrix Addition Kernel
// Each thread computes ONE output element: C[row][col] = A[row][col] + B[row][col]
//
// Analysis per thread:
//   Global Memory Reads  : 2  (one read from A, one from B)
//   Global Memory Writes : 1  (one write to C)
//   Floating-point Ops   : 1  (one integer addition, counted as 1 FLOP)
//
// For an N x N matrix:
//   Total Reads   = 2 * N * N
//   Total Writes  = 1 * N * N
//   Total FLOPs   = 1 * N * N
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matAddKernel(const int* A, const int* B, int* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];   // 2 reads, 1 add, 1 write
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU Baseline
// ─────────────────────────────────────────────────────────────────────────────
double cpuMatAdd(const std::vector<int>& A, const std::vector<int>& B,
                 std::vector<int>& C, int N) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N * N; i++) C[i] = A[i] + B[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU run: returns kernel time (ms), writes result to h_C
// ─────────────────────────────────────────────────────────────────────────────
float gpuMatAdd(const int* d_A, const int* d_B, int* d_C,
                int N, int blockSize) {
    dim3 threads(blockSize, blockSize);
    dim3 blocks((N + blockSize - 1) / blockSize,
                (N + blockSize - 1) / blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify GPU result against CPU result
// ─────────────────────────────────────────────────────────────────────────────
bool verify(const std::vector<int>& cpu, const std::vector<int>& gpu, int N) {
    for (int i = 0; i < N * N; i++)
        if (cpu[i] != gpu[i]) return false;
    return true;
}

int main() {
    // ── Fixed problem size (Strong Scaling: N stays constant) ──
    const int N    = 4096;           // 4096 x 4096 matrix
    const size_t sz = (size_t)N * N * sizeof(int);

    std::cout << "======================================\n";
    std::cout << "  CUDA Integer Matrix Addition\n";
    std::cout << "======================================\n";
    std::cout << "Matrix Size : " << N << " x " << N << " (" << N*N << " elements)\n";
    std::cout << "Element Type: int (32-bit)\n";
    std::cout << "Memory (A+B+C): " << 3.0 * sz / (1024*1024) << " MB\n\n";

    // ── Analysis answers ──
    long long total_elements = (long long)N * N;
    std::cout << "─── Kernel Analysis (per launch) ───────────────────────\n";
    std::cout << "  Floating-point ops (FLOPs) : " << total_elements
              << "  (1 addition per element)\n";
    std::cout << "  Global memory reads        : " << 2 * total_elements
              << "  (2 reads per element: A[i] and B[i])\n";
    std::cout << "  Global memory writes       : " << total_elements
              << "  (1 write per element: C[i])\n";
    std::cout << "  Arithmetic Intensity       : "
              << std::fixed << std::setprecision(4)
              << (float)total_elements / (3.0f * sz)
              << " FLOPs/byte  (memory-bound)\n";
    std::cout << "─────────────────────────────────────────────────────────\n\n";

    // Initialize host matrices
    std::vector<int> h_A(N * N), h_B(N * N), h_C_cpu(N * N), h_C_gpu(N * N);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i % 100;
        h_B[i] = (i * 2) % 100;
    }

    // ── CPU Baseline ──
    double cpu_time = cpuMatAdd(h_A, h_B, h_C_cpu, N);
    std::cout << "CPU Time (baseline): " << cpu_time << " ms\n\n";

    // ── Allocate device memory ──
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sz);
    cudaMalloc(&d_B, sz);
    cudaMalloc(&d_C, sz);
    cudaMemcpy(d_A, h_A.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sz, cudaMemcpyHostToDevice);

    // ── Warm-up ──
    gpuMatAdd(d_A, d_B, d_C, N, 16);

    // ─────────────────────────────────────────────────────────────────────
    // Strong Scaling: fixed N=4096, sweep 2D block sizes
    // Valid 2D block sizes where blockSize*blockSize <= 1024 (max threads/block)
    // ─────────────────────────────────────────────────────────────────────
    int blockSizes[] = {4, 8, 16, 32};
    int numConfigs   = sizeof(blockSizes) / sizeof(blockSizes[0]);

    std::cout << "======================================\n";
    std::cout << "  Strong Scaling Analysis\n";
    std::cout << "  (N = " << N << " fixed, block size varies)\n";
    std::cout << "======================================\n\n";

    std::cout << std::left
              << std::setw(14) << "Block Shape"
              << std::setw(14) << "Threads/Blk"
              << std::setw(16) << "Grid Shape"
              << std::setw(16) << "GPU Time (ms)"
              << std::setw(14) << "Speedup"
              << std::setw(14) << "GFLOP/s"
              << "Correct?\n";
    std::cout << std::string(90, '-') << "\n";

    float base_gpu_time = -1.0f;

    for (int k = 0; k < numConfigs; k++) {
        int bs  = blockSizes[k];
        int bpg = (N + bs - 1) / bs;

        // Average 5 runs
        float total = 0;
        for (int r = 0; r < 5; r++) total += gpuMatAdd(d_A, d_B, d_C, N, bs);
        float avg_ms = total / 5.0f;

        if (base_gpu_time < 0) base_gpu_time = avg_ms;

        // Copy result back for verification
        cudaMemcpy(h_C_gpu.data(), d_C, sz, cudaMemcpyDeviceToHost);
        bool ok = verify(h_C_cpu, h_C_gpu, N);

        double speedup = cpu_time / avg_ms;
        // GFLOP/s = total_elements / (time_in_seconds * 1e9)
        double gflops  = (double)total_elements / (avg_ms * 1e-3) / 1e9;

        std::cout << std::left
                  << std::setw(14) << (toStr(bs) + "x" + toStr(bs))
                  << std::setw(14) << (bs * bs)
                  << std::setw(16) << (toStr(bpg) + "x" + toStr(bpg))
                  << std::setw(16) << std::fixed << std::setprecision(4) << avg_ms
                  << std::setw(14) << std::setprecision(2) << speedup
                  << std::setw(14) << std::setprecision(4) << gflops
                  << (ok ? "YES" : "FAIL") << "\n";
    }

    std::cout << std::string(90, '-') << "\n";
    std::cout << "\nNote: Speedup = CPU time / GPU kernel time.\n";
    std::cout << "      GFLOP/s = N^2 additions / kernel_time_in_seconds / 1e9\n";
    std::cout << "      This kernel is MEMORY-BOUND (arithmetic intensity < 1 FLOP/byte)\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
