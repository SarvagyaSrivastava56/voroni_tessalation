#include <cuda_runtime.h>
#include <iostream>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "--- Device " << i << ": " << prop.name << " ---" << std::endl;
    std::cout << "Compute Capability:    " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Max Grid Dimensions:   (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")"
              << std::endl;
    std::cout << "Max Block Dimensions:  (" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")"
              << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "Global Memory:         "
              << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Mem per Block:  " << prop.sharedMemPerBlock / 1024
              << " KB" << std::endl;
    std::cout << "Constant Memory:       " << prop.totalConstMem / 1024 << " KB"
              << std::endl;
    std::cout << "Warp Size:             " << prop.warpSize << std::endl;
  }
  return 0;
}