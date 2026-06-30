# Heterogeneous Parallel Weighted Voronoi Tessellation

Adaptive density-weighted sampling via **Centroidal Voronoi Tessellation (CVT)** using Lloyd's relaxation. Parallelised across CPU (OpenMP) and GPU (CUDA, RTX 2050 / Ampere `sm_86`).

This project implements and compares brute-force and Jump Flooding Algorithm (JFA) approaches across CPU and GPU hardware to isolate and analyze pure algorithmic gains vs. pure hardware speedups.

---

## 🚀 Performance Highlights ($512 \times 512$ Grid, $50$ Lloyd Iterations)

By implementing a **4-way speedup decomposition** baseline, this project isolates the exact performance contributions of the hardware transition and the algorithmic transition:

| Site Count ($N$) | CPU Brute-Force (`BF_serial`) | CPU JFA (`JFA_serial`) | GPU Brute-Force (`BF_cuda`) | GPU JFA (`JFA_cuda`) | Pure Algorithmic Gain (CPU) | Pure Hardware Gain (BF) | Total Combined Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64** | 2,171.3 ms | 2,616.2 ms | 31.8 ms | 67.2 ms | 0.8× | 68.3× | **32×** |
| **128** | 4,171.7 ms | 2,772.1 ms | 26.8 ms | 61.8 ms | 1.5× | 155.9× | **67×** |
| **256** | 8,283.2 ms | 3,187.3 ms | 33.7 ms | 61.8 ms | 2.6× | 245.5× | **134×** |
| **512** | 16,056.8 ms | 3,678.3 ms | 51.8 ms | 56.1 ms | 4.4× | 310.2× | **286×** |
| **1,024** | 31,146.1 ms | 4,362.2 ms | 91.0 ms | 53.4 ms | 7.1× | 342.4× | **583×** |
| **2,048** | 60,640.6 ms | 5,133.8 ms | 142.6 ms | 51.7 ms | 11.8× | 425.1× | **1,172×** |

### 💡 Key Architectural Insights
1. **Algorithmic vs. Hardware Speedup:** At $N=2048$, the total speedup is **$1,172\times$**. Our benchmark isolates this: **$11.8\times$** comes purely from JFA's complexity reduction ($O(P\cdot N) \rightarrow O(P \log P)$ on CPU), while **$425.1\times$** comes from GPU hardware acceleration of the brute-force baseline.
2. **Compute-Bound vs. Memory-Bound Tradeoffs:** Brute-force is *compute-bound* (heavy arithmetic distance calculations in registers), allowing the GPU to fully hide latency. JFA is *memory-bound* (multiple passes of ping-pong reads/writes with strided lookups). Thus, the GPU brute-force baseline runs exceptionally fast, shrinking JFA's relative speedup on GPU to $2.76\times$ (compared to $11.8\times$ on CPU).

---

## 📂 Project Structure

```
voronoi_cvt/
├── Makefile
├── include/
│   ├── common.h              # Point2D, DensityGrid, RunResult
│   ├── timer.h               # std::chrono timer
│   ├── density.h             # Density generators (Gaussian, Checkerboard, Kirsch Stress)
│   ├── voronoi.h             # Backend declarations
│   └── output.h              # PNG & CSV output utilities
├── src/
│   ├── main.cpp              # CLI entry point and dispatch
│   ├── density.cpp           # Kirsch's stress field & standard densities
│   ├── output.cpp            # stb_image_write PNG + CSV writer
│   ├── voronoi_serial.cpp    # CPU Brute-Force [O(P*N)]
│   ├── voronoi_jfa_serial.cpp# CPU JFA [O(P log P)]
│   ├── voronoi_omp.cpp       # Multi-core CPU OpenMP Brute-Force [O(P*N)]
│   ├── voronoi_cuda_bf.cu    # GPU Brute-Force [O(P*N)]
│   └── voronoi_cuda.cu       # GPU JFA [O(P log P) + shared-memory centroids]
├── scripts/
│   └── scaling_analysis.py   # Multi-backend benchmarking & plotting
├── third_party/
│   └── stb_image_write.h     # Downloader wrapper
├── data/                     # Custom PGM density maps
└── results/                  # Plots, speedups, and CSV benchmarks
```

---

## 🛠️ Build & Run

### Prerequisites
- **Linux/WSL2:** GCC with C++17, OpenMP, CUDA Toolkit, Python 3 (`matplotlib`, `numpy`).
- **Windows:** MSYS2/MinGW64 `make`, MSVC host compiler, CUDA Toolkit (`nvcc` on PATH), Python 3.

### Compilation
```bash
# Compile Serial + OpenMP CPU backends
make

# Compile with CUDA Enabled (requires nvcc and MSVC compiler on Windows)
make ENABLE_CUDA=1
```

### Running Benchmarks
```bash
# Run all backends on a default 1024x1024 grid, 256 sites
./voronoi_cvt.exe --backend all --density gaussian --width 1024 --height 1024 --sites 256

# CLI Options:
#  --backend   [serial | jfa_serial | omp | cuda | cuda_bf | all]
#  --density   [gaussian | check | pgm | stress]
#  --input     [path/to/image.pgm]  (required if density=pgm)
#  --threads   [N] (OMP thread count limit)
```

---

## 📈 Experiments

All plots are outputted directly to the `results/` folder:

```bash
# 1. Run the 4-way Speedup Decomposition (Isolates algorithmic vs. hardware gains)
make ENABLE_CUDA=1 decompose

# 2. Run Strong and Weak Scaling Experiments
make ENABLE_CUDA=1 scale

# 3. Sweep Block Sizes (Sweeps 32 to 512 threads/block for CUDA JFA launch configuration)
make ENABLE_CUDA=1 blk

# 4. Measure PCIe Host-to-Device / Device-to-Host Bandwidth
make ENABLE_CUDA=1 bw
```

---

## 🔬 Implementation Details

### 1. GPU Jump Flooding (1+JFA Variant)
Brute-force assignment calculates the Euclidean distance from every pixel $P$ to all $N$ sites ($O(P \cdot N)$ complexity). JFA resolves this in $O(P \log P)$ by broadcasting coordinate offsets inside step strides $2^k, 2^{k-1}, \dots, 1$. To resolve boundary errors, a $+1$ correction pass at stride=1 is executed at the end.

### 2. Block-Local Shared-Memory Centroid Staging
To calculate CVT updates, each pixel adds its weighted coordinates to the corresponding site centroid accumulator. Naive global atomic operations saturate DRAM bandwidth. 
This project maps Dynamic Shared Memory to stage centroids inside on-chip shared memory blocks ($3 \times N \times \text{sizeof}(double)$ allocation):
1. Cooperatively zeros shared-memory arrays.
2. Performs localized `atomicAdd` on-chip (~100× lower latency, bank-conflict-free).
3. Flushes block-level sums to global memory with a single atomic reduction write per block-site pair.
This reduces DRAM memory traffic by a factor of the block size ($32\times$ or $256\times$).
