#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <omp.h>
#include <windows.h>
#include <psapi.h>

// Forward declarations
void correlate_seq(int, int, const float*, float*);
void correlate_par(int, int, const float*, float*);

// =================================================
// Get Page Fault Count (Windows API)
// =================================================
SIZE_T get_page_faults()
{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PageFaultCount;

    return 0;
}

// =================================================
// MAIN
// =================================================
int main(int argc, char* argv[])
{
    // ---------------------------------
    // Check arguments
    // ---------------------------------
    if (argc != 4)
    {
        std::cout << "Usage: correlate <rows ny> <cols nx> <threads>\n";
        return 0;
    }

    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);
    int threads = std::atoi(argv[3]);

    omp_set_num_threads(threads);

    std::cout << "Matrix size : " << ny << " x " << nx << "\n";
    std::cout << "Threads     : " << threads << "\n\n";

    // ---------------------------------
    // Allocate
    // ---------------------------------
    std::vector<float> data(ny * nx);
    std::vector<float> res_seq(ny * ny);
    std::vector<float> res_par(ny * ny);

    // Warm-up memory (avoid first-touch page faults affecting timing)
    for (auto &v : res_seq) v = 0;
    for (auto &v : res_par) v = 0;

    // ---------------------------------
    // Random initialization
    // ---------------------------------
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto &v : data)
        v = dist(gen);

    // ---------------------------------
    // Sequential timing + page faults
    // ---------------------------------
    SIZE_T pf_before_seq = get_page_faults();

    double t1 = omp_get_wtime();
    correlate_seq(ny, nx, data.data(), res_seq.data());
    double t2 = omp_get_wtime();

    SIZE_T pf_after_seq = get_page_faults();

    double Tseq = t2 - t1;

    // ---------------------------------
    // Parallel timing + page faults
    // ---------------------------------
    SIZE_T pf_before_par = get_page_faults();

    t1 = omp_get_wtime();
    correlate_par(ny, nx, data.data(), res_par.data());
    t2 = omp_get_wtime();

    SIZE_T pf_after_par = get_page_faults();

    double Tpar = t2 - t1;

    // ---------------------------------
    // Metrics
    // ---------------------------------
    double speedup = Tseq / Tpar;
    double efficiency = speedup / threads;

    // ---------------------------------
    // Output
    // ---------------------------------
    std::cout << "Sequential time : " << Tseq << " sec\n";
    std::cout << "Parallel time   : " << Tpar << " sec\n";
    std::cout << "Speedup         : " << speedup << "\n";
    std::cout << "Efficiency      : " << efficiency * 100 << " %\n\n";

    std::cout << "Page Faults (Sequential) : "
              << (pf_after_seq - pf_before_seq) << "\n";

    std::cout << "Page Faults (Parallel)   : "
              << (pf_after_par - pf_before_par) << "\n";

    return 0;
}
