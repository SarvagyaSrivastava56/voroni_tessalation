// =======================================================
// Smith-Waterman – Wavefront (OpenMP Benchmark Version)
// Parallel + race free + scheduling + page faults
// =======================================================

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using namespace std;


// =======================================================
// FIXED PROBLEM SIZE
// =======================================================
const int N = 5000;
const int MATCH = 2;
const int MISMATCH = -1;
const int GAP = 1;


// =======================================================
// Page fault counter (same as your MD code)
// =======================================================
long long getPageFaults()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.PageFaultCount;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_minflt + usage.ru_majflt;
#endif
}


// =======================================================
// Random DNA generator
// =======================================================
string randomDNA(int n)
{
    static const char dna[] = {'A','C','G','T'};
    string s(n,'A');

    for(int i=0;i<n;i++)
        s[i] = dna[rand()%4];

    return s;
}


// =======================================================
// Wavefront Smith-Waterman
// schedule_type:
// 0=static 1=dynamic 2=guided
// =======================================================
double compute_alignment(int threads, int schedule_type)
{
    string A = randomDNA(N);
    string B = randomDNA(N);

    omp_set_num_threads(threads);

    // set runtime schedule
    if(schedule_type==0)
        omp_set_schedule(omp_sched_static, 0);
    else if(schedule_type==1)
        omp_set_schedule(omp_sched_dynamic, 32);
    else
        omp_set_schedule(omp_sched_guided, 0);


    // contiguous allocation → fewer page faults
    vector<int> H((N+1)*(N+1), 0);

    auto cell = [&](int i,int j)->int& {
        return H[i*(N+1)+j];
    };

    int maxScore = 0;

    long long pf_start = getPageFaults();
    double start = omp_get_wtime();


    // ===================================================
    // WAVEFRONT PARALLELIZATION
    // ===================================================
    for(int d=1; d<=2*N-1; d++)
    {
        int start_i = max(1, d-N+1);
        int end_i   = min(N, d);

#pragma omp parallel for schedule(runtime) reduction(max:maxScore)
        for(int i=start_i; i<=end_i; i++)
        {
            int j = d-i+1;

            int diag = cell(i-1,j-1) +
                (A[i-1]==B[j-1] ? MATCH : MISMATCH);

            int up   = cell(i-1,j) - GAP;
            int left = cell(i,j-1) - GAP;

            int val = max(0, max(diag, max(up, left)));

            cell(i,j) = val;

            if(val > maxScore)
                maxScore = val;
        }
    }


    double end = omp_get_wtime();
    long long pf_end = getPageFaults();

    cout << "Page Faults: " << (pf_end - pf_start) << "\n";

    return end-start;
}


// =======================================================
// MAIN BENCHMARK (same style as your MD code)
// =======================================================
int main(int argc,char* argv[])
{
    if(argc < 2){
        cout << "Usage: sw <threads>\n";
        return 0;
    }

    int threads = atoi(argv[1]);

    vector<string> names = {"STATIC","DYNAMIC","GUIDED"};

    // serial baseline
    double t_serial = compute_alignment(1,0);

    cout << "\n===== Smith-Waterman Wavefront Benchmark =====\n";
    cout << "Matrix size : " << N << " x " << N << "\n\n";

    for(int s=0;s<3;s++)
    {
        double t_parallel = compute_alignment(threads,s);

        double speedup = t_serial / t_parallel;
        double efficiency = speedup / threads;

        cout << "Schedule: " << names[s] << "\n";
        cout << "Time: " << t_parallel << "\n";
        cout << "Speedup: " << speedup << "\n";
        cout << "Efficiency: " << efficiency << "\n\n";
    }
}
