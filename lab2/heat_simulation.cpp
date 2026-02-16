// =======================================================
// Heat Diffusion 2D – OpenMP Benchmark
// Parallel + reduction + scheduling + page faults
// =======================================================

#include <iostream>
#include <vector>
#include <cmath>
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
const int N = 4000;      // grid size
const int STEPS = 300;  // time steps
const double alpha = 0.1;
const double dt = 0.1;
const double dx = 1.0;

const double c = alpha * dt / (dx*dx);


// =======================================================
// Page faults
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
// Initialize plate (hot center)
// =======================================================
void initialize(vector<double>& grid)
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            grid[i*N+j] = 0.0;

    grid[(N/2)*N + N/2] = 1000.0;
}


// =======================================================
// Heat solver
// schedule_type:
// 0 static
// 1 dynamic
// 2 guided
// =======================================================
double simulate(int threads, int schedule_type)
{
    omp_set_num_threads(threads);

    if(schedule_type==0)
        omp_set_schedule(omp_sched_static, 0);
    else if(schedule_type==1)
        omp_set_schedule(omp_sched_dynamic, 32);
    else
        omp_set_schedule(omp_sched_guided, 0);


    vector<double> old(N*N), next(N*N);

    initialize(old);

    long long pf_start = getPageFaults();
    double start = omp_get_wtime();


    // ===================================================
    // TIME STEPS
    // ===================================================
    for(int step=0; step<STEPS; step++)
    {
        double totalHeat = 0.0;

#pragma omp parallel for schedule(runtime) reduction(+:totalHeat)
        for(int i=1;i<N-1;i++)
        {
            for(int j=1;j<N-1;j++)
            {
                int id = i*N + j;

                double newT =
                    old[id] +
                    c*( old[id-N] + old[id+N] +
                        old[id-1] + old[id+1] -
                        4*old[id] );

                next[id] = newT;

                totalHeat += newT;
            }
        }

        swap(old,next);
    }


    double end = omp_get_wtime();
    long long pf_end = getPageFaults();

    cout << "Page Faults: " << (pf_end-pf_start) << "\n";

    return end-start;
}


// =======================================================
// MAIN BENCHMARK
// =======================================================
int main(int argc,char* argv[])
{
    if(argc<2){
        cout<<"Usage: heat <threads>\n";
        return 0;
    }

    int threads = atoi(argv[1]);

    vector<string> names={"STATIC","DYNAMIC","GUIDED"};

    double t_serial = simulate(1,0);

    cout<<"\n===== Heat Diffusion Benchmark =====\n";
    cout<<"Grid: "<<N<<"x"<<N<<" Steps: "<<STEPS<<"\n\n";

    for(int s=0;s<3;s++)
    {
        double t_parallel = simulate(threads,s);

        double speedup = t_serial/t_parallel;
        double efficiency = speedup/threads;

        cout<<"Schedule: "<<names[s]<<"\n";
        cout<<"Time: "<<t_parallel<<"\n";
        cout<<"Speedup: "<<speedup<<"\n";
        cout<<"Efficiency: "<<efficiency<<"\n\n";
    }
}
