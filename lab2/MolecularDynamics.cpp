// =======================================================
// Molecular Dynamics – Lennard Jones (OpenMP)
// Parallel + race free + reduction + performance metrics
// =======================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif
using namespace std;


const double epsilon = 1.0;
const double sigma   = 1.0;
const double cutoff  = 3.0;


long long getPageFaults()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.PageFaultCount;

#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    // minor + major
    return usage.ru_minflt + usage.ru_majflt;
#endif
}


// -----------------------------------------------
// Random particle generator
// -----------------------------------------------
void initParticles(int N,
                   vector<double>& x,
                   vector<double>& y,
                   vector<double>& z)
{
    for(int i=0;i<N;i++){
        x[i] = (double)rand() / RAND_MAX * 50.0;
        y[i] = (double)rand() / RAND_MAX * 50.0;
        z[i] = (double)rand() / RAND_MAX * 50.0;
    }
}


// -----------------------------------------------
// Lennard-Jones force computation
// schedule_type:
// 0=static 1=dynamic 2=guided
// -----------------------------------------------
double compute_forces(int N, int threads, int schedule_type)
{
    vector<double> x(N), y(N), z(N);
    initParticles(N,x,y,z);

    vector<double> fx(N,0), fy(N,0), fz(N,0);

    omp_set_num_threads(threads);

    double energy = 0.0;

    long long pf_start = getPageFaults();
double start = omp_get_wtime();


#pragma omp parallel reduction(+:energy)
{
    int tid = omp_get_thread_num();

    vector<double> lfx(N,0), lfy(N,0), lfz(N,0); // thread local

    if(schedule_type==0)
    {
#pragma omp for schedule(static)
        for(int i=0;i<N;i++)
        for(int j=i+1;j<N;j++)
        {
            double dx=x[i]-x[j];
            double dy=y[i]-y[j];
            double dz=z[i]-z[j];

            double r2 = dx*dx+dy*dy+dz*dz;

            if(r2>cutoff*cutoff) continue;

            double inv_r2 = 1.0/r2;
            double sr2 = (sigma*sigma)*inv_r2;
            double sr6 = sr2*sr2*sr2;
            double sr12 = sr6*sr6;

            double e = 4*epsilon*(sr12-sr6);
            energy += e;

            double f = 24*epsilon*(2*sr12-sr6)*inv_r2;

            double fxij=f*dx, fyij=f*dy, fzij=f*dz;

            lfx[i]+=fxij; lfy[i]+=fyij; lfz[i]+=fzij;
            lfx[j]-=fxij; lfy[j]-=fyij; lfz[j]-=fzij;
        }
    }
    else if(schedule_type==1)
    {
#pragma omp for schedule(dynamic,4)
        for(int i=0;i<N;i++)
        for(int j=i+1;j<N;j++)
        {
            double dx=x[i]-x[j];
            double dy=y[i]-y[j];
            double dz=z[i]-z[j];

            double r2 = dx*dx+dy*dy+dz*dz;
            if(r2>cutoff*cutoff) continue;

            double inv_r2 = 1.0/r2;
            double sr2 = (sigma*sigma)*inv_r2;
            double sr6 = sr2*sr2*sr2;
            double sr12 = sr6*sr6;

            energy += 4*epsilon*(sr12-sr6);

            double f = 24*epsilon*(2*sr12-sr6)*inv_r2;

            lfx[i]+=f*dx; lfy[i]+=f*dy; lfz[i]+=f*dz;
            lfx[j]-=f*dx; lfy[j]-=f*dy; lfz[j]-=f*dz;
        }
    }
    else
    {
#pragma omp for schedule(guided)
        for(int i=0;i<N;i++)
        for(int j=i+1;j<N;j++)
        {
            double dx=x[i]-x[j];
            double dy=y[i]-y[j];
            double dz=z[i]-z[j];

            double r2 = dx*dx+dy*dy+dz*dz;
            if(r2>cutoff*cutoff) continue;

            double inv_r2 = 1.0/r2;
            double sr2 = (sigma*sigma)*inv_r2;
            double sr6 = sr2*sr2*sr2;
            double sr12 = sr6*sr6;

            energy += 4*epsilon*(sr12-sr6);

            double f = 24*epsilon*(2*sr12-sr6)*inv_r2;

            lfx[i]+=f*dx; lfy[i]+=f*dy; lfz[i]+=f*dz;
            lfx[j]-=f*dx; lfy[j]-=f*dy; lfz[j]-=f*dz;
        }
    }

#pragma omp critical
    for(int i=0;i<N;i++){
        fx[i]+=lfx[i];
        fy[i]+=lfy[i];
        fz[i]+=lfz[i];
    }
}

    double end = omp_get_wtime();
long long pf_end = getPageFaults();

cout << "Page Faults: " << (pf_end - pf_start) << "\n";

return end-start;


    
}


// -----------------------------------------------
// Main benchmark
// -----------------------------------------------
int main(int argc,char* argv[])
{
    if(argc<3){
        cout<<"Usage: md <threads> <particles>\n";
        return 0;
    }

    int threads=atoi(argv[1]);
    int N=atoi(argv[2]);

    vector<string> names={"STATIC","DYNAMIC","GUIDED"};

    double t_serial = compute_forces(N,1,0);

    cout<<"\n===== Molecular Dynamics Benchmark =====\n";

    for(int s=0;s<3;s++)
    {
        double t_parallel = compute_forces(N,threads,s);

        double speedup = t_serial/t_parallel;
        double efficiency = speedup/threads;

        cout<<"Schedule: "<<names[s]<<"\n";
        cout<<"Time: "<<t_parallel<<"\n";
        cout<<"Speedup: "<<speedup<<"\n";
        cout<<"Efficiency: "<<efficiency<<"\n\n";
    }
}
