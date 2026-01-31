#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 100000000;  // 100 million steps for accuracy
double step;

int main() {
    int i, num_threads;
    double x, pi, sum = 0.0;
    double start_time, end_time;
    
    printf("==========================================================\n");
    printf("  PI CALCULATION USING NUMERICAL INTEGRATION (OpenMP)\n");
    printf("==========================================================\n");
    
    printf("\nEnter the number of threads to use: ");
    if (scanf("%d", &num_threads) != 1) {
        printf("Error: Invalid input!\n");
        return 1;
    }
    
    if (num_threads <= 0) {
        printf("Error: Number of threads must be positive!\n");
        return 1;
    }
    
    int max_threads = omp_get_max_threads();
    printf("\nMaximum threads available: %d\n", max_threads);
    
    if (num_threads > max_threads) {
        printf("Warning: Requested threads (%d) exceeds available threads (%d)\n", 
               num_threads, max_threads);
        printf("Using maximum available threads: %d\n", max_threads);
        num_threads = max_threads;
    }
    
    printf("\nCalculating Pi with %ld steps using %d threads...\n", num_steps, num_threads);
    
    step = 1.0 / (double)num_steps;
    
    // First run sequential version for baseline
    printf("Running sequential version for baseline...\n");
    double seq_sum = 0.0;
    double seq_start = omp_get_wtime();
    
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        seq_sum += 4.0 / (1.0 + x * x);
    }
    
    double seq_pi = step * seq_sum;
    double seq_end = omp_get_wtime();
    double seq_time = seq_end - seq_start;
    
    // Now run parallel version
    printf("Running parallel version...\n");
    sum = 0.0;
    start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for reduction(+:sum) private(x)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    
    pi = step * sum;
    end_time = omp_get_wtime();
    
    printf("\n");
    printf("============================================================\n");
    printf("RESULTS\n");
    printf("============================================================\n");
    printf("Calculated Pi (sequential): %.15f\n", seq_pi);
    printf("Calculated Pi (parallel):   %.15f\n", pi);
    printf("Actual Pi:                  3.141592653589793\n");
    printf("Error (sequential):         %.15e\n", seq_pi - 3.141592653589793);
    printf("Error (parallel):           %.15e\n", pi - 3.141592653589793);
    printf("Time taken (sequential):    %.6f seconds\n", seq_time);
    printf("Time taken (parallel):      %.6f seconds\n", end_time - start_time);
    printf("Speedup:                    %.2fx\n", seq_time / (end_time - start_time));
    printf("Efficiency:                 %.2f%%\n", (seq_time / (end_time - start_time)) / num_threads * 100);
    printf("Threads used:               %d\n", num_threads);
    printf("============================================================\n");
    
    return 0;
}