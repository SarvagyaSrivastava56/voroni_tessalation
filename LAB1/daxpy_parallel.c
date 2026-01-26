#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define VECTOR_SIZE (1 << 16)
#define MAX_THREADS 16
#define NUM_RUNS 100

void daxpy_parallel(double a, double *x, double *y, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < VECTOR_SIZE; i++) {
        x[i] = a * x[i] + y[i];
    }
}

int main() {
    double *x = (double *)malloc(VECTOR_SIZE * sizeof(double));
    double *y = (double *)malloc(VECTOR_SIZE * sizeof(double));
    double a = 2.0;
    
    // Initialize vectors with some values
    for (int i = 0; i < VECTOR_SIZE; i++) {
        x[i] = (double)i;
        y[i] = (double)(VECTOR_SIZE - i);
    }

    printf("Threads\tTime (ms)\tSpeedup\n");
    printf("----------------------------------\n");
    
    double base_time = 0;
    double max_speedup = 0.0;
    int best_thread_count = 1;
    
    for (int t = 1; t <= MAX_THREADS; t++) {
        double total_time = 0;
        
        // Warm-up run
        daxpy_parallel(a, x, y, t);
        
        // Time the operation
        for (int run = 0; run < NUM_RUNS; run++) {
            double start = omp_get_wtime();
            daxpy_parallel(a, x, y, t);
            total_time += (omp_get_wtime() - start) * 1000; // Convert to milliseconds
        }
        
        double avg_time = total_time / NUM_RUNS;
        
        if (t == 1) {
            base_time = avg_time;
        }
        
        double speedup = base_time / avg_time;
        printf("%d\t%.6f\t%.2f\n", t, avg_time, speedup);
        
        if (speedup > max_speedup) {
            max_speedup = speedup;
            best_thread_count = t;
        }
    }
    
    printf("\nMaximum speedup of %.2fx achieved with %d threads\n", max_speedup, best_thread_count);
    
    free(x);
    free(y);
    return 0;
}
