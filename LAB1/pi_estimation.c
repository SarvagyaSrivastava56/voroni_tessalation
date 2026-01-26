#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define NUM_STEPS 100000000L  // 100 million steps
#define MAX_THREADS 32
#define RUNS 3

double step;

// Serial version (baseline)
double calculate_pi_serial(long num_steps, double *pi_result) {
    double sum = 0.0;
    double x;
    step = 1.0 / (double)num_steps;
    
    double start = omp_get_wtime();
    
    for (long i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    
    *pi_result = step * sum;
    return omp_get_wtime() - start;
}

// Parallel version: Private variables + Critical section once per thread
double calculate_pi_parallel(long num_steps, int num_threads, double *pi_result) {
    double sum = 0.0;
    step = 1.0 / (double)num_steps;
    
    omp_set_num_threads(num_threads);
    double start = omp_get_wtime();
    
    #pragma omp parallel
    {
        double x;
        double partial_sum = 0.0;  // Private variable - each thread has its own
        
        // Each thread processes its share of iterations independently
        #pragma omp for
        for (long i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            partial_sum = partial_sum + 4.0 / (1.0 + x * x);
        }
        
        // Critical section: only ONE thread at a time can add to global sum
        // This happens only ONCE per thread (not per iteration!)
        #pragma omp critical
        {
            sum = sum + partial_sum;
        }
    }
    
    *pi_result = step * sum;
    return omp_get_wtime() - start;
}

void print_separator() {
    printf("================================================================\n");
}

int main() {
    printf("\n");
    print_separator();
    printf("PARALLEL π CALCULATION - SPEEDUP ANALYSIS\n");
    print_separator();
    printf("Formula: π = ∫₀¹ 4/(1+x²) dx\n");
    printf("Method: Riemann sum with private variables + critical section\n");
    printf("Number of steps: %ld\n", NUM_STEPS);
    printf("Actual value of π: %.15f\n", M_PI);
    printf("Available threads: %d\n", omp_get_max_threads());
    print_separator();
    
    // Calculate serial baseline
    printf("\nCalculating serial baseline...\n");
    double serial_time = 0.0;
    double pi_value;
    
    for (int run = 0; run < RUNS; run++) {
        serial_time += calculate_pi_serial(NUM_STEPS, &pi_value);
    }
    serial_time /= RUNS;
    
    printf("Serial execution time: %.4f seconds\n", serial_time);
    printf("Calculated π value: %.15f\n", pi_value);
    printf("Error from actual π: %.2e\n\n", fabs(pi_value - M_PI));
    
    print_separator();
    printf("PARALLEL IMPLEMENTATION: PRIVATE + CRITICAL\n");
    print_separator();
    printf("\nThreads\tTime (s)\tSpeedup\t\tEfficiency\tπ Value\n");
    printf("-------\t--------\t-------\t\t----------\t-------\n");
    
    double max_speedup = 0.0;
    int optimal_threads = 1;
    
    // Test different thread counts
    for (int t = 1; t <= MAX_THREADS; t++) {
        double total_time = 0.0;
        double pi_result = 0.0;
        
        for (int run = 0; run < RUNS; run++) {
            total_time += calculate_pi_parallel(NUM_STEPS, t, &pi_result);
        }
        
        double avg_time = total_time / RUNS;
        double speedup = serial_time / avg_time;
        double efficiency = (speedup / t) * 100.0;
        
        if (speedup > max_speedup) {
            max_speedup = speedup;
            optimal_threads = t;
        }
        
        printf("%d\t%.4f\t\t%.2fx\t\t%.2f%%\t\t%.15f\n", 
               t, avg_time, speedup, efficiency, pi_result);
    }
    

    
    return 0;
}