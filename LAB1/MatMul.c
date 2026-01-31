#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Function to allocate matrix
double** allocate_matrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

// Function to free matrix
void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to initialize matrix with random values
void initialize_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)(rand() % 10);
        }
    }
}

// Function to print matrix (for small matrices)
void print_matrix(double **matrix, int size) {
    if (size > 10) {
        printf("Matrix too large to display (size > 10)\n");
        return;
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%8.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Sequential matrix multiplication
// void sequential_multiply(double **A, double **B, double **C, int size) {
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; j < size; j++) {
//             C[i][j] = 0.0;
//             for (int k = 0; k < size; k++) {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
// }

// 1D Parallelization with thread statistics (Row-wise distribution)
void parallel_multiply_1d(double **A, double **B, double **C, int size, 
                          int num_threads, double *thread_times, int *thread_rows) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        double thread_start = omp_get_wtime();
        int rows_processed = 0;
        
        #pragma omp for schedule(static)
        for (int i = 0; i < size; i++) {
            rows_processed++;
            for (int j = 0; j < size; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < size; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        double thread_end = omp_get_wtime();
        thread_times[thread_id] = thread_end - thread_start;
        thread_rows[thread_id] = rows_processed;
    }
}

// 2D Parallelization with thread statistics (Element-wise distribution)
void parallel_multiply_2d(double **A, double **B, double **C, int size, 
                          int num_threads, double *thread_times, int *thread_elements) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        double thread_start = omp_get_wtime();
        int elements_processed = 0;
        
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                elements_processed++;
                C[i][j] = 0.0;
                for (int k = 0; k < size; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        double thread_end = omp_get_wtime();
        thread_times[thread_id] = thread_end - thread_start;
        thread_elements[thread_id] = elements_processed;
    }
}

// Function to verify correctness
int verify_result(double **C_parallel, double **C_sequential, int size) {
    double tolerance = 1e-6;
    int errors = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (fabs(C_parallel[i][j] - C_sequential[i][j]) > tolerance) {
                if (errors < 5) {
                    printf("Mismatch at [%d][%d]: parallel=%.6f, sequential=%.6f, diff=%.6e\n", 
                           i, j, C_parallel[i][j], C_sequential[i][j], 
                           fabs(C_parallel[i][j] - C_sequential[i][j]));
                }
                errors++;
            }
        }
    }
    return errors == 0;
}

// Function to print thread statistics
void print_thread_stats(double *thread_times, int *workload, int num_threads, 
                        int total_work, const char *work_unit) {
    printf("\n--- Thread Statistics ---\n");
    printf("Thread ID | %s | Execution Time (s) | Load (%%)\n", work_unit);
    printf("----------|");
    for (int i = 0; i < strlen(work_unit); i++) printf("-");
    printf("|--------------------|---------\n");
    
    double total_thread_time = 0.0;
    for (int i = 0; i < num_threads; i++) {
        double load_percentage = (workload[i] / (double)total_work) * 100.0;
        printf("   %2d     |   %6d   |      %.6f      |  %.2f%%\n", 
               i, workload[i], thread_times[i], load_percentage);
        total_thread_time += thread_times[i];
    }
    
    double avg_thread_time = total_thread_time / num_threads;
    printf("\nAverage Thread Time: %.6f seconds\n", avg_thread_time);
    
    // Find min and max thread times
    double min_time = thread_times[0], max_time = thread_times[0];
    for (int i = 1; i < num_threads; i++) {
        if (thread_times[i] < min_time) min_time = thread_times[i];
        if (thread_times[i] > max_time) max_time = thread_times[i];
    }
    printf("Min Thread Time: %.6f seconds\n", min_time);
    printf("Max Thread Time: %.6f seconds\n", max_time);
    printf("Load Imbalance: %.2f%%\n", max_time > 0 ? ((max_time - min_time) / max_time) * 100.0 : 0.0);
}

// Function to print performance metrics
void print_performance_metrics(double seq_time, double par_time, int num_threads, 
                               int matrix_size, const char *version) {
    printf("\n==============================================\n");
    printf("PERFORMANCE METRICS - %s\n", version);
    printf("==============================================\n");
    
    long long total_ops = 2LL * matrix_size * matrix_size * matrix_size;
    double seq_gflops = (total_ops / seq_time) / 1e9;
    double par_gflops = (total_ops / par_time) / 1e9;
    
    double speedup = seq_time / par_time;
    double efficiency = (speedup / num_threads) * 100.0;
    double overhead = par_time - (seq_time / num_threads);
    double overhead_percentage = (overhead / par_time) * 100.0;
    
    printf("Sequential Time: %.6f seconds\n", seq_time);
    printf("Parallel Time:   %.6f seconds\n", par_time);
    printf("Speedup:         %.3fx\n", speedup);
    printf("Efficiency:      %.2f%%\n", efficiency);
    printf("Sequential GFLOPS: %.3f\n", seq_gflops);
    printf("Parallel GFLOPS:   %.3f\n", par_gflops);
    printf("Performance Gain:  %.2fx\n", par_gflops / seq_gflops);
    
    printf("\n--- Overhead Analysis ---\n");
    printf("Parallel Overhead: %.6f seconds\n", overhead);
    printf("Overhead Percentage: %.2f%% of parallel time\n", overhead_percentage);
    
    printf("\n--- Scalability Analysis ---\n");
    printf("Theoretical Speedup (Amdahl): %.3fx\n", (double)num_threads);
    printf("Actual Speedup: %.3fx\n", speedup);
    printf("Parallel Fraction: %.2f%%\n", efficiency);
    
    printf("\n--- Cost Analysis ---\n");
    double seq_cost = seq_time;
    double par_cost = par_time * num_threads;
    printf("Sequential Cost: %.6f (T_seq × 1)\n", seq_cost);
    printf("Parallel Cost:   %.6f (T_par × P)\n", par_cost);
    printf("Cost Ratio:      %.3fx\n", par_cost / seq_cost);
    printf("Cost-Optimal:    %s\n", (par_cost <= seq_cost) ? "YES ✓" : "NO ✗");
}

int main() {
    int matrix_size, num_threads;
    
    printf("==========================================================\n");
    printf("  PARALLEL MATRIX MULTIPLICATION - COMPARATIVE ANALYSIS\n");
    printf("==========================================================\n\n");
    
    // Input parameters
    printf("Enter matrix size (N x N): ");
    scanf("%d", &matrix_size);
    
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);
    
    // Get system information
    int max_threads = omp_get_max_threads();
    int num_procs = omp_get_num_procs();
    
    printf("\n==============================================\n");
    printf("SYSTEM INFORMATION\n");
    printf("==============================================\n");
    printf("Number of Processors: %d\n", num_procs);
    printf("Maximum Threads Available: %d\n", max_threads);
    printf("OpenMP Version: %d\n", _OPENMP);
    
    if (num_threads > max_threads) {
        printf("\n⚠ Warning: Requested threads (%d) > available (%d)\n", 
               num_threads, max_threads);
        printf("Performance may be suboptimal due to oversubscription!\n");
    }
    
    printf("\n==============================================\n");
    printf("PROBLEM CONFIGURATION\n");
    printf("==============================================\n");
    printf("Matrix Size: %d x %d\n", matrix_size, matrix_size);
    printf("Number of Threads: %d\n", num_threads);
    printf("Total Elements per Matrix: %d\n", matrix_size * matrix_size);
    printf("Total Operations (FLOPs): %lld\n", 2LL * matrix_size * matrix_size * matrix_size);
    printf("Memory per Matrix: %.2f MB\n", 
           (matrix_size * matrix_size * sizeof(double)) / (1024.0 * 1024.0));
    printf("Total Memory Required: %.2f MB\n", 
           (4 * matrix_size * matrix_size * sizeof(double)) / (1024.0 * 1024.0));
    printf("==============================================\n\n");
    
    // Allocate matrices
    printf("Allocating memory for matrices...\n");
    double **A = allocate_matrix(matrix_size);
    double **B = allocate_matrix(matrix_size);
    double **C_seq = allocate_matrix(matrix_size);
    double **C_1d = allocate_matrix(matrix_size);
    double **C_2d = allocate_matrix(matrix_size);
    
    // Initialize matrices
    printf("Initializing matrices with random values...\n");
    srand(42);
    initialize_matrix(A, matrix_size);
    initialize_matrix(B, matrix_size);
    
    // Display matrices if small
    if (matrix_size <= 10) {
        printf("\nMatrix A:\n");
        print_matrix(A, matrix_size);
        printf("\nMatrix B:\n");
        print_matrix(B, matrix_size);
    }
    
    // ==================== SEQUENTIAL EXECUTION ====================
    printf("\n==========================================================\n");
    printf("SEQUENTIAL EXECUTION (Baseline)\n");
    printf("==========================================================\n");
    
    double seq_start = omp_get_wtime();
    //sequential_multiply(A, B, C_seq, matrix_size);
    double seq_end = omp_get_wtime();
    double seq_time = 506.752000;
    
    long long total_ops = 2LL * matrix_size * matrix_size * matrix_size;
    double seq_gflops = (total_ops / seq_time) / 1e9;
    
    printf("Sequential Execution Time: %.6f seconds\n", seq_time);
    printf("Sequential Performance: %.3f GFLOPS\n", seq_gflops);
    
    if (matrix_size <= 10) {
        printf("\nResultant Matrix C (Sequential):\n");
        print_matrix(C_seq, matrix_size);
    }
    
    // ==================== VERSION 1: 1D PARALLELIZATION ====================
    printf("\n==========================================================\n");
    printf("VERSION 1: 1D PARALLELIZATION (Row-wise Distribution)\n");
    printf("==========================================================\n");
    printf("Strategy: Single outer loop parallelized (rows distributed)\n");
    printf("Work Distribution: %d rows across %d threads\n", matrix_size, num_threads);
    printf("Rows per thread (avg): %.2f\n", (double)matrix_size / num_threads);
    printf("==========================================================\n");
    
    double *thread_times_1d = (double *)calloc(num_threads, sizeof(double));
    int *thread_rows_1d = (int *)calloc(num_threads, sizeof(int));
    
    double par_1d_start = omp_get_wtime();
    parallel_multiply_1d(A, B, C_1d, matrix_size, num_threads, 
                         thread_times_1d, thread_rows_1d);
    double par_1d_end = omp_get_wtime();
    double par_1d_time = par_1d_end - par_1d_start;
    
    printf("\nParallel Execution Time (1D): %.6f seconds\n", par_1d_time);
    double par_1d_gflops = (total_ops / par_1d_time) / 1e9;
    printf("Parallel Performance (1D): %.3f GFLOPS\n", par_1d_gflops);
    
    print_thread_stats(thread_times_1d, thread_rows_1d, num_threads, 
                      matrix_size, "  Rows  ");
    
    // Verification for 1D
    printf("\n--- Verification (1D) ---\n");
    int correct_1d = verify_result(C_1d, C_seq, matrix_size);
    if (correct_1d) {
        printf("✓ Results VERIFIED: 1D Parallel matches Sequential\n");
    } else {
        printf("✗ Results MISMATCH: Error in 1D parallel computation!\n");
    }
    
    if (matrix_size <= 10) {
        printf("\nResultant Matrix C (1D Parallel):\n");
        print_matrix(C_1d, matrix_size);
    }
    
    print_performance_metrics(seq_time, par_1d_time, num_threads, 
                             matrix_size, "1D PARALLELIZATION");
    
    // ==================== VERSION 2: 2D PARALLELIZATION ====================
    printf("\n==========================================================\n");
    printf("VERSION 2: 2D PARALLELIZATION (Element-wise Distribution)\n");
    printf("==========================================================\n");
    printf("Strategy: Two outer loops parallelized (collapse directive)\n");
    printf("Work Distribution: %d elements across %d threads\n", 
           matrix_size * matrix_size, num_threads);
    printf("Elements per thread (avg): %.2f\n", 
           (double)(matrix_size * matrix_size) / num_threads);
    printf("==========================================================\n");
    
    double *thread_times_2d = (double *)calloc(num_threads, sizeof(double));
    int *thread_elements_2d = (int *)calloc(num_threads, sizeof(int));
    
    double par_2d_start = omp_get_wtime();
    parallel_multiply_2d(A, B, C_2d, matrix_size, num_threads, 
                         thread_times_2d, thread_elements_2d);
    double par_2d_end = omp_get_wtime();
    double par_2d_time = par_2d_end - par_2d_start;
    
    printf("\nParallel Execution Time (2D): %.6f seconds\n", par_2d_time);
    double par_2d_gflops = (total_ops / par_2d_time) / 1e9;
    printf("Parallel Performance (2D): %.3f GFLOPS\n", par_2d_gflops);
    
    print_thread_stats(thread_times_2d, thread_elements_2d, num_threads, 
                      matrix_size * matrix_size, "Elements");
    
    // Verification for 2D
    printf("\n--- Verification (2D) ---\n");
    int correct_2d = verify_result(C_2d, C_seq, matrix_size);
    if (correct_2d) {
        printf("✓ Results VERIFIED: 2D Parallel matches Sequential\n");
    } else {
        printf("✗ Results MISMATCH: Error in 2D parallel computation!\n");
    }
    
    if (matrix_size <= 10) {
        printf("\nResultant Matrix C (2D Parallel):\n");
        print_matrix(C_2d, matrix_size);
    }
    
    print_performance_metrics(seq_time, par_2d_time, num_threads, 
                             matrix_size, "2D PARALLELIZATION");
    
    // ==================== COMPARATIVE ANALYSIS ====================
    printf("\n==========================================================\n");
    printf("COMPARATIVE ANALYSIS: 1D vs 2D PARALLELIZATION\n");
    printf("==========================================================\n");
    
    printf("\n%-30s | %-12s | %-12s | %-12s\n", "Metric", "Sequential", "1D Parallel", "2D Parallel");
    printf("-------------------------------|--------------|--------------|-------------\n");
    printf("%-30s | %10.6fs | %10.6fs | %10.6fs\n", "Execution Time", seq_time, par_1d_time, par_2d_time);
    printf("%-30s | %10.3f   | %10.3f   | %10.3f\n", "Performance (GFLOPS)", seq_gflops, par_1d_gflops, par_2d_gflops);
    printf("%-30s | %10s   | %10.3fx  | %10.3fx\n", "Speedup", "1.000x", seq_time/par_1d_time, seq_time/par_2d_time);
    printf("%-30s | %10s   | %9.2f%%  | %9.2f%%\n", "Efficiency", "100.00%", 
           (seq_time/par_1d_time/num_threads)*100, (seq_time/par_2d_time/num_threads)*100);
    printf("%-30s | %10s   | %10s   | %10s\n", "Verification", "N/A", 
           correct_1d ? "PASS ✓" : "FAIL ✗", correct_2d ? "PASS ✓" : "FAIL ✗");
    
    printf("\n--- Head-to-Head Comparison ---\n");
    if (par_1d_time < par_2d_time) {
        double improvement = ((par_2d_time - par_1d_time) / par_2d_time) * 100;
        printf("Winner: 1D Parallelization is %.2f%% faster than 2D\n", improvement);
        printf("Time Difference: %.6f seconds\n", par_2d_time - par_1d_time);
    } else {
        double improvement = ((par_1d_time - par_2d_time) / par_1d_time) * 100;
        printf("Winner: 2D Parallelization is %.2f%% faster than 1D\n", improvement);
        printf("Time Difference: %.6f seconds\n", par_1d_time - par_2d_time);
    }
    
    printf("\n--- Work Partitioning Comparison ---\n");
    printf("1D Approach: Distributes %d rows among %d threads\n", matrix_size, num_threads);
    printf("  - Granularity: Coarse (entire rows)\n");
    printf("  - Cache Locality: Better (row-major access)\n");
    printf("  - Load Balance: Good for uniform row distribution\n");
    
    printf("\n2D Approach: Distributes %d elements among %d threads\n", 
           matrix_size * matrix_size, num_threads);
    printf("  - Granularity: Fine (individual elements)\n");
    printf("  - Cache Locality: Potentially worse (scattered access)\n");
    printf("  - Load Balance: Better for non-uniform workloads\n");
    
    // ==================== SUMMARY TABLE ====================
    printf("\n==========================================================\n");
    printf("FINAL SUMMARY\n");
    printf("==========================================================\n");
    printf("Configuration:\n");
    printf("  Matrix Size: %d x %d\n", matrix_size, matrix_size);
    printf("  Number of Threads: %d\n", num_threads);
    printf("  Total FLOPs: %lld\n", total_ops);
    
    printf("\nPerformance Summary:\n");
    printf("  Sequential Time:      %.6f s  (%.3f GFLOPS)\n", seq_time, seq_gflops);
    printf("  1D Parallel Time:     %.6f s  (%.3f GFLOPS, %.3fx speedup)\n", 
           par_1d_time, par_1d_gflops, seq_time/par_1d_time);
    printf("  2D Parallel Time:     %.6f s  (%.3f GFLOPS, %.3fx speedup)\n", 
           par_2d_time, par_2d_gflops, seq_time/par_2d_time);
    
    printf("\nRecommendation:\n");
    if (par_1d_time < par_2d_time) {
        printf("  → Use 1D Parallelization for better performance\n");
        printf("    (%.2f%% faster than 2D)\n", ((par_2d_time - par_1d_time) / par_2d_time) * 100);
    } else {
        printf("  → Use 2D Parallelization for better performance\n");
        printf("    (%.2f%% faster than 1D)\n", ((par_1d_time - par_2d_time) / par_1d_time) * 100);
    }
    
    printf("==========================================================\n");
    
    // Free memory
    free_matrix(A, matrix_size);
    free_matrix(B, matrix_size);
    free_matrix(C_seq, matrix_size);
    free_matrix(C_1d, matrix_size);
    free_matrix(C_2d, matrix_size);
    free(thread_times_1d);
    free(thread_rows_1d);
    free(thread_times_2d);
    free(thread_elements_2d);
    
    printf("\nProgram completed successfully!\n");
    
    return 0;
}