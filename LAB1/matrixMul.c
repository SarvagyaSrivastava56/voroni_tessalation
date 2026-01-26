#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define SIZE 1000
#define RUNS 5
#define MAX_THREADS 32

double *A, *B, *C, *C_serial;

// Access matrix element: matrix[i][j] = matrix[i * SIZE + j]
#define MAT(matrix, i, j) (matrix[(i) * SIZE + (j)])

// Allocate contiguous 1D array for matrix
double* allocate_matrix() {
    double *matrix = (double*)malloc(SIZE * SIZE * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }
    return matrix;
}

// Initialize matrices
void initialize_matrices() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            MAT(A, i, j) = (double)(i + j);
            MAT(B, i, j) = (double)(i - j);
            MAT(C, i, j) = 0.0;
        }
    }
}

// Reset result matrix
void reset_result() {
    memset(C, 0, SIZE * SIZE * sizeof(double));
}

// Serial matrix multiplication (baseline)
double serial_matmul() {
    double start = omp_get_wtime();
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                sum += MAT(A, i, k) * MAT(B, k, j);
            }
            MAT(C, i, j) = sum;
        }
    }
    
    return omp_get_wtime() - start;
}

// 1D Parallel: Parallelize outer loop (row-wise partition)
double parallel_1d(int threads) {
    omp_set_num_threads(threads);
    double start = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                sum += MAT(A, i, k) * MAT(B, k, j);
            }
            MAT(C, i, j) = sum;
        }
    }
    
    return omp_get_wtime() - start;
}

// 2D Parallel: Parallelize both outer loops (2D block partition)
double parallel_2d(int threads) {
    omp_set_num_threads(threads);
    double start = omp_get_wtime();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                sum += MAT(A, i, k) * MAT(B, k, j);
            }
            MAT(C, i, j) = sum;
        }
    }
    
    return omp_get_wtime() - start;
}

// Verify correctness by comparing a few elements
int verify_result(double *expected, double *result) {
    for (int i = 0; i < SIZE; i += SIZE/10) {
        for (int j = 0; j < SIZE; j += SIZE/10) {
            double diff = MAT(expected, i, j) - MAT(result, i, j);
            if (diff < 0) diff = -diff;
            if (diff > 1e-6) {
                printf("Mismatch at [%d][%d]: expected %.2f, got %.2f\n", 
                       i, j, MAT(expected, i, j), MAT(result, i, j));
                return 0;
            }
        }
    }
    return 1;
}

void print_separator() {
    printf("================================================================\n");
}

int main() {
    printf("\nAllocating matrices...\n");
    
    // Allocate matrices with error checking
    A = allocate_matrix();
    B = allocate_matrix();
    C = allocate_matrix();
    C_serial = allocate_matrix();
    
    printf("Matrices allocated successfully.\n");
    printf("Total memory allocated: %.2f MB\n\n", 
           (4.0 * SIZE * SIZE * sizeof(double)) / (1024.0 * 1024.0));
    
    initialize_matrices();
    
    print_separator();
    printf("PARALLEL MATRIX MULTIPLICATION ANALYSIS\n");
    print_separator();
    printf("Matrix size: %dx%d\n", SIZE, SIZE);
    printf("Total elements: %d\n", SIZE * SIZE);
    printf("Operations per multiplication: %ld FLOPS\n", (long)2 * SIZE * SIZE * SIZE);
    printf("Memory per matrix: %.2f MB\n", (SIZE * SIZE * sizeof(double)) / (1024.0 * 1024.0));
    printf("Runs per configuration: %d\n", RUNS);
    printf("Max threads to test: %d\n", MAX_THREADS);
    printf("Available threads: %d\n", omp_get_max_threads());
    print_separator();
    
    // Run serial version for baseline
    printf("\nRunning serial version...\n");
    double serial_time = 0;
    for (int run = 0; run < RUNS; run++) {
        printf("  Run %d/%d...\r", run + 1, RUNS);
        fflush(stdout);
        reset_result();
        serial_time += serial_matmul();
    }
    serial_time /= RUNS;
    
    // Save serial result for verification
    memcpy(C_serial, C, SIZE * SIZE * sizeof(double));
    
    printf("\nSerial execution time: %.4f seconds\n", serial_time);
    printf("Performance: %.2f GFLOPS\n\n", (2.0 * SIZE * SIZE * SIZE) / (serial_time * 1e9));
    
    print_separator();
    printf("1D PARALLEL IMPLEMENTATION (Row-wise partitioning)\n");
    print_separator();
    printf("Strategy: Each thread processes complete rows\n");
    printf("Work distribution: Row i assigned to thread (i %% num_threads)\n");
    printf("\nThreads\tTime (s)\tSpeedup\t\tEfficiency\tGFLOPS\n");
    printf("-------\t--------\t-------\t\t----------\t------\n");
    
    double max_speedup_1d = 0;
    int optimal_threads_1d = 1;
    
    for (int t = 1; t <= MAX_THREADS; t++) {
        double total_time = 0;
        
        for (int run = 0; run < RUNS; run++) {
            reset_result();
            total_time += parallel_1d(t);
        }
        
        double avg_time = total_time / RUNS;
        double speedup = serial_time / avg_time;
        double efficiency = (speedup / t) * 100;
        double gflops = (2.0 * SIZE * SIZE * SIZE) / (avg_time * 1e9);
        
        if (speedup > max_speedup_1d) {
            max_speedup_1d = speedup;
            optimal_threads_1d = t;
        }
        
        // Verify correctness
        if (t == 2 || t == MAX_THREADS) {
            if (!verify_result(C_serial, C)) {
                printf("WARNING: Result verification failed for %d threads!\n", t);
            }
        }
        
        printf("%d\t%.4f\t\t%.2fx\t\t%.2f%%\t\t%.2f\n", 
               t, avg_time, speedup, efficiency, gflops);
    }
    
    printf("\n");
    print_separator();
    printf("2D PARALLEL IMPLEMENTATION (2D block partitioning)\n");
    print_separator();
    printf("Strategy: Both i and j loops parallelized with collapse(2)\n");
    printf("Work distribution: 2D grid of blocks distributed among threads\n");
    printf("Advantage: Better load balancing, more parallelism opportunities\n");
    printf("\nThreads\tTime (s)\tSpeedup\t\tEfficiency\tGFLOPS\n");
    printf("-------\t--------\t-------\t\t----------\t------\n");
    
    double max_speedup_2d = 0;
    int optimal_threads_2d = 1;
    
    for (int t = 1; t <= MAX_THREADS; t++) {
        double total_time = 0;
        
        for (int run = 0; run < RUNS; run++) {
            reset_result();
            total_time += parallel_2d(t);
        }
        
        double avg_time = total_time / RUNS;
        double speedup = serial_time / avg_time;
        double efficiency = (speedup / t) * 100;
        double gflops = (2.0 * SIZE * SIZE * SIZE) / (avg_time * 1e9);
        
        if (speedup > max_speedup_2d) {
            max_speedup_2d = speedup;
            optimal_threads_2d = t;
        }
        
        // Verify correctness
        if (t == 2 || t == MAX_THREADS) {
            if (!verify_result(C_serial, C)) {
                printf("WARNING: Result verification failed for %d threads!\n", t);
            }
        }
        
        printf("%d\t%.4f\t\t%.2fx\t\t%.2f%%\t\t%.2f\n", 
               t, avg_time, speedup, efficiency, gflops);
    }
    
    printf("\n");
    print_separator();
    printf("PERFORMANCE COMPARISON\n");
    print_separator();
    printf("\n1D Implementation:\n");
    printf("  Optimal threads: %d\n", optimal_threads_1d);
    printf("  Maximum speedup: %.2fx\n", max_speedup_1d);
    
    printf("\n2D Implementation:\n");
    printf("  Optimal threads: %d\n", optimal_threads_2d);
    printf("  Maximum speedup: %.2fx\n", max_speedup_2d);
    
    if (max_speedup_2d > max_speedup_1d) {
        printf("\n2D implementation is %.2fx faster than 1D at optimal settings\n", 
               max_speedup_2d / max_speedup_1d);
    } else {
        printf("\n1D implementation performs similarly or better (simpler overhead)\n");
    }
    
    printf("\n");
    print_separator();
    printf("KEY DIFFERENCES: 1D vs 2D THREADING\n");
    print_separator();
    
    printf("\n1D Threading (Row-wise):\n");
    printf("   + Simple work distribution\n");
    printf("   + Good cache locality (processes entire row)\n");
    printf("   + Less overhead for small thread counts\n");
    printf("   - Limited parallelism (max %d parallel tasks)\n", SIZE);
    printf("   - Load imbalance if SIZE not divisible by threads\n");
    printf("   - Each thread does SIZE*SIZE operations\n");
    
    printf("\n2D Threading (Block-wise):\n");
    printf("   + More parallelism (max %d parallel tasks)\n", SIZE * SIZE);
    printf("   + Better load balancing across threads\n");
    printf("   + Finer-grained work distribution\n");
    printf("   + Better scalability with many cores\n");
    printf("   - Slightly more overhead from collapse(2)\n");
    printf("   - Each thread does SIZE operations (smaller chunks)\n");
    
    printf("\n");
    print_separator();
    printf("WHY PERFORMANCE SATURATES\n");
    print_separator();

     // Cleanup
    free(A);
    free(B);
    free(C);
    free(C_serial);
    
    printf("Program completed successfully.\n");
    
    return 0;
}