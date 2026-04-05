#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 65536 // 2^16

int main(int argc, char** argv) {
    int rank, size;
    double A = 2.5;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) printf("Error: N (%d) must be divisible by Number of Processes (%d)\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int local_N = N / size;
    double *X_full = NULL, *Y_full = NULL, *X_seq = NULL;
    double *local_X = (double*)malloc(local_N * sizeof(double));
    double *local_Y = (double*)malloc(local_N * sizeof(double));

    double seq_time = 0.0;

    if (rank == 0) {
        X_full = (double*)malloc(N * sizeof(double));
        Y_full = (double*)malloc(N * sizeof(double));
        X_seq = (double*)malloc(N * sizeof(double));

        for (int i = 0; i < N; i++) {
            X_full[i] = 1.0;
            Y_full[i] = 2.0;
            X_seq[i] = 1.0;
        }

        // Sequential Baseline
        double seq_start = MPI_Wtime();
        for (int i = 0; i < N; i++) {
            X_seq[i] = A * X_seq[i] + Y_full[i];
        }
        seq_time = MPI_Wtime() - seq_start;
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    
    double t_start, t_end, comm_start, comm_end;
    double local_comm_total = 0.0;

    // --- Parallel Phase ---
    t_start = MPI_Wtime();

    // Scatter
    comm_start = MPI_Wtime();
    MPI_Scatter(X_full, local_N, MPI_DOUBLE, local_X, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y_full, local_N, MPI_DOUBLE, local_Y, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_comm_total += (MPI_Wtime() - comm_start);

    // Computation
    for (int i = 0; i < local_N; i++) {
        local_X[i] = A * local_X[i] + local_Y[i];
    }

    // Gather
    comm_start = MPI_Wtime();
    MPI_Gather(local_X, local_N, MPI_DOUBLE, X_full, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_comm_total += (MPI_Wtime() - comm_start);

    t_end = MPI_Wtime();
    double local_total_time = t_end - t_start;

    // Collect max times from all ranks
    double max_total_time, max_comm_time;
    MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_total, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double speedup = seq_time / max_total_time;
        double efficiency = (speedup / size) * 100.0;
        double comm_percent = (max_comm_time / max_total_time) * 100.0;

        printf("\n================ DAXPY MPI RESULTS ================\n");
        printf("Vector Size:      2^16 (%d)\n", N);
        printf("Processes:        %d\n", size);
        printf("---------------------------------------------------\n");
        printf("Sequential Time:  %f s\n", seq_time);
        printf("Parallel Time:    %f s\n", max_total_time);
        printf("Comm. Time:       %f s\n", max_comm_time);
        printf("---------------------------------------------------\n");
        printf("Speedup:          %.2fx\n", speedup);
        printf("Efficiency:       %.2f%%\n", efficiency);
        printf("Comm. Overhead:   %.2f%%\n", comm_percent);
        printf("===================================================\n");

        free(X_full); free(Y_full); free(X_seq);
    }

    free(local_X); free(local_Y);
    MPI_Finalize();
    return 0;
}