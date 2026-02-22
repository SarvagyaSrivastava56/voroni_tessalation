#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8

int main(int argc, char *argv[])
{
    int rank, size;
    int A[N] = {1,2,3,4,5,6,7,8};
    int B[N] = {8,7,6,5,4,3,2,1};

    int *local_A, *local_B;
    int chunk;
    int local_dot = 0, global_dot = 0;

    double start_total, end_total;
    double start_comm, end_comm;
    double comm_time = 0.0, comp_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk = N / size;

    local_A = (int*)malloc(chunk * sizeof(int));
    local_B = (int*)malloc(chunk * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    start_total = MPI_Wtime();

    // 🔹 Scatter Vector A
    start_comm = MPI_Wtime();
    MPI_Scatter(A, chunk, MPI_INT,
                local_A, chunk, MPI_INT,
                0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    // 🔹 Scatter Vector B
    start_comm = MPI_Wtime();
    MPI_Scatter(B, chunk, MPI_INT,
                local_B, chunk, MPI_INT,
                0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    // 🔹 Local Computation
    double start_comp = MPI_Wtime();
    for (int i = 0; i < chunk; i++)
        local_dot += local_A[i] * local_B[i];
    double end_comp = MPI_Wtime();
    comp_time += (end_comp - start_comp);

    // 🔹 Reduce partial results
    start_comm = MPI_Wtime();
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    end_total = MPI_Wtime();

    double total_time = end_total - start_total;

    // Take maximum across processes
    double max_total, max_comm, max_comp;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n===== PARALLEL DOT PRODUCT =====\n");
        printf("Processes (p): %d\n", size);
        printf("Dot Product: %d\n", global_dot);
        printf("Expected: 120\n");

        printf("\nPerformance Metrics:\n");
        printf("Execution Time (Tp): %f sec\n", max_total);
        printf("Communication Time: %f sec\n", max_comm);
        printf("Computation Time: %f sec\n", max_comp);

        double comm_percent = (max_comm / max_total) * 100.0;
        printf("Communication %%: %.2f%%\n", comm_percent);
        printf("===============================\n");
    }

    free(local_A);
    free(local_B);

    MPI_Finalize();
    return 0;
}