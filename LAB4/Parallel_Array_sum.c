#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100   // array size

int main(int argc, char *argv[])
{
    int rank, size;
    int *array = NULL;
    int local_sum = 0, global_sum = 0;
    int chunk;

    double start_total, end_total;
    double start_comm, end_comm;
    double comm_time = 0.0, comp_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk = N / size;

    int *local_array = (int *)malloc(chunk * sizeof(int));

    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;   // 1 to 100
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_total = MPI_Wtime();

    // 🔹 Scatter (Communication)
    start_comm = MPI_Wtime();
    MPI_Scatter(array, chunk, MPI_INT,
                local_array, chunk, MPI_INT,
                0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    // 🔹 Local Computation
    double start_comp = MPI_Wtime();
    for (int i = 0; i < chunk; i++)
        local_sum += local_array[i];
    double end_comp = MPI_Wtime();
    comp_time += (end_comp - start_comp);

    // 🔹 Reduce (Communication)
    start_comm = MPI_Wtime();
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    end_total = MPI_Wtime();

    double total_time = end_total - start_total;

    // Take maximum time across processes
    double max_total, max_comm, max_comp;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n===== PARALLEL ARRAY SUM =====\n");
        printf("Processes (p): %d\n", size);
        printf("Global Sum: %d\n", global_sum);
        printf("Expected Sum: 5050\n");

        double average = (double)global_sum / N;
        printf("Average: %.2f\n", average);

        printf("Execution Time (Tp): %f sec\n", max_total);
        printf("Communication Time: %f sec\n", max_comm);
        printf("Computation Time: %f sec\n", max_comp);

        double comm_percent = (max_comm / max_total) * 100.0;
        printf("Communication %%: %.2f%%\n", comm_percent);
        printf("==============================\n");
    }

    if (rank == 0) free(array);
    free(local_array);

    MPI_Finalize();
    return 0;
}