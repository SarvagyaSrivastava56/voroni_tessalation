#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define COUNT 10

int main(int argc, char *argv[])
{
    int rank, size;
    int numbers[COUNT];
    int local_max, local_min;
    int global_max, global_min;

    struct {
        int value;
        int rank;
    } local_maxloc, local_minloc, global_maxloc, global_minloc;

    double start_total, end_total;
    double start_comm, end_comm;
    double comm_time = 0.0, comp_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);  // unique seed per process

    MPI_Barrier(MPI_COMM_WORLD);
    start_total = MPI_Wtime();

    // 🔹 Generate Random Numbers
    double start_comp = MPI_Wtime();
    for (int i = 0; i < COUNT; i++)
        numbers[i] = rand() % 1001;   // 0 to 1000

    local_max = numbers[0];
    local_min = numbers[0];

    for (int i = 1; i < COUNT; i++) {
        if (numbers[i] > local_max)
            local_max = numbers[i];
        if (numbers[i] < local_min)
            local_min = numbers[i];
    }
    double end_comp = MPI_Wtime();
    comp_time += (end_comp - start_comp);

    // Prepare for MAXLOC / MINLOC
    local_maxloc.value = local_max;
    local_maxloc.rank  = rank;

    local_minloc.value = local_min;
    local_minloc.rank  = rank;

    // 🔹 Reduce for MAX
    start_comm = MPI_Wtime();
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT,
               MPI_MAX, 0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    // 🔹 Reduce for MIN
    start_comm = MPI_Wtime();
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT,
               MPI_MIN, 0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    // 🔹 Reduce for MAXLOC
    start_comm = MPI_Wtime();
    MPI_Reduce(&local_maxloc, &global_maxloc, 1,
               MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    comm_time += (end_comm - start_comm);

    // 🔹 Reduce for MINLOC
    start_comm = MPI_Wtime();
    MPI_Reduce(&local_minloc, &global_minloc, 1,
               MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
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
        printf("\n===== GLOBAL MAX & MIN =====\n");
        printf("Processes: %d\n", size);

        printf("Global Maximum (MPI_MAX): %d\n", global_max);
        printf("Global Minimum (MPI_MIN): %d\n", global_min);

        printf("Global Maximum (MPI_MAXLOC): %d from Process %d\n",
               global_maxloc.value, global_maxloc.rank);

        printf("Global Minimum (MPI_MINLOC): %d from Process %d\n",
               global_minloc.value, global_minloc.rank);

        printf("\nPerformance Metrics:\n");
        printf("Execution Time (Tp): %f sec\n", max_total);
        printf("Communication Time: %f sec\n", max_comm);
        printf("Computation Time: %f sec\n", max_comp);

        double comm_percent = (max_comm / max_total) * 100.0;
        printf("Communication %%: %.2f%%\n", comm_percent);
        printf("============================\n");
    }

    MPI_Finalize();
    return 0;
}