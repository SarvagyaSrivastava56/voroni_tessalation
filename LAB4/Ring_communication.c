#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int value;
    double start_total, end_total;
    double start_comm, end_comm;
    double comm_time = 0.0, comp_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    MPI_Barrier(MPI_COMM_WORLD);   // synchronize all processes
    start_total = MPI_Wtime();

    if (rank == 0) {
        value = 100;

        // Communication
        start_comm = MPI_Wtime();
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        end_comm = MPI_Wtime();
        comm_time += (end_comm - start_comm);

    } else {
        // Communication
        start_comm = MPI_Wtime();
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        end_comm = MPI_Wtime();
        comm_time += (end_comm - start_comm);

        // Computation
        double start_comp = MPI_Wtime();
        value += rank;
        double end_comp = MPI_Wtime();
        comp_time += (end_comp - start_comp);

        // Communication
        start_comm = MPI_Wtime();
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        end_comm = MPI_Wtime();
        comm_time += (end_comm - start_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_total = MPI_Wtime();

    double total_time = end_total - start_total;

    // Reduce to get maximum time across processes
    double max_total, max_comm, max_comp;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n===== PERFORMANCE METRICS =====\n");
        printf("Processes (p): %d\n", size);
        printf("Final Value: %d\n", value);
        printf("Execution Time (Tp): %f sec\n", max_total);
        printf("Communication Time: %f sec\n", max_comm);
        printf("Computation Time: %f sec\n", max_comp);

        double comm_percent = (max_comm / max_total) * 100.0;
        printf("Communication %%: %.2f%%\n", comm_percent);
        printf("===============================\n");
    }

    MPI_Finalize();
    return 0;
}