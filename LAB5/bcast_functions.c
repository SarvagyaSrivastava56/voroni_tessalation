#include "bcast_functions.h"

void my_bcast(double* data, int count, int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        // Rank 0 (Root) sends the massive array to all other processes one by one
        for (int i = 0; i < size; i++) {
            if (i != root) {
                MPI_Send(data, count, MPI_DOUBLE, i, 0, comm);
            }
        }
    } else {
        // All other ranks receive the data from the root
        MPI_Recv(data, count, MPI_DOUBLE, root, 0, comm, MPI_STATUS_IGNORE);
    }
}