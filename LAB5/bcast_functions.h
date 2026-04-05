#ifndef BCAST_FUNCTIONS_H
#define BCAST_FUNCTIONS_H

#include <mpi.h>

// Part A: Custom function using a for-loop and standard MPI_Send/Recv
void my_bcast(double* data, int count, int root, MPI_Comm comm);

#endif