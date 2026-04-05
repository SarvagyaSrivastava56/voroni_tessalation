#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <mpi.h>

/**
 * Calculates the local dot product for a chunk of vectors.
 * A[i] = 1.0, B[i] = 2.0 * multiplier
 */
double compute_local_dot(long long local_n, double multiplier);

#endif