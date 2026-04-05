#ifndef PERFECT_WORKER_H
#define PERFECT_WORKER_H

#include <mpi.h>

/**
 * Checks if a number is perfect (sum of proper divisors equals the number).
 * Returns the number if perfect, or its negative value if not.
 */
int test_perfect(int n);

#endif