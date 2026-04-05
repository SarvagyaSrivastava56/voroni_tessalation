#ifndef PRIME_WORKER_H
#define PRIME_WORKER_H

#include <mpi.h>

/**
 * Checks if a number is prime.
 * Returns the number if prime, or its negative value if not prime.
 */
int test_prime(int n);

#endif