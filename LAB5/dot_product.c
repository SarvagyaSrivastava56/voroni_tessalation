#include "dot_product.h"
#include <stdlib.h>

double compute_local_dot(long long local_n, double multiplier) {
    double *local_A = (double*)malloc(local_n * sizeof(double));
    double *local_B = (double*)malloc(local_n * sizeof(double));
    double local_sum = 0.0;

    if (local_A == NULL || local_B == NULL) return -1.0;

    // Step 2: Local Generation to save memory
    for (long long i = 0; i < local_n; i++) {
        local_A[i] = 1.0;
        local_B[i] = 2.0 * multiplier;
    }

    // Step 3: Local Compute
    for (long long i = 0; i < local_n; i++) {
        local_sum += local_A[i] * local_B[i];
    }

    free(local_A);
    free(local_B);
    return local_sum;
}