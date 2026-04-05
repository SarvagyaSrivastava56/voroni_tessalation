#include "prime_worker.h"
#include <math.h>

int test_prime(int n) {
    if (n < 2) return -n;
    if (n == 2) return n;
    if (n % 2 == 0) return -n;
    
    int limit = (int)sqrt(n);
    for (int i = 3; i <= limit; i += 2) {
        if (n % i == 0) return -n;
    }
    return n;
}