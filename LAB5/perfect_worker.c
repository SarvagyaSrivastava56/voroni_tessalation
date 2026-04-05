#include "perfect_worker.h"

int test_perfect(int n) {
    if (n < 2) return -n;
    
    int sum = 1; // 1 is always a proper divisor
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i * i != n) {
                sum += n / i;
            }
        }
    }
    
    return (sum == n) ? n : -n;
}