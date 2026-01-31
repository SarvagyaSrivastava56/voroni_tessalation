#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* ---------- SERIAL ---------- */
void daxpy_serial(int n, double a, double *x, double *y)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}


/* ---------- PARALLEL ---------- */
void daxpy_parallel(int n, double a, double *x, double *y, int threads)
{
    omp_set_num_threads(threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}


/* ---------- MAIN ---------- */
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <size> <threads>\n", argv[0]);
        return 0;
    }

    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    double a = 2.5;

    double *x  = malloc(n * sizeof(double));
    double *y1 = malloc(n * sizeof(double));
    double *y2 = malloc(n * sizeof(double));

    if (!x || !y1 || !y2)
    {
        printf("Memory allocation failed\n");
        return 0;
    }

    /* Initialize */
    for (int i = 0; i < n; i++)
    {
        x[i] = 1.0;
        y1[i] = 2.0;
        y2[i] = 2.0;
    }

    double t1, t2, serial_time, parallel_time;

    /* ---- Serial ---- */
    t1 = omp_get_wtime();
    daxpy_serial(n, a, x, y1);
    t2 = omp_get_wtime();
    serial_time = t2 - t1;

    /* ---- Parallel ---- */
    t1 = omp_get_wtime();
    daxpy_parallel(n, a, x, y2, threads);
    t2 = omp_get_wtime();
    parallel_time = t2 - t1;

    /* ---- Metrics ---- */
    double speedup = serial_time / parallel_time;
    double efficiency = speedup / threads;

    /* ---- Output ---- */
    printf("\n===== DAXPY Performance =====\n");
    printf("Vector size     : %d\n", n);
    printf("Threads         : %d\n", threads);
    printf("Serial time     : %f sec\n", serial_time);
    printf("Parallel time   : %f sec\n", parallel_time);
    printf("Speedup         : %.3fx\n", speedup);
    printf("Efficiency      : %.3f (%.1f%%)\n", efficiency, efficiency * 100);

    /* correctness check */
    for (int i = 0; i < n; i++)
    {
        if (y1[i] != y2[i])
        {
            printf("Mismatch at %d\n", i);
            break;
        }
    }

    free(x);
    free(y1);
    free(y2);

    return 0;
}
