#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <windows.h>
#include <psapi.h>



// =================================================
// Sequential Correlation
// =================================================
void correlate_seq(int ny, int nx, const float* data, float* result)
{
    std::vector<double> norm(ny * nx);

    for (int y = 0; y < ny; ++y)
    {
        double mean = 0.0;
        for (int x = 0; x < nx; ++x)
            mean += data[x + y * nx];
        mean /= nx;

        double sumsq = 0.0;
        for (int x = 0; x < nx; ++x)
        {
            double d = data[x + y * nx] - mean;
            sumsq += d * d;
        }

        double inv_std = (sumsq == 0.0) ? 0.0 : 1.0 / std::sqrt(sumsq);

        for (int x = 0; x < nx; ++x)
            norm[x + y * nx] = (data[x + y * nx] - mean) * inv_std;
    }

    for (int i = 0; i < ny; ++i)
        for (int j = 0; j <= i; ++j)
        {
            double dot = 0.0;
            for (int x = 0; x < nx; ++x)
                dot += norm[x + i * nx] * norm[x + j * nx];

            result[i + j * ny] = dot;
        }
}

// =================================================
// Parallel Correlation
// =================================================
void correlate_par(int ny, int nx, const float* data, float* result)
{
    std::vector<double> norm(ny * nx);

    // Normalize rows in parallel
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y)
    {
        const float* row = data + y * nx;
        double* norm_row = norm.data() + y * nx;

        double mean = 0.0;
        for (int x = 0; x < nx; ++x)
            mean += row[x];
        mean /= nx;

        double sumsq = 0.0;
        for (int x = 0; x < nx; ++x)
        {
            double d = row[x] - mean;
            sumsq += d * d;
        }

        double inv_std = (sumsq == 0.0) ? 0.0 : 1.0 / std::sqrt(sumsq);

        for (int x = 0; x < nx; ++x)
            norm_row[x] = (row[x] - mean) * inv_std;
    }

    int total_pairs = (ny * (ny + 1)) / 2;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int idx = 0; idx < total_pairs; ++idx)
    {
        int i = 0;
        int sum = 0;
        while (sum + i + 1 <= idx)
        {
            sum += i + 1;
            ++i;
        }
        int j = idx - sum;

        const double* norm_i = norm.data() + i * nx;
        const double* norm_j = norm.data() + j * nx;

        double dot = 0.0;
        for (int x = 0; x < nx; ++x)
            dot += norm_i[x] * norm_j[x];

        result[i + j * ny] = dot;
    }
}