#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N        32               // Dimension of each square matrix
#define M        15               // Number of matrices to multiply
#define MAX_NNZ  (M * N * N)      // Maximum total nonzeros across all matrices

int main(void) {
    // --- 1. Open the input file containing M dense N¡ÑN matrices ---
    FILE *fin = fopen("BV-5.txt", "r");
    if (!fin) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    // --- 2. Read all M dense matrices into mats[m][i][j] ---
    static double mats[M][N][N];
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (fscanf(fin, "%lf", &mats[m][i][j]) != 1) {
                    fprintf(stderr,
                            "Read error at matrix %d, element (%d,%d)\n",
                            m, i, j);
                    fclose(fin);
                    return EXIT_FAILURE;
                }
            }
        }
    }
    fclose(fin);

    // --- 3. Convert each dense matrix into CSR format ---
    // row_ptr[m][r] = start index in vals/col_idx for row r of matrix m
    static int    row_ptr[M][N+1];
    static int    col_idx[MAX_NNZ];  // column indices of stored nonzero values
    static double vals[MAX_NNZ];     // stored nonzero values
    int offset = 0;                  // running total of nonzeros

    for (int m = 0; m < M; ++m) {
        row_ptr[m][0] = offset;

        // 3.1 Count nonzeros in each row to build row_ptr
        for (int r = 0; r < N; ++r) {
            int count_in_row = 0;
            for (int c = 0; c < N; ++c) {
                if (mats[m][r][c] != 0.0) {
                    ++count_in_row;
                }
            }
            row_ptr[m][r+1] = row_ptr[m][r] + count_in_row;
        }

        // 3.2 Populate col_idx and vals arrays for matrix m
        for (int r = 0; r < N; ++r) {
            int write_idx = row_ptr[m][r];
            for (int c = 0; c < N; ++c) {
                double v = mats[m][r][c];
                if (v != 0.0) {
                    vals[write_idx]    = v;
                    col_idx[write_idx] = c;
                    ++write_idx;
                }
            }
        }

        // Advance offset past this matrix's entries
        offset = row_ptr[m][N];
    }

    // --- 4. Initialize result (dense) with the first CSR matrix ---
    static double result[N][N], tmp[N][N];

    // Zero out result
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            result[i][j] = 0.0;

    // Scatter nonzeros of matrix 0 into result
    for (int r = 0; r < N; ++r) {
        for (int idx = row_ptr[0][r]; idx < row_ptr[0][r+1]; ++idx) {
            int c = col_idx[idx];
            result[r][c] = vals[idx];
        }
    }

    // --- 5. Multiply remaining CSR matrices into result ---
    long long mul_count = 0;         // total scalar multiplications
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int m = 1; m < M; ++m) {
        // 5.1 Zero out temporary accumulator
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                tmp[i][j] = 0.0;

        // 5.2 Perform sparse-dense multiplication:
        //     result (dense) ¡Ñ mats[m] (CSR) ¡÷ tmp (dense)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                double a = result[i][k];
                if (a == 0.0) continue;  // skip zeros in result
                // traverse nonzeros in row k of matrix m
                for (int idx = row_ptr[m][k]; idx < row_ptr[m][k+1]; ++idx) {
                    int c = col_idx[idx];
                    tmp[i][c] += a * vals[idx];
                    ++mul_count;
                }
            }
        }

        // 5.3 Copy tmp back into result for next iteration
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                result[i][j] = tmp[i][j];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    // --- 6. Output statistics and final result matrix ---
    printf("Compressed CSR: total scalar multiplications = %lld\n", mul_count);
    printf("Elapsed time = %.9f seconds\n\n", elapsed);

    printf("Final result matrix (%dx%d):\n", N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("% .6f%s",
                   result[i][j],
                   (j + 1 < N) ? " " : "\n");
        }
    }

    return EXIT_SUCCESS;
}

