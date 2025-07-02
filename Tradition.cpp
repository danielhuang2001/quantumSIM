#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 32        // dimension of each square matrix
#define M 15        // number of matrices to multiply

int main(void) {
    // 1. Open the input file containing M N¡ÑN matrices
    FILE *fin = fopen("BV-5.txt", "r");
    if (!fin) {
        perror("cannot open input file");
        return EXIT_FAILURE;
    }

    // 2. Read all M dense N¡ÑN matrices into mats[m][i][j]
    static double mats[M][N][N];
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (fscanf(fin, "%lf", &mats[m][i][j]) != 1) {
                    fprintf(stderr,
                            "Error reading matrix %d element (%d,%d)\n",
                            m, i, j);
                    fclose(fin);
                    return EXIT_FAILURE;
                }
            }
        }
    }
    fclose(fin);

    // 3. Allocate result and tmp arrays for dense multiplication
    static double result[N][N];
    static double tmp[N][N];

    // 4. Initialize result with the first matrix (mats[0])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = mats[0][i][j];
        }
    }

    // 5. Multiply the remaining M-1 matrices into result,
    //    timing the computation and counting zero elements
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int m = 1; m < M; ++m) {
        // 5.1 Compute result ¡Ñ mats[m] ¡÷ tmp
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += result[i][k] * mats[m][k][j];
                }
                tmp[i][j] = sum;
            }
        }

        // 5.2 Copy tmp back into result for next iteration
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                result[i][j] = tmp[i][j];
            }
        }

        // 5.3 Count zero elements in the current result matrix
        int zero_count = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (result[i][j] == 0.0) {
                    ++zero_count;
                }
            }
        }
        printf("Stage %2d: zero elements = %4d / %4d\n",
               m, zero_count, N * N);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("\nTotal elapsed (dense): %.9f seconds\n\n", elapsed);

    // 6. Print the final result matrix
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

