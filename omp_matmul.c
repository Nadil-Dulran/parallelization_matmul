#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_test(float *A, float *B, int N) {
    // Deterministic values: A has a 1..7 pattern, B = identity
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[(size_t)i * N + j] = (float)((i + j) % 7 + 1);
            B[(size_t)i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

static void print_mat(const char *name, const float *M, int N) {
    printf("%s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%6.1f ", M[(size_t)i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s N num_threads [--print]\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
    const int num_threads = atoi(argv[2]);
    const int do_print = (argc >= 4 && strcmp(argv[3], "--print") == 0);

    if (N <= 0 || num_threads <= 0) {
        fprintf(stderr, "N and num_threads must be positive integers.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)calloc((size_t)N * (size_t)N, sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    fill_test(A, B, N);

    // Parallel matrix multiplication: C = A × B
    double t0 = now_seconds();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[(size_t)i * N + k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = sum;
        }
    }

    double t1 = now_seconds();
    double exec_time = t1 - t0;

    // Final Outputs
    printf("N=%d, threads=%d, time=%.6f s\n", N, num_threads, exec_time);

    if (do_print && N <= 8) {
        print_mat("A", A, N);
        print_mat("B", B, N);
        print_mat("C = A*B", C, N);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}