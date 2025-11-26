#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_seconds(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_test(float *A, float *B, int N) {
    // Deterministic values (not random): handy for reproducibility & checks
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*(size_t)N + j] = (float)((i + j) % 7 + 1);    // 1..7 pattern
            B[i*(size_t)N + j] = (float)((i == j) ? 1 : 0);   // identity by default
        }
    }
}

static void print_mat(const char *name, const float *M, int N) {
    printf("%s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) printf("%6.1f ", M[i*(size_t)N + j]);
        printf("\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s N [--print]\n", argv[0]);
        return 1;
    }
    const int N = atoi(argv[1]);
    const int do_print = (argc >= 3 && strcmp(argv[2], "--print") == 0);

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)calloc((size_t)N * (size_t)N, sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "Allocation failed\n"); return 1; }

    fill_test(A, B, N);

    // Time the pure O(N^3) triple loop (row-major, no blocking)
    double t0 = now_seconds();
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float aik = A[i*(size_t)N + k];
            for (int j = 0; j < N; ++j) {
                C[i*(size_t)N + j] += aik * B[k*(size_t)N + j];
            }
        }
    }
    double t1 = now_seconds();

    // Lightweight correctness probe: when B is identity, C should equal A
    // We compare a few entries and print an aggregate absolute difference.
    double probe = 0.0;
    int p1 = 0, p2 = (N>1? N/2 : 0), p3 = (N>0? N-1 : 0);
    if (N > 0) {
        probe += fabs((double)C[p1*(size_t)N + p1] - (double)A[p1*(size_t)N + p1]);
        probe += fabs((double)C[p2*(size_t)N + p2] - (double)A[p2*(size_t)N + p2]);
        probe += fabs((double)C[p3*(size_t)N + p3] - (double)A[p3*(size_t)N + p3]);
    }

    printf("N=%d  time=%.6f s  probe_abs_diff=%.3g\n", N, t1 - t0, probe);

    if (do_print && N <= 8) {
        print_mat("A", A, N);
        print_mat("B", B, N);
        print_mat("C = A*B", C, N);
    }

    free(A); free(B); free(C);
    return 0;
}
