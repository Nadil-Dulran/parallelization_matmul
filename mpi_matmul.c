#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

static void fill_test(float *A, float *B, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[(size_t)i * N + j] = (float)((i + j) % 7 + 1);   // 1..7 pattern
            B[(size_t)i * N + j] = (i == j) ? 1.0f : 0.0f;     // identity
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s N [--print]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const int N = atoi(argv[1]);
    const int do_print = (argc >= 3 && strcmp(argv[2], "--print") == 0);

    if (N <= 0) {
        if (rank == 0)
            fprintf(stderr, "N must be positive.\n");
        MPI_Finalize();
        return 1;
    }

    // For simplicity: require N divisible by number of processes.
    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N (%d) must be divisible by number of processes (%d).\n",
                    N, size);
        MPI_Finalize();
        return 1;
    }

    const int rows_per_proc = N / size;
    const size_t mat_bytes  = (size_t)N * (size_t)N * sizeof(float);
    const size_t block_bytes = (size_t)rows_per_proc * (size_t)N * sizeof(float);

    float *A  = NULL;   // full A (only rank 0 uses this)
    float *B  = NULL;   // full B (all ranks need this)
    float *C  = NULL;   // full C (only rank 0 gathers here)

    float *local_A = (float*)malloc(block_bytes); // rows_per_proc x N
    float *local_C = (float*)malloc(block_bytes); // rows_per_proc x N

    if (!local_A || !local_C) {
        fprintf(stderr, "Rank %d: Allocation failed for local buffers.\n", rank);
        free(local_A); free(local_C);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        A = (float*)malloc(mat_bytes);
        B = (float*)malloc(mat_bytes);
        C = (float*)malloc(mat_bytes);
        if (!A || !B || !C) {
            fprintf(stderr, "Rank 0: Allocation failed for full matrices.\n");
            free(A); free(B); free(C);
            free(local_A); free(local_C);
            MPI_Finalize();
            return 1;
        }
        fill_test(A, B, N);
    } else {
        B = (float*)malloc(mat_bytes); // non-root ranks still need B for computation
        if (!B) {
            fprintf(stderr, "Rank %d: Allocation failed for B.\n", rank);
            free(local_A); free(local_C);
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast full matrix B to all processes
    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter rows of A to each process
    MPI_Scatter(A, rows_per_proc * N, MPI_FLOAT,
                local_A, rows_per_proc * N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Local matrix multiplication: each process computes its rows of C
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += local_A[(size_t)i * N + k] * B[(size_t)k * N + j];
            }
            local_C[(size_t)i * N + j] = sum;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time;

    // Use max time across ranks as the parallel execution time
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather local C blocks back to root
    MPI_Gather(local_C, rows_per_proc * N, MPI_FLOAT,
               C,        rows_per_proc * N, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("N=%d, procs=%d, time=%.6f s\n", N, size, max_time);
    }

    if (rank == 0 && do_print && N <= 8) {
        print_mat("A", A, N);
        print_mat("B", B, N);
        print_mat("C = A*B", C, N);
        }

    free(local_A);
    free(local_C);
    free(B);
    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}