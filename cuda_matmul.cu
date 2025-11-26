#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(1);                                                \
        }                                                           \
    } while (0)

static void fill_test(float *A, float *B, int N) {
    // A: pattern 1..7, B: identity 
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
        for (int j = 0; j < N; ++j)
            printf("%6.1f ", M[(size_t)i * N + j]);
        printf("\n");
    }
}

// Each thread computes one C[row, col]
__global__ void matmul_kernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[(size_t)row * N + k] * B[(size_t)k * N + col];
    }
    C[(size_t)row * N + col] = sum;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s N blockDimX [blockDimY] [--print]\n", argv[0]);
        return 1;
    }

    const int N          = atoi(argv[1]);
    const int blockDimX  = atoi(argv[2]);
    const int blockDimY  = (argc >= 4 && argv[3][0] != '-') ? atoi(argv[3]) : blockDimX;
    const bool do_print  =
        (argc >= 4 && strcmp(argv[argc-1], "--print") == 0);

    if (N <= 0 || blockDimX <= 0 || blockDimY <= 0) {
        fprintf(stderr, "N, blockDimX, blockDimY must be positive.\n");
        return 1;
    }

    if (blockDimX * blockDimY > 1024) {
        fprintf(stderr, "Error: blockDimX * blockDimY must be <= 1024 (hardware limit).\n");
        return 1;
    }

    const size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    // Host memory
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    if (!hA || !hB || !hC) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    fill_test(hA, hB, N);

    // Device memory
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // Configure grid and block
    dim3 block(blockDimX, blockDimY);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_kernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // ms -> seconds
    double time_s = ms / 1000.0;

    printf("N=%d, blockDim=(%d,%d), grid=(%d,%d), time=%.6f S \n", N, blockDimX, blockDimY, grid.x, grid.y, time_s);

    if (do_print && N <= 8) {
        print_mat("A", hA, N);
        print_mat("B", hB, N);
        print_mat("C = A*B", hC, N);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);

    return 0;
}