## Matrix multiply implementations — compile & run

Contains multiple implementations of a square matrix multiplication (A * B = C)

- `serial_matmul.c` — single-threaded reference
- `omp_matmul.c` — OpenMP parallel version
- `mpi_matmul.c` — MPI distributed-memory version
- `cuda_matmul.cu` — CUDA GPU version 

This README shows how to compile and run each version on macOS or Linux and gives troubleshooting tips.

## Prerequisites

- C compiler (clang or gcc). On macOS, clang is default; for OpenMP prefer Homebrew `gcc` or install libomp.
- Open MPI or MPICH for MPI (`mpicc`, `mpirun` / `mpiexec`).
- CUDA Toolkit (nvcc) and an NVIDIA GPU to use the CUDA version. Note: CUDA is not supported on Apple Silicon Macs.

Useful package install commands (macOS with Homebrew):

```bash
# Update brew
brew update

# Install Open MPI
brew install open-mpi

# Install GCC (if you want GNU gcc with OpenMP support)
brew install gcc

# Install libomp (if you prefer clang + OpenMP on macOS)
brew install libomp

# On Linux use your package manager, e.g.:
# sudo apt update && sudo apt install build-essential libopenmpi-dev openmpi-bin
```

Check installs:

```bash
which mpicc    # should point to mpicc (open-mpi)
which mpirun
which gcc || which gcc-13
which nvcc     # for CUDA (Linux/Windows only, and only if you have an NVIDIA GPU)
```

## Compile & run

All examples assume your current working directory is this `Codes` folder.

Serial (single-threaded)

```bash
cc serial_matmul.c -O2 -o serial_matmul
./serial_matmul 1024
```

OpenMP (shared-memory)

Note: macOS's system clang does not ship with OpenMP support. Two common options:

1) Use Homebrew GCC (recommended on macOS):

```bash
# Homebrew gcc is often named gcc-13 or similar. Check `brew info gcc`.
gcc-13 -fopenmp -O2 omp_matmul.c -o omp_matmul
./omp_matmul 2048 16
```

2) Use clang + libomp (macOS):

```bash
# Install libomp via brew: brew install libomp
clang -Xpreprocessor -fopenmp omp_matmul.c -L/opt/homebrew/lib -lomp -I/opt/homebrew/include -O2 -o omp_matmul
./omp_matmul 2048 16
```

On Linux with GCC:

```bash
gcc -fopenmp -O2 omp_matmul.c -o omp_matmul
./omp_matmul 2048 16
```

MPI (distributed-memory)

```bash
mpicc mpi_matmul.c -O2 -o mpi_matmul
# Run with 4 processes (adjust -n / -np as needed)
mpirun -n 4 ./mpi_matmul 1024
```

Notes:
- `mpirun -n` or `mpirun -np` are both commonly accepted; some MPI installations prefer `mpiexec -n`.
- If `mpicc` can't find `mpi.h` in your editor (VS Code squiggles), ensure the C/C++ extension includePath includes the Open MPI include directory (e.g., `/opt/homebrew/include` or `/usr/local/include`).

CUDA (GPU)

Only compile and run this if you have an NVIDIA GPU and the CUDA toolkit installed. CUDA is typically not available on modern macOS machines with Apple Silicon.

```bash
# Example: compile a CUDA .cu file
nvcc -O2 -arch=sm_70 cuda_matmul.cu -o cuda_matmul
./cuda_matmul 2048
```

Pick the `-arch=sm_XX` value that matches your GPU's compute capability. Use `nvcc --list-gpus` or `nvidia-smi` to inspect your GPU.

## Common runtime examples

- Run OpenMP with 8 threads:

```bash
export OMP_NUM_THREADS=8
./omp_matmul 2048 8
```

- Run MPI with 8 processes (on a single machine):

```bash
mpirun -n 8 ./mpi_matmul 2048
```

## Troubleshooting

- mpi.h not found in VS Code (editor shows squiggle):
  - Ensure `c_cpp_properties.json` (in `.vscode/`) includes the MPI include path, e.g. `/opt/homebrew/include` or `/usr/local/include`.
  - Ensure `mpicc` and `mpirun` are installed (e.g., `brew install open-mpi`).

- OpenMP errors on macOS:
  - If using clang, install libomp (`brew install libomp`) and compile with `-Xpreprocessor -fopenmp -lomp -I... -L...`.
  - Alternatively use the GNU compiler from Homebrew (gcc-13 etc.) and `-fopenmp`.

- CUDA: "nvcc not found" or no GPU available:
  - Install CUDA toolkit and drivers on supported OS (Linux or Windows with NVIDIA GPU). New Apple Silicon Macs do not support CUDA.

## Editor / VS Code tips

- I added a `.vscode/c_cpp_properties.json` with common include paths (Homebrew `/opt/homebrew/include`, `/usr/local/include`) so the C/C++ extension can find `mpi.h` if Open MPI is installed via Homebrew. If your MPI headers are installed elsewhere, add that path.
- To build from VS Code, consider adding a `tasks.json` task that runs `mpicc` or `gcc -fopenmp`.

## Final notes

- Replace numbers (e.g., `1024`, `2048`, and thread/process counts) with values appropriate for your machine and test case.
- If you'd like, I can add a `tasks.json` to automate builds in VS Code and a sample `cuda_matmul.cu` harness if you don't yet have one.

---
