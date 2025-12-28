Backend scaffolding
===================

This folder will host implementation files for execution backends.

- serial/: single-thread CPU reference path.
- omp/: shared-memory CPU parallelism (OpenMP/TBB).
- cuda/: single-node GPU kernels (cuFFT/cuBLAS/hipFFT/rocBLAS).
- mpi_omp/: distributed CPU (MPI + OpenMP).
- mpi_cuda/: distributed GPU (MPI + CUDA).

No code here yet - this is a placeholder to stage work without impacting the core.
