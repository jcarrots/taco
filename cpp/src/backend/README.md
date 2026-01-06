Backend scaffolding
===================

This folder hosts implementation files for execution backends.

- serial/: single-thread CPU reference path.
- omp/: shared-memory CPU parallelism (OpenMP/TBB).
- cuda/: single-node GPU kernels (cuFFT today).
- mpi_omp/: distributed CPU (MPI + OpenMP) planning notes.
- mpi_cuda/: distributed GPU (MPI + CUDA) planning notes.

Implementation status:
- serial/omp/cuda contain working code.
- mpi_omp/mpi_cuda are placeholders with design notes only.
