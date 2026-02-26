Backend scaffolding
===================

This folder hosts implementation files for execution backends.

- cpu/: CPU backend implementation (currently MPI+OpenMP TCL4 batch).
- serial/: single-thread CPU reference path.
- omp/: shared-memory CPU parallelism (OpenMP/TBB).
- cuda/: single-node GPU kernels (cuFFT today).
- mpi_omp/: distributed CPU (MPI + OpenMP) planning notes.
- mpi_cuda/: distributed GPU (MPI + CUDA) planning notes.

Implementation status:
- serial/omp/cuda contain working code.
- `cpu/` contains the current MPI+OpenMP TCL4 batch implementation (`TACO_WITH_MPI=ON`) used by `Backend::MpiOmp`.
- mpi_omp/mpi_cuda contain design notes/placeholders; distributed GPU (`Backend::MpiCuda`) is still TODO.
