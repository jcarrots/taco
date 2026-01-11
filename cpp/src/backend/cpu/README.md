CPU backend
===========

This folder hosts CPU-only backend implementations that don't fit the serial/omp reference paths.

## MPI + OpenMP (TCL4)
- Entry point: `taco/backend/cpu/tcl4_mpi_omp.hpp` (`build_TCL4_generator_cpu_mpi_omp_batch`).
- Parallelism:
  - MPI: partitions requested time indices across ranks (contiguous block decomposition).
  - OpenMP: parallelizes the per-time `MIKX -> GW -> L4` loop within each rank.
- Output semantics: rank 0 gathers and returns all `L4(t)` in the same order as the input `time_indices`;
  non-root ranks return an empty vector.
- Build: requires `TACO_WITH_MPI=ON`.

Related:
- Test: `tests/tcl4_mpi_omp_tests.cpp` (run with `mpiexec -n <ranks> ...` when MPI is enabled).

