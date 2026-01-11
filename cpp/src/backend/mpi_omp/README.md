MPI+OMP backend
===============

Planned work:
- MPI decomposition across nodes with OpenMP within each rank.
- One rank per CPU socket or NUMA domain; threads controlled via Exec.threads.
- Controlled by Exec{backend=MpiOmp, threads}.

Current status:
- Initial MPI+OpenMP CPU implementation exists for TCL4 batched L4 construction:
  - Header: `cpp/include/taco/backend/cpu/tcl4_mpi_omp.hpp`
  - Source: `cpp/src/backend/cpu/tcl4_mpi_omp.cpp`
  - Build flag: `TACO_WITH_MPI=ON` (sets `TACO_HAS_MPI=1`)
- Exec-based dispatch (`Exec{backend=MpiOmp,...}`) is still TODO.
