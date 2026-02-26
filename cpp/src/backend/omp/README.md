OMP backend
===========

Status
------
- Source: `cpp/src/backend/omp/tcl4_omp.cpp`
- Header: `cpp/include/taco/backend/omp/tcl4_omp.hpp`
- `compute_triple_kernels` uses OpenMP across outer (i,j) loops when `Exec.backend=Omp`.
- `build_correction_series` dispatches to the OMP backend orchestration when `Exec.backend=Omp`.
- `build_mikx_omp` and `assemble_liouvillian` are still implemented in `cpp/src/tcl/` and reused by this backend.
- Threads controlled via `Exec.threads` (0 uses the runtime default).

Notes
-----
- MIKX and assembly are column-major; consider block scheduling for cache tuning.
