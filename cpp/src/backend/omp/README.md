OMP backend
===========

Status
------
- `compute_triple_kernels` uses OpenMP across outer (i,j) loops when `Exec.backend=Omp`.
- `build_mikx_omp` and `assemble_liouvillian` use OpenMP when available.
- Threads controlled via `Exec.threads` (0 uses the runtime default).

Notes
-----
- MIKX and assembly are column-major; consider block scheduling for cache tuning.
