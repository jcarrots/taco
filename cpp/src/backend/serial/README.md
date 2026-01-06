Serial backend
==============

Status
------
- CPU reference path for deterministic debugging.
- `compute_triple_kernels` runs single-threaded when `Exec.backend=Serial`.
- `build_mikx_serial` and `assemble_liouvillian` provide the single-thread baseline.
