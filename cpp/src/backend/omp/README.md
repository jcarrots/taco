OMP backend
===========

Planned work:
- OpenMP/TBB parallelization of TCL4 triple-kernel driver (outer (i,j,k) loops).
- Column-block partitioning for MIKX/assemble to avoid write contention.
- Unit/perf tests and toggles via Exec{backend=Omp, threads}.
