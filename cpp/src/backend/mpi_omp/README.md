MPI+OMP backend
===============

Planned work:
- MPI decomposition across nodes with OpenMP within each rank.
- One rank per CPU socket or NUMA domain; threads controlled via Exec.threads.
- Controlled by Exec{backend=MpiOmp, threads}.
