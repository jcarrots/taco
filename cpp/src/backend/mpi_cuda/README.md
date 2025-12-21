MPI+CUDA backend
================

Planned work:
- One MPI rank per GPU; use CUDA for on-node kernels and NCCL for collectives if needed.
- Device selection via rank and Exec.gpu_id.
- Controlled by Exec{backend=MpiCuda, gpu_id, streams, pinned}.
