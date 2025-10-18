GPU backend
===========

Planned work:
- Scalar F/C/R kernels using cuFFT/hipFFT (batched) + simple kernels for phases/prefix.
- (Optional) pagewise GEMM for matrix kernels.
- Pinned buffers + streams for overlap.
- Controlled by Exec{backend=GPU, gpu_id, streams}.

