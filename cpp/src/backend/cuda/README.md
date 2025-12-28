CUDA backend
===========

Planned work:
- Scalar F/C/R kernels using cuFFT/hipFFT (batched) + simple kernels for phases/prefix.
- (Optional) pagewise GEMM for matrix kernels.
- Overlap GPU triple-kernel compute with CPU MIKX/assembly using streams and pinned transfers.
- Sliding windows in time or triple slabs to bound memory.
- Controlled by Exec{backend=Cuda, gpu_id, streams, pinned}.
