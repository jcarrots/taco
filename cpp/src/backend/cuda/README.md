CUDA backend
===========

Status
------
- Implements scalar (1x1) TCL4 F/C/R series with batched cuFFT and CUB prefix scans.
- Entry points: `compute_triple_kernels_cuda` (F/C/R) and `build_mikx_cuda` (M/I/K/X at one time index).

Usage
-----
- Build with `-DTACO_WITH_CUDA=ON` (defines `TACO_HAS_CUDA`).
- Select with `Exec{backend=Backend::Cuda, gpu_id, streams, pinned}`.
  - Current implementation uses a single stream and honors `gpu_id` and `pinned`.

Notes and limitations
---------------------
- `compute_triple_kernels_cuda` supports `FCRMethod::Convolution` only.
- Results are copied back to host per (i,j) batch; no overlap or device-resident assembly yet.
- `build_mikx_cuda` is a standalone helper; `build_mikx` remains CPU by default.

Performance
-----------
- GPU wins show up only when N is large enough to amortize launch/transfer overhead; small N tends to favor CPU.
- FFT sizes round up to the next power of two, so timings jump at those boundaries.

Tuning
------
- F/C/R kernel tiling can be tuned via environment variables (read once per process):
  - `TACO_CUDA_FCR_BLOCK_T`: time tile size (threads in x; rounded to warp, max 1024).
  - `TACO_CUDA_FCR_BLOCK_B`: batch lanes per block (threads in y; clamped to batch and 1024/`BLOCK_T`).
- Optional one-shot timing: set `TACO_CUDA_FCR_PROFILE=1` to print F/C/R stage timings for the first batched call.

Key files
---------
- `cpp/src/backend/cuda/tcl4_kernels_cuda.cu`
- `cpp/src/backend/cuda/tcl4_fcr_kernels.cu`
- `cpp/src/backend/cuda/tcl4_mikx_cuda.cu`
- Headers: `cpp/include/taco/backend/cuda/*.hpp`
