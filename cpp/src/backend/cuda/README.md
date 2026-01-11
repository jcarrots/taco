CUDA backend
===========

Status
------
- Implements scalar (1x1) TCL4 F/C/R series with batched cuFFT and CUB prefix scans.
- Entry points: `compute_triple_kernels_cuda` (F/C/R), `build_mikx_cuda` (M/I/K/X at one time index),
  and fused L4 builders in `tcl4_fused_cuda` (single time or batched series).

Usage
-----
- Build with `-DTACO_WITH_CUDA=ON` (defines `TACO_HAS_CUDA`).
- Select with `Exec{backend=Backend::Cuda, gpu_id, streams, pinned}`.
  - Current implementation uses a single stream and honors `gpu_id` and `pinned`.
- Fused L4 helpers:
  - `build_TCL4_generator_cuda_fused(...)` for a single time index.
  - `build_TCL4_generator_cuda_fused_batch(...)` for multiple time indices in one call.

Notes and limitations
---------------------
- `compute_triple_kernels_cuda` supports `FCRMethod::Convolution` only.
- The fused path keeps F/C/R on device, runs MIKX + GW + L4 on device, and copies all L4 outputs
  back in one D2H transfer (batch mode).
- `build_mikx_cuda` is a standalone helper; `build_mikx` remains CPU by default.

Recent optimizations
--------------------
- Reuse a persistent CUDA stream + FCR workspace (cuFFT plan + scratch) across calls.
- Single-time extraction uses `cudaMemcpy2DAsync` to gather a time slice from `[batch, Nt]`.
- Batch extraction uses a tiled transpose kernel when all times are requested.
- GW->L4 reshuffle uses a tiled, division-free kernel.
- CUDA Graphs: capture/replay the fixed MIKX -> GW -> L4 launch sequence to reduce host-side launch overhead.
  - Disable with `TCL4_USE_CUDA_GRAPH=0`.
  - Diagnostics with `TCL4_CUDA_GRAPH_VERBOSE=1`.

Performance
-----------
- GPU wins show up only when N is large enough to amortize launch/transfer overhead; small N tends to favor CPU.
- FFT sizes round up to the next power of two, so timings jump at those boundaries.
- CUDA Graphs help when the per-time MIKX/GW/L4 stage is dominated by kernel-launch overhead (many small time steps).
  - Repro (PowerShell, Release build):
    - Note: the compare tool uses `--threads=...` (plural).
    - Graphs off:
      - `$env:TCL4_USE_CUDA_GRAPH=0; $env:TCL4_CUDA_GRAPH_VERBOSE=0; .\build-cuda-vs\Release\tcl4_e2e_cuda_compare.exe --N=200000 --tidx=0:1:10000 --gpu_warmup=1 --threads=8`
    - Graphs on (capture+replay, prints once):
      - `$env:TCL4_USE_CUDA_GRAPH=1; $env:TCL4_CUDA_GRAPH_VERBOSE=1; .\build-cuda-vs\Release\tcl4_e2e_cuda_compare.exe --N=200000 --tidx=0:1:10000 --gpu_warmup=1 --threads=8`
  - Example result (RTX 3070, CUDA 12.5, Win11, Release):
    - Graphs off: `gpu_total_ms=417.3` (`gpu_avg_ms=0.0417`)
    - Graphs on:  `gpu_total_ms=358.7` (`gpu_avg_ms=0.0359`)

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
- `cpp/src/backend/cuda/tcl4_fused_cuda.cu`
- Headers: `cpp/include/taco/backend/cuda/*.hpp`
