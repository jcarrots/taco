# Parallelization Plan for TCL4 (serial/omp/cuda/mpi_omp/mpi_cuda)

Goal
- Add parallel backends to accelerate TCL4 without changing numerics or public call sites. Keep TCL2 and TCL4 modules orthogonal; compose at the application layer.

Design Principles
- Preserve double precision and column-major vec/unvec semantics.
- Operate in frequency (bucket) space; rebuild to full (j,k,p,q,r,s) only when required.
- Introduce a lightweight backend abstraction: serial, omp, cuda, mpi_omp, mpi_cuda.
- Stream/batch to keep memory bounded - avoid nf^3 and N^6 materialization.

API Surface (non-breaking)
- Add a small execution descriptor in a new header:
  - `enum class Backend { Serial, Omp, Cuda, MpiOmp, MpiCuda }`
  - `struct Exec { Backend backend=Omp; int threads; int gpu_id; int streams=2; bool pinned=true; }`
- Overloads with defaulted `Exec` to keep existing code compiling:
  - `compute_triple_kernels(..., method, Exec exec={})`
  - `build_TCL4_generator(..., method, Exec exec={})`
  - `build_correction_series(..., method, Exec exec={})`
- Drivers/tests accept `--backend=serial|omp|cuda|mpi_omp|mpi_cuda`, `--threads`, `--gpu_id`, `--streams`.

Work Packages

Phase 0 - Scaffolding (done)
- `cpp/include/taco/exec.hpp` defines `Backend` and `Exec`.
- Backend folders exist under `cpp/src/backend/`.
- Documentation: this plan + structure updates.

Phase 1 - OMP (shared-memory CPU)
- `compute_triple_kernels` uses OpenMP for outer (i,j) loops when `Exec.backend=Omp`.
- `build_mikx_omp` and `assemble_liouvillian` use OpenMP when available.
- TODO: tune scheduling/blocking and add perf tests.

Phase 2 - CUDA (single-node)
- Implemented scalar (1x1) convolution path on GPU:
  - Batched FFTs for `causal_conv_fft`.
  - Kernels for phase multiply, prefix scan, elementwise ops.
- Entry point: `compute_triple_kernels_cuda` (supports `FCRMethod::Convolution`).
- `build_mikx_cuda` is available as a helper; assembly remains on CPU.

Phase 3 - CUDA overlap
- TODO: overlap GPU F/C/R with CPU MIKX/assembly using streams and pinned host buffers.
- Batch by triple slabs or time windows; tune for throughput and memory.

Phase 4 - MPI backends
- `mpi_omp`: distributed CPU (MPI + OpenMP).
- `mpi_cuda`: distributed GPU (MPI + CUDA), one rank per GPU.

Phase 5 - Optional GPU MIKX/Assemble
- Only if profiling shows clear wins; otherwise keep on CPU.
- Implement 2D/3D kernels to accumulate T, respecting column-major mapping.

Numerics & Testing
- Use complex<double> throughout.
- Tests:
  - Direct vs Convolution (existing): ensure unchanged.
  - CPU vs GPU: small problems, bitwise/relative checks (1e-12 to 1e-10).
  - End-to-end spin-boson TCL4: <sigma_z> agreement within tolerance.

Build Flags
- CMake options (future): `TACO_WITH_OPENMP`, `TACO_WITH_CUDA` (or HIP/SYCL variants).
- Detect cuFFT/hipFFT; fallback to CPU FFT.

Risks & Mitigations
- Memory: compute-at-tidx/windowed series to avoid nf^3 blow-up.
- Transfer overhead: pinned buffers + overlap.
- Small N performance: prefer CPU for MIKX/assemble unless profiling suggests otherwise.

Deliverables Summary
- Header: `cpp/include/taco/exec.hpp`.
- Backends: `cpp/src/backend/{serial,omp,cuda,mpi_omp,mpi_cuda}/README.md`.
- Overloads for TCL4 entry points (default to Omp path).
