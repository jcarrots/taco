# Parallelization Plan for TCL4 (CPU/GPU/Hybrid)

Goal
- Add parallel backends to accelerate TCL4 without changing numerics or public call sites. Keep TCL2 and TCL4 modules orthogonal; compose at the application layer.

Design Principles
- Preserve double precision and column‑major vec/unvec semantics.
- Operate in frequency (bucket) space; rebuild to full (j,k,p,q,r,s) only when required.
- Introduce a lightweight backend abstraction: CPU (OpenMP/TBB), GPU (cuFFT/cuBLAS), Hybrid (overlap GPU compute and CPU assembly).
- Stream/batch to keep memory bounded — avoid nf^3 and N^6 materialization.

API Surface (non‑breaking)
- Add a small execution descriptor in a new header:
  - `enum class Backend { CPU, GPU, Hybrid }`
  - `struct Exec { Backend backend=CPU; int threads; int gpu_id; int streams=2; bool pinned=true; }`
- Overloads with defaulted `Exec` to keep existing code compiling:
  - `compute_triple_kernels(..., method, Exec exec={})`
  - `build_TCL4_generator(..., method, Exec exec={})`
  - `build_correction_series(..., method, Exec exec={})`
- Drivers/tests accept `--backend=cpu|gpu|hybrid`, `--threads`, `--gpu_id`, `--streams`.

Work Packages

Phase 0 — Scaffolding (this PR)
- Add `cpp/include/taco/exec.hpp` with `Backend` and `Exec` (header‑only, no call‑site changes).
- Create backend folders:
  - `cpp/src/backend/cpu/` (README)
  - `cpp/src/backend/gpu/` (README)
  - `cpp/src/backend/hybrid/` (README)
- Documentation: this plan + structure updates.

Phase 1 — CPU Parallel
- F/C/R: parallelize outer `(i,j,k)` loops with OpenMP/TBB; keep time‑major inner loops sequential (prefix/FFT).
- MIKX/assemble: col‑block partitioning for GW; per‑thread blocks to avoid write contention; iterate in column‑major order.
- Benchmark and verify equality vs baseline.

Phase 2 — GPU Scalar Kernels
- Implement scalar (1×1) convolution path on GPU:
  - Batched FFTs for `causal_conv_fft`.
  - Kernels for phase multiply, prefix scan, elementwise ops.
- Keep MIKX/assemble on CPU initially; compare vs GPU compute + CPU assemble.

Phase 3 — Hybrid Streaming
- Overlap GPU F/C/R computation with CPU MIKX/assembly using streams and pinned host buffers.
- Batch by triple slabs or time windows; tune for throughput and memory.

Phase 4 — Optional GPU MIKX/Assemble
- Only if profiling shows clear wins; otherwise keep on CPU.
- Implement 2D/3D kernels to accumulate T, respecting column‑major mapping.

Numerics & Testing
- Use complex<double> throughout.
- Tests:
  - Direct vs Convolution (existing): ensure unchanged.
  - CPU vs GPU: small problems, bitwise/relative checks (1e‑12–1e‑10).
  - End‑to‑end spin‑boson TCL4: ⟨σ_z⟩ agreement within tolerance.

Build Flags
- CMake options (future): `TACO_WITH_OPENMP`, `TACO_WITH_CUDA` (or HIP/SYCL variants).
- Detect cuFFT/hipFFT; fallback to CPU FFT.

Risks & Mitigations
- Memory: compute‑at‑tidx/windowed series to avoid nf^3 blow‑up.
- Transfer overhead: pinned buffers + overlap.
- Small N performance: prefer CPU for MIKX/assemble unless profiling suggests otherwise.

Deliverables Summary
- Header: `cpp/include/taco/exec.hpp`.
- Backends: `cpp/src/backend/{cpu,gpu,hybrid}/README.md`.
- Overloads for TCL4 entry points (subsequent phases; default to CPU path).
