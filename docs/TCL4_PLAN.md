# TCL4 Integration Plan

This document captures the steps required to translate the MATLAB TCL4 workflow (`tcl4_driver.m`, `tcl4_kernels.m`, `MIKX.m`, `NAKZWAN_v9.m`) into the C++ taco runtime. The plan tracks implementation phases, data shapes, and intended deliverables so that future contributors can resume or review the work easily.

**Goal**: produce a reusable C++ builder that constructs the fourth–order TCL Liouvillian (and supporting time-series data) given a system Hamiltonian, coupling operators, and bath correlation.

## Phase 0 – Setup and Data Model

- Inputs: system Hamiltonian `H`, coupling operators `{Aα}` in lab basis, bath correlation `C(t)` (or spectral density `J(ω)` + β), integration step `dt`, time horizon `T`.
- Use `taco::sys::System` to diagonalize `H`, obtain eigenbasis `{V, ε}`, and build unique Bohr-frequency buckets:
  - `map.freq_to_pair`: list of `(m,n)` for each unique frequency ω
  - `map.pair_to_freq`: `N×N` matrix mapping (m,n) to bucket index
  - `map.t`: time grid (vector), `map.omegas`: unique ω list.
- Compute Γ(ω_b, t_k) for all buckets b and time samples k using `gamma::compute_trapz_prefix_multi_matrix`. Store as `Eigen::MatrixXcd GammaSeries` (rows = time, columns = frequencies).
- Define `struct Tcl4Map { int N; int nf; std::vector<double> t; Eigen::MatrixXi pair_to_freq; std::vector<std::pair<int,int>> freq_to_pair; };`

## Phase 1 – Kernel Time-Series (F, C, R)

### 1.1 Building Blocks
- Implement `taco::tcl4::compute_FCR_timeseries` mirroring MATLAB `tcl4_kernels.m`:
  - `prefix_int_left(A, dt)` – left Riemann cumulative integral along time.
  - `time_matmul(A(t), B(t))` – page-wise matrix multiplication for each time slice.
  - `volterra_conv_matmul(F, G, dt)` – causal convolution via FFT (time-major) returning `∫₀^t F(t−s)G(s) ds`.
  - Support operations on `G2`: op ∈ {identity, transpose, conjugation, Hermitian}; mimic MATLAB’s `apply_op_g2`.

### 1.2 Driver for All Triples (vector path)
- Implement `compute_FCR_timeseries_all(system, GammaSeries, dt, nmax)` producing time-series arrays for all `(ω1, ω2, ω3)` combinations. Shapes:
  - `F_all`, `C_all`, `R_all` – stored as `std::vector<Eigen::MatrixXcd>` or custom tensor wrapper (`time × nf × nf × nf`).
  - Use `map` mappings to fill data in the same layout as MATLAB (`map.ij` style conversions).
- For each time index `k` (matching `tcl4_driver` logic: `tidx = 2*nmax*ns + 1`), later reshape these into 6D tensors for system indices.
  - Current implementation computes F/C/R using Γ columns (`Eigen::VectorXcd`) with `FCRMethod::Convolution` by default; Γ^T is realized by using the mirrored frequency bucket.
  - Results are kept in frequency space (unique buckets). Dedicated rebuild helpers project back to full frequency‑index tensors:
    - `build_FCR_6d_at(map, kernels, t_idx, F,C,R)` and `build_FCR_6d_series(...)`.

## Phase 1b – Convolution F/C/R (Fast Path)

- Introduce `taco::tcl4::FCRMethod { Convolution (default), Direct }`.
- Implement `compute_FCR_time_series_convolution` using FFT‑based Volterra convolution and pagewise GEMM:
  - Precompute phase factors; use zero‑padding (2×–4×) to reduce circular wrap.
  - Overlap‑add or sufficiently long grids to mimic causal convolution.
  - Compare against the Direct path (max abs/rel errors, especially at edges).
- Wire method selection through `compute_FCR_time_series(..., method)` and `compute_triple_kernels(..., method)`.
- Keep default = `Convolution` so call sites remain stable as we optimize.
- Status: scalar (1×1) path implemented with FFT-based Volterra convolution; matrix-valued FFT path still TODO.

## Phase 2 – M, I, K, X Assembly (MIKX)

Status: Implemented

- Implemented `taco::tcl4::build_mikx(map, kernels, time_index)` to mirror MATLAB `MIKX.m` contractions.
- Inputs: `TripleKernelSeries kernels` where `F/C/R[f1][f2][f3]` is an `Eigen::VectorXcd` time series; `time_index` selects the sample.
- Outputs:
- `M`, `I`, `K`: `Eigen::MatrixXcd` of size `N^2 × N^2` with row `(j,k)` and col `(p,q)` flattened using column‑major mapping to match vec/unvec: `idx = row + col*N` (so `(j,k)` → `j + k*N`, `(p,q)` → `p + q*N`).
  - `X`: `std::vector<std::complex<double>>` of length `N^6` stored row‑major over `(j,k,p,q,r,s)`.
- Contractions (direct index mapping to MATLAB formulas):
  - `M(j,k,p,q) = F[f(j,k), f(j,q), f(p,j)] − R[f(j,q), f(p,q), f(q,k)]`
  - `I(j,k,p,q) = F[f(j,k), f(q,p), f(k,q)]`
  - `K(j,k,p,q) = R[f(j,k), f(p,q), f(q,j)]`
  - `X(j,k,p,q,r,s) = C[f(j,k), f(p,q), f(r,s)] + R[f(j,k), f(p,q), f(r,s)]`
- Layout decisions:
  - Pair‑flatten `flat2(N,a,b) = a*N + b` (row‑major) to match innermost loops.
  - Six‑index flatten `flat6(N,j,k,p,q,r,s)` uses row‑major (last index varies fastest) for sequential writes across `s`.
  - Frequency lookup via `map.pair_to_freq(a,b)`; validated to be non‑negative for all required pairs.
- Future optimization: batch reshapes + GEMM to reduce scalar indexing overhead.

## Phase 3 – Liouvillian Assembly (NAKZWAN)

- Implement `taco::tcl4::assemble_liouvillian(M, I, K, X, coupling_ops)` to form the TCL4 correction matrix `GW`:
  - Follows the nested loops in `NAKZWAN_v9.m`, summing over system indices and coupling channels.
  - Compensate for `T + T'` symmetrization and reshape into Liouville space `N^2 × N^2`.
  - Provide intermediate interface returning `Eigen::MatrixXcd`.

## Phase 4 – Public API Integration

- Add `taco/tcl4.hpp` exposing:
  - `struct Tcl4Components { Eigen::MatrixXcd L_tcl4; /* plus optionally F,C,R,M,I,K,X logs */ }`
  - High‑level wrappers:
    - `build_TCL4_generator(system, gamma_series, dt, time_index, method)` → GW at time index.
    - `build_correction_series(system, gamma_series, dt, method)` → GW for all times.
    - Rebuild helpers as noted in Phase 1.2.
  - Optionally, helper overload to accept a `tcl::TCL2Generator` (stateful) to reuse Simpson integrals.
- Update `examples/spin_boson.cpp` CLI: new flag `--generator=tcl2|tcl4|both` to compute TCL4 correction and optionally add to TCL2 Liouvillian.
- Provide YAML knobs (`spectral`, `cutoff`, `generator`) already extended earlier.
 - Convenience rebuild helpers are available:
   - `build_gamma_matrix_at(map, gamma_series, t_idx)`
   - `build_FCR_6d_at(map, kernels, t_idx, F,C,R)` and `build_FCR_6d_final(...)`

## Phase 5 – Validation & Testing

- Create `tests/tcl4_tests.cpp` replicating the MATLAB `tcl4_driver` scenario (H=σx/2, A=σz/2, parameters α, β, nmax, dt).
  - Compare `GW(t)` at several `tidx` with MATLAB output saved as reference (tolerances ~1e-3).
  - Check that the assembled Liouvillian reduces to zero when couplings vanish.
- Sanity tests: ensure `GW` is Hermitian in Liouville sense (`GW ≈ GW†`), confirm time zero behavior matches expected zeros.

## Phase 6 – Performance Enhancements (post-correctness)

- Parallelize heavy loops:
  - Over `(ω1, ω2, ω3)` when filling `F_all`, `C_all`, `R_all`.
  - Over system indices `(n,i,m,j)` in `assemble_liouvillian`.
  - Use OpenMP or TBB as consistent with the rest of the project.
- Optimize time-series operations:
  - Replace manual loops with Eigen `pagemtimes` for pagewise GEMM when available.
  - Ensure FFT zero-padding (2× length) for smoother Volterra convolutions.
- Consider caching repeated contractions when multiple times are requested.

## Deliverables Summary

### Headers
- `taco/tcl4_kernels.hpp` – F/C/R builders
- `taco/tcl4_mikx.hpp` – M/I/K/X construction
- `taco/tcl4_assemble.hpp` – Liouvillian assembly (NAKZWAN)
- `taco/tcl4.hpp` – convenience API returning `Tcl4Components`

### Sources
- `cpp/src/tcl/tcl4_kernels.cpp`
- `cpp/src/tcl/tcl4_mikx.cpp`
- `cpp/src/tcl/tcl4_assemble.cpp`
- (optional) `cpp/src/tcl/tcl4_components.cpp`

### Tests & Examples
- New TCL4 unit test(s): `tests/tcl4_tests.cpp`
- Modified spin-boson CLI supporting `--generator=tcl4` and/or `--generator=both`
- Documentation update for finite CLI, module references (`DEV_GUIDE.md`) and structure file (`docs/STRUCTURE.md`)

---
_Status_: planning complete. Implementation can now proceed Phase by Phase, starting with kernel builders.
