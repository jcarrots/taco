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

### 1.2 Driver for All Triples
- Implement `compute_FCR_timeseries_all(system, GammaSeries, dt, nmax)` producing time-series arrays for all `(ω1, ω2, ω3)` combinations. Shapes:
  - `F_all`, `C_all`, `R_all` – stored as `std::vector<Eigen::MatrixXcd>` or custom tensor wrapper (`time × nf × nf × nf`).
  - Use `map` mappings to fill data in the same layout as MATLAB (`map.ij` style conversions).
- For each time index `k` (matching `tcl4_driver` logic: `tidx = 2*nmax*ns + 1`), later reshape these into 6D tensors for system indices.

## Phase 2 – M, I, K, X Assembly (MIKX)

- Implement `taco::tcl4::build_mikx(F_tensor, C_tensor, R_tensor, map)` matching MATLAB `MIKX.m`:
  - Input tensors `A2_1`, `A3_1`, `A4_1` (size `N^6` for system indices). v1 representation: `std::vector<std::complex<double>>` with indexing helpers or nested loops.
  - Output: `M(j,k,p,q)`, `I(j,k,p,q)`, `K(j,k,p,q)` (4-index) + `X(i,k,p,q,r,s)` (6-index).
  - Reproduce the identity tensor contractions performed in MATLAB `tensorprod`, `permute` using explicit loops.
  - Later optimization: restructure as series of reshapes + matrix multiplications.

## Phase 3 – Liouvillian Assembly (NAKZWAN)

- Implement `taco::tcl4::assemble_liouvillian(M, I, K, X, coupling_ops)` to form the TCL4 correction matrix `GW`:
  - Follows the nested loops in `NAKZWAN_v9.m`, summing over system indices and coupling channels.
  - Compensate for `T + T'` symmetrization and reshape into Liouville space `N^2 × N^2`.
  - Provide intermediate interface returning `Eigen::MatrixXcd`.

## Phase 4 – Public API Integration

- Add `taco/tcl4.hpp` exposing:
  - `struct Tcl4Components { Eigen::MatrixXcd L_tcl4; /* plus optionally F,C,R,M,I,K,X logs */ }`
  - `build_tcl4_components(const sys::System&, const Eigen::MatrixXcd& GammaSeries, double dt, std::size_t time_index, const Options&)`
    - Internally: Phase 1 (F,C,R for selected time), Phase 2 (MIKX), Phase 3 (assemble).
  - Optionally, helper overload to accept a `tcl::TCL2Generator` (stateful) to reuse Simpson integrals.
- Update `examples/spin_boson.cpp` CLI: new flag `--generator=tcl2|tcl4|both` to compute TCL4 correction and optionally add to TCL2 Liouvillian.
- Provide YAML knobs (`spectral`, `cutoff`, `generator`) already extended earlier.

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
