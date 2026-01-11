# TCL4 Integration Plan

This document captures the steps required to translate the MATLAB TCL4 workflow (`tcl4_driver.m`, `tcl4_kernels.m`, `MIKX.m`, `NAKZWAN_v9.m`) into the C++ taco runtime. The plan tracks implementation phases, data shapes, and intended deliverables so that future contributors can resume or review the work easily.

Goal: produce a reusable C++ builder that constructs the fourth-order TCL Liouvillian (and supporting time-series data) given a system Hamiltonian, coupling operators, and bath correlation.

## Phase 0 - Setup and Data Model

- Inputs: system Hamiltonian `H`, coupling operators `{A_alpha}` in lab basis, bath correlation `C(t)` (or spectral density `J(omega)` + beta), integration step `dt`, time horizon `T`.
- Use `taco::sys::System` to diagonalize `H`, obtain eigenbasis `{V, epsilon}`, and build unique Bohr-frequency buckets:
  - `map.freq_to_pair`: list of `(m,n)` for each unique frequency omega
  - `map.pair_to_freq`: `N x N` matrix mapping (m,n) to bucket index
  - `map.time_grid`: time grid (vector), `map.omegas`: unique omega list
  - `map.mirror_index`: bucket index for `-omega` (self for omega ~ 0)
  - `map.zero_index`: bucket index for omega ~ 0
- Compute Gamma(omega_b, t_k) for all buckets b and time samples k using `gamma::compute_trapz_prefix_multi_matrix`. Store as `Eigen::MatrixXcd gamma_series` (rows = time, cols = frequencies).
- `taco::tcl4::Tcl4Map` stores these fields plus `N`, `nf`, and lookup maps.

## Phase 1 - Kernel Time-Series (F, C, R)

### 1.1 Building Blocks
- Implement `taco::tcl4::compute_FCR_time_series` mirroring MATLAB `tcl4_kernels.m`:
  - `prefix_int_left(A, dt)` - left Riemann cumulative integral along time.
  - `time_matmul(A(t), B(t))` - page-wise matrix multiplication for each time slice.
  - `volterra_conv_matmul(F, G, dt)` - causal convolution via FFT (time-major) returning `int_0^t F(t-s) G(s) ds`.
  - `SpectralOp` covers {Identity, Transpose, Conjugate, Hermitian} for matrix paths; the vector path handles transpose by using mirrored frequency buckets.

### 1.2 Driver for All Triples (vector path)
- `compute_triple_kernels(system, gamma_series, dt, nmax, method, exec)` produces time-series arrays for all `(omega1, omega2, omega3)` combinations.
  - `TripleKernelSeries` stores `F/C/R` as `std::vector<std::vector<std::vector<Eigen::VectorXcd>>>` (time series per triple).
  - `F` uses the mirrored bucket for the frequency "transpose"; `C`/`R` use the original bucket.
- Results are kept in frequency space (unique buckets). Rebuild helpers project back to full frequency-index tensors:
  - `build_FCR_6d_at(map, kernels, t_idx, F,C,R)`
  - `build_FCR_6d_series(map, kernels, ...)`
  - `build_FCR_6d_final(map, kernels, ...)`

## Phase 1b - Convolution F/C/R (Fast Path)

- `taco::tcl4::FCRMethod { Convolution (default), Direct }`.
- `compute_FCR_time_series_convolution` uses FFT-based Volterra convolution and pagewise GEMM:
  - Precompute phase factors; use zero-padding (2x-4x) to reduce circular wrap.
  - Compare against the Direct path (max abs/rel errors, especially at edges).
- Method selection flows through `compute_FCR_time_series(..., method)` and `compute_triple_kernels(..., method)`.
- Status:
  - Scalar (vector) path uses the FFT-based convolution.
  - Matrix path uses the FFT-based convolution in `compute_FCR_time_series_convolution`.
  - Matrix split helpers `compute_F_series/compute_C_series/compute_R_series` still fall back to Direct (TODO).

## Phase 2 - M, I, K, X Assembly (MIKX)

Status: implemented.

- `build_mikx_serial(map, kernels, time_index)` mirrors MATLAB `MIKX.m` contractions.
- Inputs: `TripleKernelSeries kernels` where `F/C/R[f1][f2][f3]` is an `Eigen::VectorXcd` time series; `time_index` selects the sample.
- Outputs:
  - `M`, `I`, `K`: `Eigen::MatrixXcd` of size `N^2 x N^2` with row `(j,k)` and col `(p,q)` flattened using column-major mapping to match vec/unvec: `idx = row + col*N` (so `(j,k) -> j + k*N`, `(p,q) -> p + q*N`).
  - `X`: `std::vector<std::complex<double>>` of length `N^6` stored column-major over `(j,k,p,q,r,s)`.
- Contractions (direct index mapping to MATLAB formulas):
  - `M(j,k,p,q) = F[f(j,k), f(j,q), f(p,j)] - R[f(j,q), f(p,q), f(q,k)]`
  - `I(j,k,p,q) = F[f(j,k), f(q,p), f(k,q)]`
  - `K(j,k,p,q) = R[f(j,k), f(p,q), f(q,j)]`
  - `X(j,k,p,q,r,s) = C[f(j,k), f(p,q), f(r,s)] + R[f(j,k), f(p,q), f(r,s)]`
- Layout decisions:
  - Pair flatten `flat2(N,a,b) = a + b*N` (column-major) to match innermost loops.
  - Six-index flatten `flat6(N,j,k,p,q,r,s)` uses column-major (first index varies fastest).
  - Frequency lookup via `map.pair_to_freq(a,b)`; validated to be non-negative for all required pairs.
- Optional: `build_mikx_omp` (OpenMP) and `build_mikx_cuda` (GPU helper).

## Phase 3 - Liouvillian Assembly (NAKZWAN)

Status: implemented.

- `assemble_liouvillian(tensors, coupling_ops)` forms the TCL4 correction matrix `GW`:
  - Follows the nested loops in `NAKZWAN_v9.m`, summing over system indices and coupling channels.
  - Symmetrizes `T + T^H` and returns `GW` in NAKZWAN indexing (row=(n,i), col=(m,j)).
- `gw_to_liouvillian` reshuffles `GW` into the Liouvillian superoperator `L4` (row=(n,m), col=(i,j)).

## Phase 4 - Public API Integration

- `taco/tcl4.hpp` exposes:
  - `Tcl4Map`, `TripleKernelSeries`, `compute_triple_kernels(...)`.
  - High-level wrappers:
    - `build_TCL4_generator(system, gamma_series, dt, time_index, method)` -> L4 at a single time index.
    - `build_correction_series(system, gamma_series, dt, method)` -> L4 for all times.
  - Rebuild helpers (`build_gamma_matrix_at`, `build_FCR_6d_*`).
- Composition of TCL2 + TCL4 is handled at the application layer (e.g., `examples/tcl_driver.cpp` uses `simulation.order` = 0|2|4).

## Phase 5 - Validation & Testing

- `tests/tcl4_tests.cpp`: compares Direct vs Convolution F/C/R for the MATLAB reference scenario (H=sigma_x/2, A=sigma_z/2, parameters alpha, beta, nmax, dt).
- `tests/tcl4_h5_compare.cpp`: optional MATLAB HDF5 compare tool.
- Sanity tests: ensure `GW` is Hermitian in Liouville sense (`GW ~= GW^dagger`), confirm time zero behavior.

## Phase 6 - Performance Enhancements (post-correctness)

- Parallelize heavy loops:
  - Over `(omega1, omega2, omega3)` when filling `F/C/R`.
  - Over system indices `(n,i,m,j)` in `assemble_liouvillian`.
  - Use OpenMP or TBB as consistent with the rest of the project.
- Optimize time-series operations:
  - Replace manual loops with Eigen `pagemtimes` for pagewise GEMM when available.
  - Ensure FFT zero-padding (2x length) for smoother Volterra convolutions.
- Consider caching repeated contractions when multiple times are requested.

## Deliverables Summary

Headers
- `taco/exec.hpp`
- `taco/tcl4_kernels.hpp`
- `taco/tcl4_mikx.hpp`
- `taco/tcl4_assemble.hpp`
- `taco/tcl4.hpp`
- `taco/backend/cpu/tcl4_mpi_omp.hpp` (optional MPI+OpenMP TCL4 batch API)

Sources
- `cpp/src/tcl/tcl4.cpp`
- `cpp/src/tcl/tcl4_kernels.cpp`
- `cpp/src/tcl/tcl4_mikx.cpp`
- `cpp/src/tcl/tcl4_assemble.cpp`
- `cpp/src/backend/cpu/tcl4_mpi_omp.cpp` (optional MPI+OpenMP TCL4 batch implementation)

Tests & Examples
- `tests/tcl4_tests.cpp`
- `tests/tcl4_h5_compare.cpp`
- `tests/tcl4_mpi_omp_tests.cpp` (optional MPI+OpenMP smoke test)
- `examples/tcl4_bench.cpp`
- `examples/TCL4_spin_boson_example.cpp`

Status: phases 0-4 implemented; validation ongoing; performance work in progress; initial MPI+OpenMP TCL4 batch builder implemented (Exec-based dispatch still TODO).
