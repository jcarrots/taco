Developer Guide
===============

This guide summarizes how to build, run, test, and tune the project’s core components (TCL2 runtime, FFT correlation, integration utilities).

Prereqs
-------
- CMake 3.20+
- A C++17 compiler (MSVC 2022 on Windows; Clang/GCC on Linux/macOS)
- Python 3.x (optional, for the pybind11 extension)

Quick Build
-----------
- Configure: `cmake -S . -B build`
- Build Debug: `cmake --build build --config Debug -j 8`
- Build Release: `cmake --build build --config Release -j 8`
- Run demo (Win): `build\Release\tcl2_demo.exe`

CMake Options
-------------
- `TACO_BUILD_PYTHON` (default ON): build the pybind11 extension module.
  - If CMake cannot find Python, pass `-DPython_EXECUTABLE=...` (and, on Windows, `-DPython_INCLUDE_DIR=...`, `-DPython_LIBRARY=...`).
- `TACO_BUILD_GAMMA_TESTS` (default ON): build `gamma_tests` (and only then look for Boost).

Python Extension
----------------
- Configure: `cmake -S . -B build -DTACO_BUILD_PYTHON=ON -DPython_EXECUTABLE=...`
- Build: `cmake --build build --config Release --target _taco_native`
- Import test: `python -c "import sys; sys.path.insert(0,'python'); import taco; print(taco.version())"`

VS Code
-------
- Terminal -> Run Task (build/run Debug/Release)
- Debug: select "Launch tcl2_demo (Debug)" and press F5

FFT Backend
-----------
- Built-in radix-2 FFT only (pocketfft disabled). Correlation/conv helpers zero-pad automatically.

Correlation FFT API
-------------------
Header: `taco/correlation_fft.hpp`
- `bcf::bcf_fft_fun(N, dt, J, beta, t, C)` builds time‑domain C(t) from spectral J(ω) at inverse temperature β.
- Convention: we compute `C = fft(S) * (dω/π)` on a symmetric KMS spectrum `S(ω)` (DC/+ω/Nyquist/−ω). This matches the MATLAB helper (`bcf_fft_ohmic_simple.m`).
- Choose `dt` so `π/dt` exceeds J’s support; choose `N` so `N·dt` covers your required time window

Modules Overview
----------------
- `taco/ops.hpp` — Pauli operators, ladder/basis ops, state builders, trace/purity helpers, vec/unvec, superoperators, norms.
- `taco/rk4_dense.hpp` - dense-matrix RK4 utilities (serial/omp) for r' = L r.
- `taco/system.hpp` — diagonalize the Hamiltonian, compute Bohr frequencies, group transitions into frequency buckets, and slice jump operators spectrally.
- `taco/gamma.hpp` — trapezoid/Simpson integrators for Γ(ω,t); streaming accumulator; multi-ω prefix/final helpers and ω deduplication.
- `taco/bath_tabulated.hpp` — tabulated correlation function with linear interpolation; FFT-based builders; Ohmic factory.
- `taco/bath_models.hpp` — spectral density models (Ohmic), helpers to go from J(ω) to C(t) and to spectral kernel series aligned with system buckets.
- `taco/generator.hpp` + `cpp/src/tcl/generator.cpp` — assemble TCL2 unitary and dissipative superoperators from spectral kernels (returns `TCL2Components`).
- `taco/tcl2.hpp` + `cpp/src/tcl/tcl2_generator.cpp` — stateful TCL2 generator (reset/advance/apply) for time stepping.
- `taco/spin_boson.hpp` — convenience builders and `Model` wrapper for the spin-boson Hamiltonian, bath, and generator.
- `taco/tcl4_kernels.hpp`, `taco/tcl4.hpp` — TCL4 kernel builder and triple-series helper (Γ→F/C/R) [work-in-progress].
- Executables: `examples/spin_boson.cpp` (CLI simulator), `examples/generator_demo.cpp` (L builder example), `examples/tcl2_demo.cpp` (legacy demo).
- Overall layout: see `docs/STRUCTURE.md` for a filesystem tree.

Module Reference
----------------

### `taco/ops.hpp`
- Pauli matrices, ladder operators, projectors, generic `|i><j|` builders.
- Pure-state/density helpers (`rho_pure`, `rho_qubit_0/1`, Bloch conversions).
- Trace, inner products, partial traces, Hermitization/normalization.
- Vectorization utilities (`vec`, `unvec`, `super_left/right`, `kron`).
- Norms/distances (`fro_norm`, `op_norm2`, `trace_norm`, `fidelity`, `trace_distance`).

### `taco/system.hpp`
- `Eigensystem`: diagonalize Hermitian H, cache eigenpairs, transform operators between lab/eigen bases.
- `BohrFrequencies`: precompute ω_{mn} = ε_m − ε_n.
- `FrequencyIndex`: group transitions into buckets (unique ω within tolerance) and supply lookup maps.
- Spectral decomposition helpers that slice each jump operator per bucket and cache A(ω), A†(ω), A†A(ω).
- `System`: aggregates the above for convenient downstream use.

### `taco/gamma.hpp`
- Batch prefix integrators (`compute_trapz`, `compute_simpson`) for Γ(ω,t).
- Streaming accumulator `GammaTrapzAccumulator` for multi-ω incremental updates.
- Multi-ω matrix outputs: `compute_trapz_prefix_multi` (vector-of-vectors), `compute_trapz_prefix_multi_matrix` (Eigen matrix), and `compute_trapz_final_multi` (final values only).
- Omega deduplication helper (`deduplicate_omegas`) to avoid redundant work when frequencies repeat.

### `taco/bath_tabulated.hpp`
- `TabulatedCorrelation`: interpolation-based implementation of `bath::CorrelationFunction` backed by sampled data.
- Factory functions:
  - `TabulatedCorrelation::diagonal` for diagonal baths.
  - `TabulatedCorrelation::from_spectral` to FFT from J(ω).
  - `make_ohmic_bath` convenience wrapper (Ohmic with exponential cutoff).
- Accessors expose the time grid (`times()`) and raw data blocks for diagnostics.

### `taco/bath_models.hpp`
- Spectral density structs (currently `OhmicDrude` with `J(ω)`).
- Builders from models:
  - `build_correlation_from_J` → `TabulatedCorrelation`.
  - `build_spectral_kernels_from_correlation` / `build_spectral_kernels_from_J` → asymptotic Γ(ω) aligned to system buckets.
  - `build_spectral_kernels_prefix_series` → time series Γ(ω, t_k) using the streaming accumulator.

### `taco/generator.hpp` / `cpp/src/tcl/generator.cpp`
- `SpectralKernels`: per-bucket complex kernel matrices Γ_{αβ}(ω).
- `TCL2Components`: returns unitary (`L_unitary`), dissipator (`L_dissipator`), Lamb shift (`H_lamb_shift`), and `total()` convenience.
- Builders:
  - `build_lamb_shift`: accumulate S(ω) contributions into H_ls.
  - `build_unitary_superop`: form −i[I⊗H_eff − H_eff^T⊗I].
  - `build_dissipator_superop`: assemble the Redfield/TCL2 dissipator with optional Lamb-shift accumulation.
  - `build_tcl2_components`: one-call wrapper combining the above.

### `taco/tcl2.hpp` / `cpp/src/tcl/tcl2_generator.cpp`
- Stateful `TCL2Generator` for time stepping (reset/advance/apply) with cached buckets, Γ integrals (Simpson-based), and Lamb shift.
- `GeneratorOptions` to tune tolerances and integration segments.
- Used by propagation helpers (`propagate_rk4`, `propagate_expm`).

### `taco/spin_boson.hpp`
- `Params` struct (Δ, ε, α, ω_c, β, rank, coupling operator).
- Builders: Hamiltonian, jump operators, tabulated bath.
- `Model`: owns `TabulatedCorrelation` and `TCL2Generator` with proper lifetime management.

### Executables & Tests
- `examples/spin_boson.cpp`: CLI simulator (parameterized via flags/config), writes observables and full ρ(t).
- `examples/generator_demo.cpp`: showcases `tcl2::build_tcl2_components` for small systems.
- `examples/tcl2_demo.cpp`: legacy fixed-step TCL2 demo (kept for reference).
- Tests: `tests/integrator_tests.cpp`, `tests/gamma_tests.cpp`, `tests/spin_boson_tests.cpp` (now a data dump for regression analysis).

Propagation Helpers
-------------------
Header: `taco/propagate.hpp`
- Steppers: `propagate_rk4`, `propagate_expm` (small N)
- Utilities: `hermitize_and_normalize`, `build_liouvillian_at`, `precompute_expm`, `apply_precomputed_expm`

Dense RK4 Helpers
-----------------
Header: `taco/rk4_dense.hpp`
- Fixed-step RK4 for dense L: `rk4_dense_step_serial`, `rk4_dense_step_omp`
- Propagators: `propagate_rk4_dense_serial`, `propagate_rk4_dense_omp`

- Spin-Boson Simulator
----------------------
- Build: `cmake --build build --config Release --target spin_boson`
- Run with defaults: `build/Release/spin_boson.exe`
- Outputs (CSV):
  - `spin_boson_observables.csv` — time, ⟨σ_z⟩, excited population
  - `spin_boson_density.csv` — time, full density matrix entries (real/imag)
- Override parameters via `--key=value` flags:
  - System/bath: `--delta`, `--epsilon`, `--alpha`, `--omega_c`, `--beta`, `--rank`, `--coupling=sz|sx|sy|sm|sp`
  - Spectral density: `--spectral=ohmic|subohmic|superohmic|custom`, `--s=<exponent>`, `--cutoff=exponential|drude`
  - Simulation: `--t0`, `--tf`, `--dt`, `--sample_every`
  - Correlation grid: `--ncorr`, `--dt_corr`
- Output files: `--observables=...`, `--density=...`
- Config template: `configs/spin_boson.yaml` (copy & override via CLI flags)
- Example: `spin_boson.exe --delta=0.8 --epsilon=0.1 --tf=20 --dt=0.02 --coupling=sx`

Integration Utilities
---------------------
Header: `cpp/src/core/integrator.hpp`
- Quadrature: `integrate_trapezoid`, `integrate_simpson`, `integrate_infinite_R`
- Discrete grids: `integrate_discrete_trapz`, `cumulative_trapz`
- Convolution (uniform dx):
  - `convolve_trapz(a,b,dx,mode)` per‑window trapezoid end‑weights (`ConvMode::{Full,Same,Valid}`)
  - `convolve_fft(a,b,dx,mode)` endpoint weights applied once, FFT→multiply→IFFT, scale by `dx`

Convolution Accuracy Notes
--------------------------
- Time‑domain uses local end‑weights; FFT uses global separable weighting → small edge differences (interior matches closely)
- To reduce discrepancy without changing `n`,`m`:
  - Decrease `dx` (denser sampling)
  - Increase zero‑padding in FFT (2×–4×)
  - Compare interior only (ignore first/last ~`m` outputs) if edges aren’t used

Integration Tests
-----------------
- Build: `cmake --build build --config Release --target integrator_tests -j 8`
- Run (Win): `build\Release\integrator_tests.exe`
- Output: writes `integrator_test_results.txt` next to the exe and in the current directory
- Tests cover quadrature, discrete trapz, cumulative trapz, and convolution (trapz vs FFT)

Choosing dt, N, m
-----------------
- Resolve oscillations: ≥12–16 points per period of highest `ω` → `dt ≤ 2π/(p·ωmax)`, p≈12–16
- Cover kernel support: if `τc` is a decay time, `N·dt ≥ 6–8·τc` and similarly for `m·dt`
- For correlation FFT, ensure `Tper = Nfft·dt` comfortably exceeds your analysis window to avoid wrap‑around

TCL4 MIKX Notes
----------------
- API: `MikxTensors build_mikx(const Tcl4Map& map, const TripleKernelSeries& kernels, std::size_t time_index)`
- Inputs: `kernels.F/C/R[f1][f2][f3]` is an `Eigen::VectorXcd` time series; `time_index` selects the sample.
- Outputs:
  - `M`, `I`, `K`: `N^2×N^2` matrices with row `(j,k)` and col `(p,q)` flattened using column‑major pair mapping to match `vec`/`unvec`:
    - row `(j,k)` → `idx = j + k*N`
    - col `(p,q)` → `idx = p + q*N`
  - `X`: flat `N^6` vector, row‑major over `(j,k,p,q,r,s)`.
- Mapping to MATLAB MIKX.m:
  - `M = F[f(j,k), f(j,q), f(p,j)] − R[f(j,q), f(p,q), f(q,k)]`
  - `I = F[f(j,k), f(q,p), f(k,q)]`
  - `K = R[f(j,k), f(p,q), f(q,j)]`
  - `X = C[f(j,k), f(p,q), f(r,s)] + R[f(j,k), f(p,q), f(r,s)]`

TCL4 F/C/R Methods
------------------
- Two parallel implementations are surfaced via `taco::tcl4::FCRMethod`:
  - `Convolution` (default): fast path intended to use FFT‑based Volterra convolutions and pagewise GEMM.
  - `Direct`: time-domain construction (current implementation).
- APIs:
  - `compute_FCR_time_series_direct(...)`
  - `compute_FCR_time_series_convolution(...)` (scalar path uses FFT-based Volterra convolution; matrix path still delegates to Direct)
  - `compute_FCR_time_series(..., method=FCRMethod::Convolution)`
  - `compute_triple_kernels(..., method=FCRMethod::Convolution)`
- Default selection is `Convolution` so future upgrades don’t change call sites.
- Implementation status: scalar (1×1) series use the FFT-based convolution path; matrix-valued kernels still fall back to the direct method pending page-wise GEMM integration.

TCL4 Vector Path & Rebuild Helpers
----------------------------------
- Kernels operate on unique Γ(ω,t) columns (bucket‑major) as `Eigen::VectorXcd` for cache locality.
- Frequency “transpose” in F is implemented by using the mirrored bucket: Γ^T(ω_b) ≡ Γ(−ω_b).
- Rebuild helpers (for inspection or downstream consumers):
  - `build_gamma_matrix_at(map, gamma_series, t_idx)` → N×N Γ at a given time.
  - `build_FCR_6d_at(map, kernels, t_idx, F,C,R)` → flat N^6 tensors at time t_idx.
  - `build_FCR_6d_final(map, kernels, F,C,R)` → convenience for last time index.
  - `build_FCR_6d_series(map, kernels, F_series, C_series, R_series)` → full time series, time‑major; `F_series[t]` is flat N^6.

TCL4 Driver & Tests
-------------------
- Driver: `examples/tcl4_driver.cpp` runs the full TCL4 pipeline (Γ via FFT → F/C/R → MIKX → assemble) and prints diagnostics.
- Test: `tests/tcl4_tests.cpp` compares Direct vs Convolution F/C/R across multiple (N, dt, T) cases and reports max relative errors.
- Spin‑Boson TCL4 demo: `examples/spin_boson_tcl4.cpp` composes `L_total = L2 + α²·GW` and propagates ρ(t) (frozen‑L Euler) while printing ⟨σ_z⟩.

High‑Level TCL4 Wrappers
------------------------
- Build `GW` at a single time: `build_TCL4_generator(system, gamma_series, dt, time_index, method)`.
- Build `GW` for all times: `build_correction_series(system, gamma_series, dt, method)`.
- These call `compute_triple_kernels` internally (frequency space) and then `build_mikx` + `assemble_liouvillian` with `system.A_eig`.

Frequency Buckets Symmetry
--------------------------
- `taco::tcl4::Tcl4Map` includes `mirror_index[b]` such that `omegas[mirror_index[b]] ≈ -omegas[b]` (and self for `ω≈0`).
- Use this to implement operations that require flipping frequency sign (e.g., frequency‑domain “transpose” across +/−ω) without reordering buckets.
