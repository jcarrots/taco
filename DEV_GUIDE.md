Developer Guide
===============

This guide summarizes how to build, run, test, and tune the project’s core components (TCL2 runtime, FFT correlation, integration utilities).

Prereqs
-------
- CMake 3.20+
- A C++17 compiler (MSVC 2022 on Windows; Clang/GCC on Linux/macOS)

Quick Build
-----------
- Configure: `cmake -S . -B build`
- Build Debug: `cmake --build build --config Debug -j 8`
- Build Release: `cmake --build build --config Release -j 8`
- Run demo (Win): `build\Release\tcl2_demo.exe`

VS Code
-------
- Terminal -> Run Task (build/run Debug/Release)
- Debug: select "Launch tcl2_demo (Debug)" and press F5

FFT Backend
-----------
- Preferred: pocketfft (BSD, header‑only)
  - vcpkg: `vcpkg install pocketfft:x64-windows` (configure with vcpkg toolchain)
  - Vendored: add `third_party/pocketfft/pocketfft_hdronly.h` (auto‑detected)
- Fallback: in‑house radix‑2 FFT (power‑of‑two). Correlation/conv helpers zero‑pad automatically.

Correlation FFT API
-------------------
Header: `taco/correlation_fft.hpp`
- `bcf::bcf_fft_fun(N, dt, J, beta, t, C)` builds time‑domain C(t) from spectral J(ω) at inverse temperature β
- Choose `dt` so `π/dt` exceeds J’s support; choose `N` so `N·dt` covers your required time window

Modules Overview
----------------
- `taco/ops.hpp` — Pauli operators, ladder/basis ops, state builders, trace/purity helpers, vec/unvec, superoperators, norms.
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
