Developer Guide
===============

This guide summarizes how to build, run, test, and tune the projectâ€™s core components (TCL2 runtime, FFT correlation, integration utilities).

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

GCC (MSYS2)
-----------
- Install toolchain in the "MSYS2 MinGW 64-bit" shell:
  - `pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja`
- Add `C:\msys64\mingw64\bin` to PATH and reopen your terminal.
- Configure/build:
  - `cmake -S . -B build-gcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++`
  - `cmake --build build-gcc`

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
- `bcf::bcf_fft_fun(N, dt, J, beta, t, C)` builds timeâ€‘domain C(t) from spectral J(Ï‰) at inverse temperature Î².
- Convention: we compute `C = fft(S) * (dÏ‰/Ï€)` on a symmetric KMS spectrum `S(Ï‰)` (DC/+Ï‰/Nyquist/âˆ’Ï‰). This matches the MATLAB helper (`bcf_fft_ohmic_simple.m`).
- Choose `dt` so `Ï€/dt` exceeds Jâ€™s support; choose `N` so `NÂ·dt` covers your required time window

Modules Overview
----------------
- `taco/ops.hpp` â€” Pauli operators, ladder/basis ops, state builders, trace/purity helpers, vec/unvec, superoperators, norms.
- `taco/rk4_dense.hpp` - dense-matrix RK4 utilities (serial/omp) for r' = L r.
- `taco/system.hpp` â€” diagonalize the Hamiltonian, compute Bohr frequencies, group transitions into frequency buckets, and slice jump operators spectrally.
- `taco/gamma.hpp` â€” trapezoid/Simpson integrators for Î“(Ï‰,t); streaming accumulator; multi-Ï‰ prefix/final helpers and Ï‰ deduplication.
- `taco/bath_tabulated.hpp` â€” tabulated correlation function with linear interpolation; FFT-based builders; Ohmic factory.
- `taco/bath_models.hpp` â€” spectral density models (Ohmic), helpers to go from J(Ï‰) to C(t) and to spectral kernel series aligned with system buckets.
- `taco/generator.hpp` + `cpp/src/tcl/generator.cpp` â€” assemble TCL2 unitary and dissipative superoperators from spectral kernels (returns `TCL2Components`).
- `taco/tcl2.hpp` + `cpp/src/tcl/tcl2_generator.cpp` â€” stateful TCL2 generator (reset/advance/apply) for time stepping.
- `taco/spin_boson.hpp` â€” convenience builders and `Model` wrapper for the spin-boson Hamiltonian, bath, and generator.
- `taco/tcl4_kernels.hpp`, `taco/tcl4.hpp` â€” TCL4 kernel builder and triple-series helper (Î“â†’F/C/R) [work-in-progress].
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
- `BohrFrequencies`: precompute Ï‰_{mn} = Îµ_m âˆ’ Îµ_n.
- `FrequencyIndex`: group transitions into buckets (unique Ï‰ within tolerance) and supply lookup maps.
- Spectral decomposition helpers that slice each jump operator per bucket and cache A(Ï‰), Aâ€ (Ï‰), Aâ€ A(Ï‰).
- `System`: aggregates the above for convenient downstream use.

### `taco/gamma.hpp`
- Batch prefix integrators (`compute_trapz`, `compute_simpson`) for Î“(Ï‰,t).
- Streaming accumulator `GammaTrapzAccumulator` for multi-Ï‰ incremental updates.
- Multi-Ï‰ matrix outputs: `compute_trapz_prefix_multi` (vector-of-vectors), `compute_trapz_prefix_multi_matrix` (Eigen matrix), and `compute_trapz_final_multi` (final values only).
- Omega deduplication helper (`deduplicate_omegas`) to avoid redundant work when frequencies repeat.

### `taco/bath_tabulated.hpp`
- `TabulatedCorrelation`: interpolation-based implementation of `bath::CorrelationFunction` backed by sampled data.
- Factory functions:
  - `TabulatedCorrelation::diagonal` for diagonal baths.
  - `TabulatedCorrelation::from_spectral` to FFT from J(Ï‰).
  - `make_ohmic_bath` convenience wrapper (Ohmic with exponential cutoff).
- Accessors expose the time grid (`times()`) and raw data blocks for diagnostics.

### `taco/bath_models.hpp`
- Spectral density structs (currently `OhmicDrude` with `J(Ï‰)`).
- Builders from models:
  - `build_correlation_from_J` â†’ `TabulatedCorrelation`.
  - `build_spectral_kernels_from_correlation` / `build_spectral_kernels_from_J` â†’ asymptotic Î“(Ï‰) aligned to system buckets.
  - `build_spectral_kernels_prefix_series` â†’ time series Î“(Ï‰, t_k) using the streaming accumulator.

### `taco/generator.hpp` / `cpp/src/tcl/generator.cpp`
- `SpectralKernels`: per-bucket complex kernel matrices Î“_{Î±Î²}(Ï‰).
- `TCL2Components`: returns unitary (`L_unitary`), dissipator (`L_dissipator`), Lamb shift (`H_lamb_shift`), and `total()` convenience.
- Builders:
  - `build_lamb_shift`: accumulate S(Ï‰) contributions into H_ls.
  - `build_unitary_superop`: form âˆ’i[IâŠ—H_eff âˆ’ H_eff^TâŠ—I].
  - `build_dissipator_superop`: assemble the Redfield/TCL2 dissipator with optional Lamb-shift accumulation.
  - `build_tcl2_components`: one-call wrapper combining the above.

### `taco/tcl2.hpp` / `cpp/src/tcl/tcl2_generator.cpp`
- Stateful `TCL2Generator` for time stepping (reset/advance/apply) with cached buckets, Î“ integrals (Simpson-based), and Lamb shift.
- `GeneratorOptions` to tune tolerances and integration segments.
- Used by propagation helpers (`propagate_rk4`, `propagate_expm`).

### `taco/spin_boson.hpp`
- `Params` struct (Î”, Îµ, Î±, Ï‰_c, Î², rank, coupling operator).
- Builders: Hamiltonian, jump operators, tabulated bath.
- `Model`: owns `TabulatedCorrelation` and `TCL2Generator` with proper lifetime management.

### Executables & Tests
- `examples/spin_boson.cpp`: CLI simulator (parameterized via flags/config), writes observables and full Ï(t).
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
- Time-dependent L: pass endpoint series `L_series` (size steps + 1). For best RK4 accuracy, pass `L_half_series` (size steps) for midpoints.
Example (endpoints + midpoints):
```cpp
const std::size_t steps = static_cast<std::size_t>((tf - t0) / dt);
std::vector<Eigen::MatrixXcd> L_series(steps + 1);
std::vector<Eigen::MatrixXcd> L_half_series(steps);

for (std::size_t i = 0; i < steps; ++i) {
    L_series[i] = build_L_at(t0 + i * dt);
    L_half_series[i] = build_L_at(t0 + (i + 0.5) * dt);
}
L_series[steps] = build_L_at(t0 + steps * dt);

Eigen::VectorXcd r = initial_state_vec; // size N^2
taco::tcl::propagate_rk4_dense_serial(L_series, L_half_series, r, t0, dt);
```

- Spin-Boson Simulator
----------------------
- Build: `cmake --build build --config Release --target spin_boson`
- Run with defaults: `build/Release/spin_boson.exe`
- Outputs (CSV):
  - `spin_boson_observables.csv` â€” time, âŸ¨Ïƒ_zâŸ©, excited population
  - `spin_boson_density.csv` â€” time, full density matrix entries (real/imag)
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
  - `convolve_trapz(a,b,dx,mode)` perâ€‘window trapezoid endâ€‘weights (`ConvMode::{Full,Same,Valid}`)
  - `convolve_fft(a,b,dx,mode)` endpoint weights applied once, FFTâ†’multiplyâ†’IFFT, scale by `dx`

Convolution Accuracy Notes
--------------------------
- Timeâ€‘domain uses local endâ€‘weights; FFT uses global separable weighting â†’ small edge differences (interior matches closely)
- To reduce discrepancy without changing `n`,`m`:
  - Decrease `dx` (denser sampling)
  - Increase zeroâ€‘padding in FFT (2Ã—â€“4Ã—)
  - Compare interior only (ignore first/last ~`m` outputs) if edges arenâ€™t used

Integration Tests
-----------------
- Build: `cmake --build build --config Release --target integrator_tests -j 8`
- Run (Win): `build\Release\integrator_tests.exe`
- Output: writes `integrator_test_results.txt` next to the exe and in the current directory
- Tests cover quadrature, discrete trapz, cumulative trapz, and convolution (trapz vs FFT)

Choosing dt, N, m
-----------------
- Resolve oscillations: â‰¥12â€“16 points per period of highest `Ï‰` â†’ `dt â‰¤ 2Ï€/(pÂ·Ï‰max)`, pâ‰ˆ12â€“16
- Cover kernel support: if `Ï„c` is a decay time, `NÂ·dt â‰¥ 6â€“8Â·Ï„c` and similarly for `mÂ·dt`
- For correlation FFT, ensure `Tper = NfftÂ·dt` comfortably exceeds your analysis window to avoid wrapâ€‘around

TCL4 MIKX Notes
----------------
- API: `MikxTensors build_mikx_serial(const Tcl4Map& map, const TripleKernelSeries& kernels, std::size_t time_index)`
- Inputs: `kernels.F/C/R[f1][f2][f3]` is an `Eigen::VectorXcd` time series; `time_index` selects the sample.
- Outputs:
  - `M`, `I`, `K`: `N^2Ã—N^2` matrices with row `(j,k)` and col `(p,q)` flattened using columnâ€‘major pair mapping to match `vec`/`unvec`:
    - row `(j,k)` â†’ `idx = j + k*N`
    - col `(p,q)` â†’ `idx = p + q*N`
  - `X`: flat `N^6` vector, columnâ€‘major over `(j,k,p,q,r,s)`.
- Mapping to MATLAB MIKX.m:
  - `M = F[f(j,k), f(j,q), f(p,j)] âˆ’ R[f(j,q), f(p,q), f(q,k)]`
  - `I = F[f(j,k), f(q,p), f(k,q)]`
  - `K = R[f(j,k), f(p,q), f(q,j)]`
  - `X = C[f(j,k), f(p,q), f(r,s)] + R[f(j,k), f(p,q), f(r,s)]`

TCL4 F/C/R Methods
------------------
- Two parallel implementations are surfaced via `taco::tcl4::FCRMethod`:
  - `Convolution` (default): fast path intended to use FFTâ€‘based Volterra convolutions and pagewise GEMM.
  - `Direct`: time-domain construction (current implementation).
- APIs:
  - `compute_FCR_time_series_direct(...)`
  - `compute_FCR_time_series_convolution(...)` (scalar path uses FFT-based Volterra convolution; matrix path still delegates to Direct)
  - `compute_FCR_time_series(..., method=FCRMethod::Convolution)`
  - `compute_triple_kernels(..., method=FCRMethod::Convolution)`
- Default selection is `Convolution` so future upgrades donâ€™t change call sites.
- Implementation status: scalar (1Ã—1) series use the FFT-based convolution path; matrix-valued kernels still fall back to the direct method pending page-wise GEMM integration.

TCL4 Vector Path & Rebuild Helpers
----------------------------------
- Kernels operate on unique Î“(Ï‰,t) columns (bucketâ€‘major) as `Eigen::VectorXcd` for cache locality.
- Frequency â€œtransposeâ€ in F is implemented by using the mirrored bucket: Î“^T(Ï‰_b) â‰¡ Î“(âˆ’Ï‰_b).
- Rebuild helpers (for inspection or downstream consumers):
  - `build_gamma_matrix_at(map, gamma_series, t_idx)` â†’ NÃ—N Î“ at a given time.
  - `build_FCR_6d_at(map, kernels, t_idx, F,C,R)` â†’ flat N^6 tensors at time t_idx.
  - `build_FCR_6d_final(map, kernels, F,C,R)` â†’ convenience for last time index.
  - `build_FCR_6d_series(map, kernels, F_series, C_series, R_series)` â†’ full time series, timeâ€‘major; `F_series[t]` is flat N^6.

TCL4 Driver & Tests
-------------------
- Driver: `examples/tcl4_driver.cpp` runs the full TCL4 pipeline (Î“ via FFT â†’ F/C/R â†’ MIKX â†’ assemble) and prints diagnostics.
- Test: `tests/tcl4_tests.cpp` compares Direct vs Convolution F/C/R across multiple (N, dt, T) cases and reports max relative errors.
- Spinâ€‘Boson TCL4 demo: `examples/spin_boson_tcl4.cpp` composes `L_total = L2 + Î±Â²Â·GW` and propagates Ï(t) (frozenâ€‘L Euler) while printing âŸ¨Ïƒ_zâŸ©.

Highâ€‘Level TCL4 Wrappers
------------------------
- Build `GW` at a single time: `build_TCL4_generator(system, gamma_series, dt, time_index, method)`.
- Build `GW` for all times: `build_correction_series(system, gamma_series, dt, method)`.
- These call `compute_triple_kernels` internally (frequency space) and then `build_mikx_serial` + `assemble_liouvillian` with `system.A_eig`.

Frequency Buckets Symmetry
--------------------------
- `taco::tcl4::Tcl4Map` includes `mirror_index[b]` such that `omegas[mirror_index[b]] â‰ˆ -omegas[b]` (and self for `Ï‰â‰ˆ0`).
- Use this to implement operations that require flipping frequency sign (e.g., frequencyâ€‘domain â€œtransposeâ€ across +/âˆ’Ï‰) without reordering buckets.

TCL4 HDF5 Compare (MATLAB Benchmarks)
-------------------------------------
- Compare against MATLAB-exported HDF5: `tests/tcl4_h5_compare.cpp`.
- Build: `cmake --build build --config Release --target tcl4_h5_compare`
- List datasets: `build/Release/tcl4_h5_compare.exe --file=tests/tcl_test.h5 --list`
  - Gt is the benchmark reference; compare in matrix mode using map/ij.
    - Example (MATLAB column-major flatten, omega mapping):
      - `build/Release/tcl4_h5_compare.exe --file=tests/tcl_test.h5 --compare-gt --gt-flat=col --gt-time=row --gt-map=omega --gamma-rule=rect`
    - If the file was built using map/ij, use `--gt-map=ij`.
    - If the file omegas are sorted (e.g., `[-1,0,1]`), use `--gt-omega-order=sorted`.
    - If the file is row-major, use `--gt-flat=row`.
    - If time is stored in columns (dataset dims `[flat_len, Nt]`), use `--gt-time=col` (default is row).
  - If the file includes an extra leading sample, use `--gt-offset=1`.
  - After Gt matches, compare GW (column-major flatten):
    - `build/Release/tcl4_h5_compare.exe --file=tests/tcl_test.h5 --compare-gw --gw-flat=col --tidx=1 --one-based`
    - Use `--gw-time=col` if GW time is stored in columns.
  - Print MIKX tensors (M/I/K/X) for a specific time index:
    - Computed kernels (from C(t)): `build/Release/tcl4_h5_compare.exe --file=tests/tcl_test.h5 --print-mikx --tidx=100`
    - From file kernels (to match MATLAB exports): `build/Release/tcl4_h5_compare.exe --file=tests/tcl_test.h5 --print-mikx --mikx-source=file --fcr-time=first --fcr-order=col --fcr-omega-order=sorted --fcr-axes=0,1,2 --tidx=100`
  - To compare kernels (F/C/R) against MATLAB exports:
    - `build/Release/tcl4_h5_compare.exe --file=tests/tcl_test.h5 --compare-fcr --fcr-time=first --fcr-order=col --fcr-omega-order=sorted --fcr-axes=0,1,2 --tidx=0,1,10`
  - Add `--print-fcr --fcr=f` (or `--fcr=c`, `--fcr=r`) to dump a single kernel slice at the requested time.
  - Use `--fcr-omega=0` or `--fcr-ijk=i,j,k` to compare a single (i,j,k) triple.
  - If the file orders omegas differently (e.g., sorted `[-1,0,1]`), use `--fcr-omega-order=sorted`.
  - If file kernels are shifted in time, use `--fcr-offset=1` (or another offset).
  - If the omega axes are permuted in the file, use `--fcr-axes=0,1,2` to map file axes to computed axes.
  - If MATLAB wrote `Nt x nf x nf x nf` but the file reports `nf x nf x nf x Nt`, use `--fcr-time=first`.
  - If the file data is row-major (C-style), use `--fcr-order=row` (default is column-major/MATLAB).
  - To match MATLAB FFT padding (e.g., 8N), use `--fcr-fft-pad=8`.
- To compare direct vs convolution methods (self-check, no HDF5 kernels needed):
  - `build/Release/tcl4_h5_compare.exe --file=tests/tcl_test.h5 --compare-fcr-methods --fcr-nt=1024 --fcr-omega=0 --tidx=0,1,10`


