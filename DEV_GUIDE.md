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

Propagation Helpers
-------------------
Header: `taco/propagate.hpp`
- Steppers: `propagate_rk4`, `propagate_expm` (small N)
- Utilities: `hermitize_and_normalize`, `build_liouvillian_at`, `precompute_expm`, `apply_precomputed_expm`

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

