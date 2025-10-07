Integration Utilities
---------------------
Headers: `taco/correlation_fft.hpp`, `cpp/src/core/integrator.hpp`

- Quadrature
  - `integrate_trapezoid(f, a, b, n)`; `integrate_simpson(f, a, b, n)`; `integrate_infinite_R(f, tol)` (composite Simpson)
- Discrete grids
  - `integrate_discrete_trapz(y, dx)`; `cumulative_trapz(y, dx)` (prefix trapezoid)
- Convolution (uniform `dx`)
  - `convolve_trapz(a, b, dx, mode)` (per-window trapezoid end-weights)
  - `convolve_fft(a, b, dx, mode)` (endpoint weights applied once, then IFFT·dx)
  - `ConvMode::{Full, Same, Valid}`

Notes on convolution accuracy
-----------------------------
- Time-domain uses local per-output end-weights; FFT uses a global separable weighting.
- Differences appear near edges; interior typically matches closely.
- To reduce discrepancy without changing `n`,`m`:
  - Decrease `dx` (denser sampling)
  - Increase zero-padding in FFT (2×–4×)
  - Compare interior only (ignore first/last ~`m` outputs) if edges aren’t used

Running the integration tests
-----------------------------
- Build: `cmake --build build --config Release --target integrator_tests -j 8`
- Run:  `build/Release/integrator_tests.exe`
- Output: writes `integrator_test_results.txt` next to the exe and in the current directory.

Choosing dt, N, m (rules of thumb)
----------------------------------
- Resolve oscillations: =12–16 points per period of the highest `?` ? `dt = 2p/(p·?max)` (p˜12–16)
- Cover kernel support: if `tc` is a decay time, `N·dt = 6–8·tc` and similarly for `m·dt`
- For correlation FFT, ensure `Tper = Nfft·dt` comfortably exceeds your analysis window to avoid wrap-arounddetected by CMake).
- Fallback: in-house radix-2 FFT (power-of-two). `bcf_fft_fun` zero-pads up to a power of two automatically.

Correlation FFT API
-------------------
Header: `taco/correlation_fft.hpp`

- Build C(t) from a spectral density J(Ï‰) at inverse temperature Î²:
  - `bcf::bcf_fft_fun(N, dt, J, beta, t, C)`
  - Returns non-negative times `t[0..N]` and complex `C[0..N]`.
- Choose `dt` so Nyquist `Ï€/dt` exceeds the support of J(Ï‰), and `N` so `NÂ·dt` covers your largest Ï„.

Propagation Helpers
-------------------
Header: `taco/propagate.hpp`

- Generator-agnostic fixed-step drivers:
  - `propagate_rk4(gen, rho, t0, tf, dt, on_sample)`
  - `propagate_expm(gen, rho, t0, tf, dt, on_sample)` (small N only; dense L)
- Utilities:
  - `hermitize_and_normalize(rho)`
  - `build_liouvillian_at(gen, t)`, `precompute_expm(L, dt)`, `apply_precomputed_expm(M, rho)`



Integration Utilities
---------------------
Header: 	aco/correlation_fft.hpp and cpp/src/core/integrator.hpp`n
- Quadrature
  - integrate_trapezoid(f, a, b, n); integrate_simpson(f, a, b, n); integrate_infinite_R(f, tol) (composite Simpson).
- Discrete grids
  - integrate_discrete_trapz(y, dx); cumulative_trapz(y, dx) (prefix trapezoid).
- Convolution (uniform dx)
  - convolve_trapz(a, b, dx, mode) (per-window trapezoid end-weights).
  - convolve_fft(a, b, dx, mode) (endpoint weights applied once, then IFFT·dx).
  - ConvMode::{Full, Same, Valid}. 

Notes on convolution accuracy
-----------------------------
- Time-domain uses local per-output end-weights; FFT uses a global separable weighting.
- Differences appear near edges; interior typically matches closely.
- To reduce discrepancy without changing n,m:
  - Decrease dx (denser sampling).
  - Increase zero-padding in FFT (2×–4×).
  - Compare interior only (ignore first/last ~m outputs) if edges aren’t used.

Running the integration tests
-----------------------------
- Build: cmake --build build --config Release --target integrator_tests -j 8`n- Run:  uild/Release/integrator_tests.exe`n- Output: writes integrator_test_results.txt next to the exe and in the current directory.

Choosing dt, N, m (rules of thumb)
----------------------------------
- Resolve oscillations: =12–16 points per period of the highest ? ? dt = 2p/(p·?max) (p˜12–16).
- Cover kernel support: if tc is a decay time, N·dt = 6–8·tc and similarly for m·dt.
- For correlation FFT, ensure Tper = Nfft·dt comfortably exceeds your analysis window to avoid wrap-around.


