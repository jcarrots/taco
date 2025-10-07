Developer Change Log
====================

Purpose
- This log is for developers to track every feature/change over time.
- Never delete or rewrite past entries. Append new entries at the top.
- Each entry should include: date, summary, rationale, main files, notes/migrations.

Template (copy/paste for new entries)
- Date: YYYY‑MM‑DD
- Summary: One sentence summary
- Details:
  - Rationale: why the change was made
  - Files: list of primary files touched
  - Notes: behavior changes, edge cases, perf, tests
  - Migration: anything users/devs must do after pulling

-----------------------------------------------------------------------

Date: 2025-10-07
Summary: FFT correlation + integration utilities + tests; dev guide; pocketfft backend
Details:
  - Rationale:
    - Provide a permissive, SIMD‑friendly FFT correlation path and keep a fallback.
    - Add robust integration helpers (cumulative, convolution), plus tests.
    - Improve developer experience with a concise dev guide and test outputs.
  - Files:
    - FFT correlation
      - Added header‑only API: `taco/correlation_fft.hpp` (pocketfft preferred; in‑house radix‑2 fallback)
      - Slimmed TU: `cpp/src/core/correlation.cpp` now just includes the header
      - CMake: detect pocketfft or use fallback
    - Integration utilities
      - `cpp/src/core/integrator.hpp`: added `cumulative_trapz`, `convolve_trapz`, `convolve_fft`, `ConvMode`
      - `integrate_infinite_R` switched to composite Simpson (no adaptive dependency)
    - Propagation helpers
      - `cpp/include/taco/propagate.hpp`: generator‑agnostic RK4 and expm (small N) + helpers
    - Tests
      - `tests/integrator_tests.cpp`: coverage for quadrature, discrete trapz, cumulative, convolution (trapz vs FFT)
      - Writes `integrator_test_results.txt` next to exe and in CWD
    - Build system
      - `CMakeLists.txt`: added test target; include paths for `cpp/src`, `cpp/include`; pocketfft detection
    - Docs / scripts
      - Added dev guide: `DEV_GUIDE.md`
      - Existing `README_DEV.md` kept (contains encoding artifacts); prefer `DEV_GUIDE.md` for a clean reference
      - Dev script: `scripts/dev.ps1` for configure/build/run/clean
  - Notes:
    - Convolution: FFT path applies endpoint weights globally; time‑domain applies per‑window trapezoid. Expect ~1–2% edge mismatch; interior aligns closely. Reduce by denser `dx`, larger `n`, modest padding (2×–4×), or per‑window weighting via overlap‑add (future).
    - Correlation FFT: `bcf_fft_fun(N, dt, J, beta, t, C)` returns 0..N samples; ensure `π/dt` covers spectral support and `N·dt` covers required window.
    - Tests: pass in Release on MSVC; tolerances set appropriately for convolution comparison.
  - Migration:
    - Include `taco/correlation_fft.hpp` where correlation FFT is used.
    - Link tests via `integrator_tests` target; run `build/Release/integrator_tests.exe` (Windows) after build.

