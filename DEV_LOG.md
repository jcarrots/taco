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

Date: 2025-10-10
Summary: Implement TCL4 M/I/K/X assembly (Phase 2) with explicit contractions
Details:
  - Rationale:
    - Unblock TCL4 pipeline by building tensors M, I, K, X from precomputed F/C/R triple-kernel series at a selected time.
    - Keep layout simple and cache-friendly; mirror MATLAB MIKX.m index permutations explicitly for correctness.
  - Files:
    - Core: cpp/src/tcl/tcl4_mikx.cpp, cpp/include/taco/tcl4_mikx.hpp
    - Docs: docs/STRUCTURE.md, docs/TCL4_PLAN.md, DEV_GUIDE.md (new notes)
  - Notes:
    - M, I, K are N^2×N^2 Eigen matrices (row=(j,k), col=(p,q)); X is a flat N^6 std::vector in row-major over (j,k,p,q,r,s).
    - Contractions map 1–1 to MATLAB:
      - M = F[f(j,k), f(j,q), f(p,j)] − R[f(j,q), f(p,q), f(q,k)]
      - I = F[f(j,k), f(q,p), f(k,q)]
      - K = R[f(j,k), f(p,q), f(q,j)]
      - X = C[f(j,k), f(p,q), f(r,s)] + R[f(j,k), f(p,q), f(r,s)]
    - Frequency lookup uses map.pair_to_freq; throws if any required pair is unmapped (−1).
    - Time selection via time_index into Eigen::VectorXcd stored at F/C/R[f1][f2][f3].
  - Migration:
    - None. Downstream Phase 3 (assemble_liouvillian) still pending; API unchanged.

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
