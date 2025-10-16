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

Date: 2025-10-16
Summary: Implement TCL4 Liouvillian assembly (NAKZWAN) in C++
Details:
  - Rationale:
    - Complete Phase 3 of TCL4 pipeline by porting MATLAB NAKZWAN_v9.m to C++ so the M/I/K/X tensors produce the fourth-order correction matrix GW.
  - Files:
    - Core: cpp/src/tcl/tcl4_assemble.cpp
  - Notes:
    - Function `assemble_liouvillian(const MikxTensors&, const std::vector<Eigen::MatrixXcd>&)` now computes T(n,i,m,j) with nested sums over coupling channels and indices, using the same index mapping as MATLAB and `tcl4_mikx`.
    - Indexing:
      - 4D tensors M/I/K are accessed as M(b,i,n,a) = M[row=(b,i), col=(n,a)] etc. with row/col flatteners `flat2(N,·,·)`.
      - 6D tensor X is accessed as X(j,k,p,q,r,s) with row-major `flat6` (matches `tcl4_mikx`).
    - Output GW is symmetrized as `T + T.adjoint()` (MATLAB `'`).
    - Validates shapes and coupling operator dimensions; throws on mismatch.
  - Migration:
    - None. Performance optimizations (blocking/parallelization) can be added later after correctness validation against MATLAB.

Date: 2025-10-16
Summary: Add FCRMethod and default to Convolution; direct path retained
Details:
  - Rationale:
    - Support parallel implementations of TCL4 F/C/R kernels with a stable, explicit selection knob. Default to the faster convolution path without changing call sites later.
  - Files:
    - Headers: cpp/include/taco/tcl4_kernels.hpp (added FCRMethod; direct/convolution selectors; wrapper), cpp/include/taco/tcl4.hpp (method plumbed into triple-series)
    - Sources: cpp/src/tcl/tcl4_kernels.cpp (factored direct; added convolution stub delegating to direct), cpp/src/tcl/tcl4.cpp (method parameter)
    - Docs: DEV_GUIDE.md (methods section), docs/TCL4_PLAN.md (Phase 1b plan)
  - Notes:
    - `compute_FCR_time_series_convolution` currently calls the Direct implementation; will be replaced with FFT‑based Volterra convolution.
    - API defaults to `FCRMethod::Convolution` in both per‑pair and triple‑series builders.
  - Migration:
    - Existing calls to `compute_FCR_time_series(G1,G2,omega,dt,op2)` still compile via the wrapper. Behavior currently unchanged because Convolution delegates to Direct.

Date: 2025-10-16
Summary: Change F builder to use mirrored frequency for Γ^T (drop channel apply_op)
Details:
  - Rationale:
    - In the TCL4 kernel construction, the transpose acts in frequency space: Γ(ω)^T = Γ(−ω). Implement this by selecting the mirrored bucket instead of transposing channel matrices.
  - Files:
    - Kernels: cpp/src/tcl/tcl4_kernels.cpp (removed `apply_op`; F now uses provided G2 as-is)
    - Triple driver: cpp/src/tcl/tcl4.cpp (build both original and mirrored G2; compute F with mirrored, C/R with original)
  - Notes:
    - Mirror lookups use `Tcl4Map::mirror_index`; zero bucket maps to itself.
    - `SpectralOp` parameter remains in signatures for now but is ignored in the direct path; callers pass `Identity`.
  - Migration:
    - No public API change required. Behavior now matches intended Γ^T semantics.

Date: 2025-10-16
Summary: Add scalar-series F/C/R builders and use Γ columns directly
Details:
  - Rationale:
    - Avoid building vectors of 1×1 matrices per time sample; improve cache locality and reduce allocations by operating on `Eigen::VectorXcd` time columns directly.
  - Files:
    - Kernels: cpp/include/taco/tcl4_kernels.hpp (vector-based overloads), cpp/src/tcl/tcl4_kernels.cpp (implementations)
    - Triple driver: cpp/src/tcl/tcl4.cpp (now passes `gamma_series.col(...)` to vector overloads)
  - Notes:
    - Direct method uses tk = k·dt on the fly; no explicit time array required.
    - Convolution method will later replace the direct versions under the `FCRMethod::Convolution` dispatch.
  - Migration:
    - No API breaks; existing matrix-series path remains for multi-channel generalization.

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
