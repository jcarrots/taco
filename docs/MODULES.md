TACO Module Guide
=================

This guide documents the main C++ headers and source files in the taco runtime that you will interact with when building TCL(2) solvers, baths, and models.

Conventions
-----------
- `N`: Hilbert space dimension of the system
- `NN = N*N`: Liouville (vectorized) dimension
- Matrices are `Eigen::MatrixXcd` unless stated otherwise; vectors are `Eigen::VectorXcd`
- Column-major flattening (`vec`) matches Eigen’s default memory layout


ops.hpp — Operators, states, norms, and superoperators
------------------------------------------------------
Header: `cpp/include/taco/ops.hpp`

What it provides:
- Pauli, ladder and basis operators (2×2 or generic)
  - `sigma_x()`, `sigma_y()`, `sigma_z()`, `sigma_plus()`, `sigma_minus()`, `I2()`
  - `basis_op(N,i,j)` returns `|i><j|`, `projector(N,i)` returns `|i><i|`
  - Harmonic ladder: `a(N)`, `adag(N)`, `number(N)`
- States and density-matrix helpers
  - Kets: `ket(N,i)`, `ket0()`, `ket1()`, superpositions `ket_plus_x/y()`, …
  - Density builders: `rho_pure(psi)`, `rho_qubit_0()`, `rho_qubit_1()`
  - Bloch: `bloch_from_rho(rho)`, `rho_from_bloch(r)` (qubit)
- Algebraic utilities
  - `comm(A,B)` = `[A,B]`, `anti(A,B)` = `{A,B}`
  - `hermitize(A)`, `hermitize_and_normalize(rho)`, `is_density(rho)`
- Trace and inner products
  - `tr(A)`, `tr_real_hermitian(A)`, `hs_inner(A,B)`, `purity(rho)`
  - Partial traces: `ptrace_A(rho,dA,dB)`, `ptrace_B(rho,dA,dB)`
- Superoperators and vectorization
  - `kron(A,B)`, `vec(A)`, `unvec(v,N)`, `super_left(A)=I⊗A`, `super_right(B)=B^T⊗I`
- Norms and distances
  - `fro_norm(A)`, `op_norm2(A)`, `trace_norm(A)`, `schatten_p_norm(A,p)`
  - `op_norm1(A)`, `op_norm_inf(A)`, `max_abs_entry(A)`
  - `fidelity(rho,sigma)`, `trace_distance(rho,sigma)`

Example:
```
auto H = 0.5 * w0 * ops::sigma_z();
auto sm = ops::sigma_minus();
Eigen::MatrixXcd rho = ops::rho_qubit_1();
double p = ops::purity(rho);
```


system.hpp — Eigensystem, Bohr frequencies, and spectral buckets
----------------------------------------------------------------
Header: `cpp/include/taco/system.hpp`

What it provides:
- `Eigensystem` — diagonalize Hermitian `H`: stores `eps` (eigenvalues), `U`, `U†`, and provides `to_eigen`/`to_lab`
- `BohrFrequencies` — precomputes `omega(m,n) = eps(m) - eps(n)` as an `N×N` matrix
- `FrequencyBucket` / `FrequencyIndex` — group all transitions with equal (within tolerance) Bohr frequencies. Supplies mapping and buckets for spectral decomposition.
- Spectral decomposition:
  - `decompose_operator_by_frequency(A_eig, bf, fidx)` returns `A(ω)` slices per bucket
  - Bulk form for multiple channels
- `System` — one-call builder bundling eigensystem, frequency buckets and channel decompositions

Example:
```
sys::System S; S.build(H, {A_lab}, 1e-9);
double w0 = S.fidx.buckets[0].omega;       // first unique Bohr freq
auto A0 = S.A_eig_parts[0][0];             // channel 0, bucket 0 slice
```


gamma.hpp — Discrete Γ(ω,t) integrators and helpers
---------------------------------------------------
Header: `cpp/include/taco/gamma.hpp`

What it provides:
- Batch integrators
  - `compute_trapz(C, dt, omega)` → `G_k` vector (k=0..N-1) via trapezoid
  - `compute_simpson(C, dt, omega, hold_odd)` → Simpson prefix (even indices exact)
- Streaming (multi-ω)
  - `GammaTrapzAccumulator(dt, omegas)`: push samples `C_k` and query current `G(ω)` for all `ω` efficiently (caches `e^{iωdt}` per ω)
- Multi-ω prefix/final builders
  - `compute_trapz_prefix_multi(C, dt, omegas)` → vector of `G(ω_j, t_k)` series
  - `compute_trapz_prefix_multi_matrix(C, dt, omegas)` → `N×M` Matrix (columns per ω)
  - `compute_trapz_final_multi(C, dt, omegas)` → final values `G(ω_j, T)`
- Omega de-duplication (when input ω has repeats)
  - `deduplicate_omegas(omegas, tol)` returns unique `values`, `map` (orig→unique), and `groups`

Notes:
- All trapezoid integrators use correct end-weights. The streaming accumulator maintains `phi[j]=e^{iω_j t_k}` and `phi_step[j]=e^{iω_j dt}` to avoid `exp` in inner loops.


bath_tabulated.hpp — Tabulated correlations and simple factory
---------------------------------------------------------------
Header: `cpp/include/taco/bath_tabulated.hpp`

What it provides:
- `TabulatedCorrelation` — implements `bath::CorrelationFunction` from sampled data:
  - Construction: `TabulatedCorrelation(t, C)` where `C[α][β][k]` matches `t[k]`
  - Evaluation: linear interpolation on `[t[k-1], t[k]]`; zero outside support
  - Accessors: `rank()`, `times()`, `data()`
- Convenience factories:
  - `TabulatedCorrelation::diagonal(r, t, Cdiag)` — fills only `α=β`
  - `TabulatedCorrelation::from_spectral(N, dt, J, beta)` — builds `C(t)` via FFT from spectral density `J(ω)` and inverse temperature `β`
  - `make_ohmic_bath(rank, alpha, omega_c, beta, N, dt)` — Ohmic with exponential cutoff


bath_models.hpp — Common spectral models and Γ builders
-------------------------------------------------------
Header: `cpp/include/taco/bath_models.hpp`

What it provides:
- Spectral models
  - `OhmicDrude{ alpha, omega_c }` with `J(w)`
- Correlation builders
  - `build_correlation_from_J(rank, N, dt, model, beta)` → `TabulatedCorrelation`
- Γ(ω) builders aligned to system frequency buckets
  - `build_spectral_kernels_from_correlation(system, corr, t)` → asymptotic Γ(ω) (final time)
  - `build_spectral_kernels_from_J(system, N, dt, model, beta)` → J→C→Γ
  - `build_spectral_kernels_prefix_series(system, corr, t)` → vector of `SpectralKernels` for all time steps (uses a single streaming pass)

Note: bucket alignment uses `system.fidx.buckets[b].omega` so repeated transition frequencies are already grouped.


generator.hpp / generator.cpp — TCL2 superoperator builders
-----------------------------------------------------------
Headers: `cpp/include/taco/generator.hpp`, `cpp/src/tcl/generator.cpp`

What it provides:
- Data structures
  - `tcl2::SpectralKernels` — for each bucket, a complex kernel matrix `Γ_{αβ}(ω)` (channels × channels)
  - `tcl2::TCL2Components` — `L_unitary`, `L_dissipator`, and `H_lamb_shift` with `total()` convenience
- Builders
  - `build_lamb_shift(system, kernels, imag_cutoff)` → `H_ls = Σ S_{αβ}(ω) A_α†(ω) A_β(ω)` (Hermitized)
  - `build_unitary_superop(system, H_eff)` → `-i[I⊗H_eff − H_eff^T⊗I]`
  - `build_dissipator_superop(system, kernels, gamma_cutoff, &H_ls_out)` → Kronecker-assembled Redfield/TCL2 dissipator; optionally accumulates `H_ls`
  - `build_tcl2_components(system, kernels, cutoff)` → one-shot unitary + dissipator + `H_ls`

Shapes:
- `L_*` are `NN×NN` matrices (with `NN=N*N`), `H_ls` is `N×N`

Notes:
- Builders validate bucket/channel shapes and skip negligible slices via cutoffs. Internally, adjoints/conjugates are cached to reduce allocations in tight loops.


tcl.hpp / tcl2.cpp — Stateful TCL2 generator
---------------------------------------------
Headers: `cpp/include/taco/tcl.hpp`, `cpp/src/tcl/tcl2.cpp`

What it provides:
- `taco::tcl::TCL2Generator` — maintains time, frequency buckets, kernel integrals, and applies the generator to a density matrix
- `GeneratorOptions` — frequency tolerance and quadrature settings
- Methods:
  - `reset(t0)`, `advance(t1)` (accumulate kernels via composite Simpson), `apply(rho, drho)`

Relationship to the builder:
- The stateful `TCL2Generator` is convenient for stepping propagation; the builder (`tcl2::build_*`) assembles full Liouvillians for diagnostics or batch evaluation.


spin_boson.hpp — Spin–boson model scaffolding
----------------------------------------------
Header: `cpp/include/taco/spin_boson.hpp`

What it provides:
- `Params{ Delta, epsilon, alpha, omega_c, beta, rank }`
- Builders:
  - `build_hamiltonian(params)` → `-0.5 Δ σ_x − 0.5 ε σ_z`
  - `build_jump_operators()` → `{ σ_z }`
  - `build_bath(params, N, dt)` → Ohmic `TabulatedCorrelation`
- `Model` — owns `TabulatedCorrelation` and `TCL2Generator` with correct lifetimes
  - `Model(params, N, dt, opts)` constructs everything with one call


Examples and tests
------------------
- `examples/generator_demo.cpp` — show assembling `L_unitary`/`L_dissipator` for a two-level system via `tcl2::build_tcl2_components`
- `examples/spin_boson_demo.cpp` — spin–boson propagation; writes `spin_boson_observables.csv`
- `tests/gamma_tests.cpp` — accuracy checks for Γ integrators using an analytic Ohmic model
- `tests/integrator_tests.cpp` — scalar quadrature, discrete trapz, and convolution tests
- `tests/spin_boson_tests.cpp` — regression guard for final excited-state population

