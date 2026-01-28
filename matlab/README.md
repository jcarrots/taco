# MATLAB implementation (reference / prototyping)

This folder contains a MATLAB reference implementation of parts of the TCL4 pipeline used in TACO (Time-Convolutionless master-equation solvers). It is primarily useful for **prototyping**, **sanity checks**, and **derivation validation**; the production CPU/CUDA implementations live under `cpp/` and are exposed to Python under `python/`.

## What’s in this folder

- `bcf_fft_ohmic_simple.m`
  - Computes a bath correlation function `C(t)` for an Ohmic spectral density `J(ω)=ω exp(-ω/ωc)` using an FFT-based construction.
- `tcl4_kernels.m`
  - Computes the TCL4 intermediate kernels `F(t)`, `C(t)`, `R(t)` from time series inputs using prefix integrals and (causal) convolutions.
  - Uses `pagemtimes` when available and falls back to a loop otherwise.
- `MIKX.m`
  - Assembles the intermediate objects `M, I, K, X` used downstream in the TCL4 generator construction.
  - Uses MATLAB’s `tensorprod` if available (see requirements below).
- `NAKZWAN_v9.m`
  - Builds the TCL4 Liouvillian from `M, I, K, X` and coupling operators.
- `tcl4_driver.m`
  - Example script that wires together a full pipeline (BCF → F/C/R time series → M/I/K/X → generator).
  - **Note:** this driver references helper functions that are not currently included in this repo (see “Known gaps”).

## Requirements

- MATLAB R2016b+ (used by implicit expansion in several places).
- For best performance: MATLAB R2020b+ (provides `pagemtimes`).
- `MIKX.m` uses `tensorprod`. If your MATLAB does not provide `tensorprod`, you must:
  - upgrade MATLAB, or
  - supply a compatible `tensorprod` implementation on your MATLAB path.

## Quick start

From the repo root in MATLAB:

```matlab
addpath(fullfile(pwd, "matlab"));
```

### Bath correlation function (Ohmic, FFT-based)

```matlab
beta = 0.5;      % inverse temperature
dt   = 1e-2;     % time step
T    = 10.0;     % end time for returned C(t)
omegac = 5.0;    % cutoff frequency

[C, t, meta] = bcf_fft_ohmic_simple(beta, dt, T, omegac);
plot(t, real(C)); xlabel("t"); ylabel("Re C(t)");
```

### TCL4 kernel construction (F/C/R)

`tcl4_kernels.m` expects time series inputs `G1(t)`, `G2(t)` (scalar or matrix-valued) on a uniform grid with spacing `dt`, plus a scalar `Omega`.

```matlab
N = 1024;
dt = 1e-2;
t = (0:N-1).'*dt;

Omega = 1.0;
G1 = exp(-t);                 % example scalar time series
G2 = exp(-2*t) .* (1+1i);     % example scalar time series

[F, C, R] = tcl4_kernels(G1, G2, Omega, dt, "T");
```

## Relationship to the C++/Python implementation

- The repository’s supported, end-to-end workflows are:
  - C++ tools under `tests/` (e.g. `tcl4_e2e_cuda_compare`), and
  - Python API under `python/` (e.g. `taco.tcl.simulate(...)` and `python/examples/tcl4_e2e_cuda_compare.ipynb`).

## Known gaps (driver dependencies)

`matlab/tcl4_driver.m` currently references helper functions that are **not** in this repository:

- `bcfFT_v1`
- `compute_FCR_timeseries`
- `getAsymptoticALL`
- `G2D`

