# TACO Python

The Python package `taco` is a thin wrapper over the existing C++ CPU and CUDA implementations (no physics/numerics are redesigned in Python).

## What’s wired (today)

- **TCL simulation**: `taco.tcl.simulate(H, A, bath, cfg, rho0, device=...)`
  - **CPU**: uses the existing C++ CPU TCL2/TCL4 builders and **dense RK4** propagation.
  - **CUDA (if built with CUDA)**: uses the existing fused CUDA TCL4 builder and **dense RK4 on CUDA** for propagation.
  - Inputs/outputs are **host NumPy arrays**; CUDA uses internal H2D/D2H copies.
- **BCF precompute**: `taco.tcl.precompute_bcf(bath, dt)` (optional, for reuse across sweeps).
- **E2E benchmark helper** (mirrors `tcl4_e2e_cuda_compare` workload):
  - `taco.tcl.e2e_cuda_compare_spin_boson(...)` (timings + CPU/GPU diffs; use `check=False` for perf-only FP32 runs).

## Examples

- Notebook: `python/examples/tcl4_e2e_cuda_compare.ipynb` (defines a spin-boson model + bath + simulation parameters, runs `simulate()` on CPU/CUDA, then runs the E2E compare).
- Script: `python/examples/spin_boson_simulate.py` (minimal CLI-style spin-boson run).

## RK4 in Python

RK4 is **not a separate Python-callable integrator**; it is used internally by `taco.tcl.simulate(...)` for time propagation:

- CPU path calls `taco::tcl::rk4_update_serial(...)`
- CUDA path calls `taco::tcl::rk4_update_cuda(...)` (FP64) or `taco::tcl::rk4_update_cuda_f32(...)` (FP32)

So: **yes, RK4 is wired in the Python pipeline**, but **only as the default propagation method inside `simulate()`**.

## Install / build

### CPU-only

```powershell
pip install .
```

### CUDA + Python

```powershell
$env:CMAKE_ARGS='-DTACO_BUILD_CUDA=ON -DTACO_BUILD_PYTHON=ON -DCMAKE_CUDA_ARCHITECTURES=native'
pip install .
```

### Building for a specific Python (important for VS Code / Jupyter)

The native module is ABI-tagged (e.g. `...cp310...pyd`, `...cp39...pyd`) and **must match the Python used by your notebook/kernel**.

If you build with CMake directly (Visual Studio generator), force the Python interpreter:

```powershell
cmake -S . -B build-cuda-vs-conda39 -G "Visual Studio 17 2022" -A x64 `
  -DTACO_WITH_CUDA=ON -DTACO_BUILD_PYTHON=ON `
  -DPython_EXECUTABLE=C:\Users\59405\anaconda3\python.exe

cmake --build build-cuda-vs-conda39 --config Release --target _taco
```

Then make sure your VS Code notebook kernel uses the same Python version.

## Tests

```powershell
pytest -q
```
