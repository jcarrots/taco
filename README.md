# TACO — TCL-Accelerated Compute Orchestrator 🌮
This is TCL = **Time-Convolutionless** master-equation solvers.
A fast parallel and scalable time-convolutionless (TCL) runtime with C++ backend for open-quantum-system dynamics. TACO backends: serial, omp, cuda, mpi_omp, mpi_cuda.

## Features
- Backends: serial, omp, cuda, mpi_omp, mpi_cuda
- TCL2 generator (stateful) + Liouvillian builders
- TCL4 kernels and assembly in seconds
- Higher‑order TCL (TCL6/TCL2n) planning in docs; symbolic road‑map.

## Install
pip install taco-qme

## Quickstart
```python
import taco as tc
L = tc.kernels.tcl4.build_liouvillian(H, C_ops, bath, order=4)
rt = tc.runtime.Engine(backend="cuda")  # "serial" | "omp" | "cuda" | "mpi_omp" | "mpi_cuda"
ρt = rt.propagate(L, ρ0, tspan, dt_adapt=True)

```

## Build from source (C++)
- Configure: `cmake -S . -B build`
- Build (Release): `cmake --build build --config Release`
- Enable Python extension: `-DTACO_BUILD_PYTHON=ON` (add `-DPython_EXECUTABLE=...` if needed)
- Disable gamma tests: `-DTACO_BUILD_GAMMA_TESTS=OFF`

## Python extension
- Configure: `cmake -S . -B build -DTACO_BUILD_PYTHON=ON -DPython_EXECUTABLE=...`
- Build: `cmake --build build --config Release --target _taco_native`
- Import test: `python -c "import sys; sys.path.insert(0,'python'); import taco; print(taco.version())"`
- Windows output: `_taco_native.pyd` lands under `python/taco/Release` or `python/taco/Debug`

## TCL4 Demo & Test
- Demo driver: `tcl_driver` loads a YAML config (matrix `H`, `A` and `J_expr`) and runs TCL4 assembly
  - Build: `cmake --build build --config Release --target tcl_driver` (requires `yaml-cpp`)
  - Run (Win): `build\Release\tcl_driver.exe --config=configs\tcl_driver.yaml`
- Test: `tcl4_tests` compares Direct vs Convolution F/C/R
  - Build: `cmake --build build --config Release --target tcl4_tests`
  - Run: `build\Release\tcl4_tests.exe`
