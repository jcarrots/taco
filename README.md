# TACO - TCL-Accelerated Compute Orchestrator
This is TCL = **Time-Convolutionless** master-equation solvers.
A fast parallel and scalable time-convolutionless (TCL) runtime with C++ backend for open-quantum-system dynamics. TACO backends: serial, omp, cuda, mpi_omp, mpi_cuda.

## Features
- Backends: serial, omp, cuda, mpi_omp, mpi_cuda
- TCL2 generator + Liouvillian builders
- TCL4 kernels and assembly in seconds
- Higher-order TCL (TCL6/TCL2n) planning in docs; symbolic road-map.

## Install
pip install taco-qme

## Quickstart
```python
import taco as tc
L = tc.kernels.tcl4.build_liouvillian(H, C_ops, bath, order=4)
rt = tc.runtime.Engine(backend="cuda")  # "serial" | "omp" | "cuda" (MPI backends are C++-only today)
rho_t = rt.propagate(L, rho0, tspan, dt_adapt=True)
```

## Build from source (C++)
- Configure: `cmake -S . -B build`
- Build (Release): `cmake --build build --config Release`
- Enable MPI (distributed CPU): `-DTACO_WITH_MPI=ON` (requires MPI)
- Enable Python extension: `-DTACO_BUILD_PYTHON=ON` (add `-DPython_EXECUTABLE=...` if needed)
- Disable gamma tests: `-DTACO_BUILD_GAMMA_TESTS=OFF`

## CUDA backend (C++, performance-focused)
- Build: `cmake -S . -B build-cuda -DTACO_WITH_CUDA=ON` then `cmake --build build-cuda --config Release`
- Implementation highlights:
  - F/C/R construction uses cuFFT + CUB scans (`compute_triple_kernels_cuda`).
  - Fused end-to-end L4 builders keep intermediates on device and copy L4 back in one transfer:
    - `build_TCL4_generator_cuda_fused(...)` (single time index)
    - `build_TCL4_generator_cuda_fused_batch(...)` (multiple time indices)
  - CUDA Graphs can capture/replay the fixed MIKX -> GW -> L4 launch sequence to reduce host launch overhead:
    - Disable with `TCL4_USE_CUDA_GRAPH=0`
    - Diagnostics with `TCL4_CUDA_GRAPH_VERBOSE=1`
- CPU vs CUDA compare tool: `tcl4_e2e_cuda_compare`
  - Build: `cmake --build build-cuda --config Release --target tcl4_e2e_cuda_compare`
  - Run (PowerShell): `.\build-cuda\Release\tcl4_e2e_cuda_compare.exe --N=200000 --tidx=0:1:10000 --gpu_warmup=1 --threads=8`
- More details: `cpp/src/backend/cuda/README.md`

## MPI + OpenMP (CPU over distributed memory system, experimental)
- C++ API: `taco/backend/cpu/tcl4_mpi_omp.hpp` (`build_TCL4_generator_cpu_mpi_omp_batch`).
- Rank 0 returns the gathered `L4(t)` vector; non-root ranks return `{}`.

## Python extension(Under development)
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
- Test (MPI, optional): `tcl4_mpi_omp_tests`
  - Build: `cmake --build build --config Release --target tcl4_mpi_omp_tests`
  - Run: `mpiexec -n 4 build\Release\tcl4_mpi_omp_tests.exe`
