# TACO — TCL-Accelerated Compute Orchestrator 🌮
This is TCL = **Time-Convolutionless** master-equation solvers.
A fast parallel and scalable time-convolutionless (TCL) runtime with C++ backend for open-quantum-system dynamics. TACO features two parallel choices (CPU only, hybrid CPU+GPU) across nodes, and can work in distributed system. 

## Features
- CPU (OpenMP/MPI) and GPU (CUDA) backends
- TCL2 generator (stateful) + Liouvillian builders
- TCL4 kernels and assembly:
  - Unique‑frequency Γ(ω,t) series (bucket‑major)
  - F/C/R builders with FFT‑based causal convolution for scalar Γ (default)
  - M/I/K/X assembly (MIKX) and Liouvillian assembly (NAKZWAN)
  - Helpers to rebuild Γ (N×N) and F/C/R (N^6) at selected times
- Higher‑order TCL (TCL6/TCL2n) planning in docs; symbolic road‑map

## Install
pip install taco-qme

## Quickstart
```python
import taco as tc
L = tc.kernels.tcl4.build_liouvillian(H, C_ops, bath, order=4)
rt = tc.runtime.Engine(backend="gpu")  # "cpu" | "gpu" | "mpi"
ρt = rt.propagate(L, ρ0, tspan, dt_adapt=True)
