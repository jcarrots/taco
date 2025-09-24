# TACO — TCL-Accelerated Compute Orchestrator 🌮
A fast parallel time-convolutionless (TCL) runtime for open-quantum-system dynamics on CPU & GPU.

This is TCL = **Time-Convolutionless** master-equation solvers.

## Features
- CPU (OpenMP) and GPU (CUDA/ROCm) backends
- TCL2/4/6 generators, rTCL hooks, adaptive timestepping
- MPI/distributed sweeps, checkpointing, metrics

## Install
pip install taco-qme

## Quickstart
```python
import taco as tc
L = tc.kernels.tcl4.build_liouvillian(H, C_ops, bath, order=4)
rt = tc.runtime.Engine(backend="gpu")  # "cpu" | "gpu" | "mpi"
ρt = rt.propagate(L, ρ0, tspan, dt_adapt=True)
