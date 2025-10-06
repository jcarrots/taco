# TACO — TCL-Accelerated Compute Orchestrator 🌮
A fast parallel and scalable time-convolutionless (TCL) runtime for open-quantum-system dynamics on CPU & GPU.

This is TCL = **Time-Convolutionless** master-equation solvers.

## Features
- CPU (OpenMP/MPI) and GPU (CUDA) backends
- TCL2/4/6 generators(manually encoded)
- TCL2n(symbolic generated and simplified once future computing power make it possible)


## Install
pip install taco-qme

## Quickstart
```python
import taco as tc
L = tc.kernels.tcl4.build_liouvillian(H, C_ops, bath, order=4)
rt = tc.runtime.Engine(backend="gpu")  # "cpu" | "gpu" | "mpi"
ρt = rt.propagate(L, ρ0, tspan, dt_adapt=True)
