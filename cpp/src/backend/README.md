Backend scaffolding
===================

This folder will host implementation files for parallel backends.

- cpu/: CPU parallel code paths (OpenMP/TBB) for TCL4 triple kernels and assembly.
- gpu/: GPU implementations (cuFFT/cuBLAS/hipFFT/rocBLAS) for scalar F/C/R kernels.
- hybrid/: Streaming/overlap utilities to pipeline GPU compute with CPU assembly.

No code here yet — this is a placeholder to stage work without impacting the core.

