Serial backend
==============

Status
------
- Source: `cpp/src/backend/serial/tcl4_serial.cpp`
- Header: `cpp/include/taco/backend/serial/tcl4_serial.hpp`
- CPU reference path for deterministic debugging.
- `compute_triple_kernels` and `build_correction_series` dispatch to serial backend orchestration when `Exec.backend=Serial`.
- `build_mikx_serial` and `assemble_liouvillian` provide the single-thread baseline.
