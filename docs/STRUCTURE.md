# Taco C++ Layout

```
taco/
|-- CMakeLists.txt             # build script (taco_tcl + executables/tests)
|-- configs/
|   `-- tcl_driver.yaml         # sample config for `tcl_driver`
|-- examples/
|   |-- tcl_driver.cpp          # TCL driver (YAML system/bath config; BCF -> Gamma -> F/C/R -> MIKX -> GW/L4)
|   |-- tcl4_bench.cpp          # kernel builder benchmark / quick OpenMP sanity
|   `-- TCL4_spin_boson_example.cpp  # spin-boson TCL4 example (prints GW + L4; optional propagation)
|-- tests/
|   |-- integrator_tests.cpp    # quadrature / convolution tests
|   |-- gamma_tests.cpp         # Gamma(omega,t) integrator accuracy check
|   |-- spin_boson_tests.cpp    # dumps rho(t) + Liouvillian for regression analysis
|   |-- tcl4_tests.cpp          # compares Direct vs Convolution F/C/R builders
|   |-- tcl4_h5_compare.cpp     # MATLAB HDF5 compare tool (requires HDF5)
|   |-- tcl4_mpi_omp_tests.cpp  # MPI+OpenMP TCL4 smoke test (requires MPI build)
|   `-- tcl4_e2e_cuda_compare.cpp  # CPU vs CUDA end-to-end compare
|-- cpp/
|   |-- include/taco/
|   |   |-- bath.hpp             # correlation-function interface (abstract)
|   |   |-- bath_tabulated.hpp   # tabulated correlation implementation
|   |   |-- bath_models.hpp      # Ohmic Drude model + helpers (J->C->Gamma)
|   |   |-- correlation_fft.hpp  # FFT-based C(t) builder (built-in radix-2)
|   |   |-- exec.hpp             # backend selector and execution hints
|   |   |-- gamma.hpp            # Gamma(omega,t) integrators + streaming accumulator
|   |   |-- generator.hpp        # TCL2 superoperator builders (unitary + dissipator)
|   |   |-- ops.hpp              # operators, states, trace/norm utilities
|   |   |-- propagate.hpp        # fixed-step propagators (RK4/expm)
|   |   |-- rk4_dense.hpp        # dense-matrix RK4 utilities (serial/omp)
|   |   |-- spin_boson.hpp       # spin-boson parameters + model wrapper
|   |   |-- system.hpp           # eigensystem + frequency buckets + spectral slices
|   |   |-- tcl2.hpp             # stateful TCL2 generator API
|   |   |-- tcl4.hpp             # TCL4 high-level wrappers + rebuild helpers
|   |   |-- tcl4_assemble.hpp    # TCL4 Liouvillian assembly (NAKZWAN)
|   |   |-- tcl4_kernels.hpp     # TCL4 kernel builders (F, C, R)
|   |   |-- tcl4_mikx.hpp        # TCL4 M/I/K/X tensor builder
|   |   `-- backend/
|   |       |-- cpu/             # CPU backend headers (MPI+OpenMP)
|   |       `-- cuda/            # CUDA TCL4 headers
|   |-- src/
|   |   |-- core/
|   |   |   |-- correlation.cpp   # thin TU for correlation_fft
|   |   |   `-- integrator.hpp    # implementation of quadrature / convolution helpers
|   |   |-- backend/
|   |   |   |-- README.md         # backend overview
|   |   |   |-- cpu/              # CPU backend implementation (MPI+OpenMP TCL4 batch)
|   |   |   |-- serial/           # serial backend notes
|   |   |   |-- omp/              # OpenMP backend notes
|   |   |   |-- cuda/             # CUDA backend implementation
|   |   |   |-- mpi_omp/           # MPI+OpenMP planning notes
|   |   |   `-- mpi_cuda/          # MPI+CUDA placeholders
|   |   `-- tcl/
|   |       |-- generator.cpp      # TCL2 builder implementation (L unitary/dissipator)
|   |       |-- tcl2_generator.cpp # stateful TCL2Generator implementation
|   |       |-- tcl4_kernels.cpp   # F/C/R kernel time-series (discrete integrals)
|   |       |-- tcl4_mikx.cpp      # M/I/K/X assembly (explicit contractions)
|   |       `-- tcl4_assemble.cpp  # TCL4 Liouvillian assembly (NAKZWAN)
|-- docs/
|   |-- STRUCTURE.md           # this file
|   |-- TCL4_PLAN.md           # TCL4 implementation plan + status
|   `-- PARALLEL_PLAN.md       # serial/omp/cuda/mpi_omp/mpi_cuda plan (Exec, phases, backends)
```

Key binaries after a Release build live under `build/Release/`:

```
integrator_tests.exe    # quadrature/convolution tests
gamma_tests.exe         # Gamma integrator tests
spin_boson_tests.exe    # spin-boson regression dump
tcl_driver.exe          # TCL driver (YAML config; Gamma FFT -> F/C/R -> MIKX -> GW/L4)
tcl4_tests.exe          # Direct vs Convolution consistency check
tcl4_bench.exe          # kernel builder benchmark / quick OpenMP sanity
tcl4_spin_boson_example.exe  # Spin-boson TCL4 example (GW->L4 reshuffle + propagation)
tcl4_h5_compare.exe     # (optional) MATLAB HDF5 compare tool (requires HDF5)
tcl4_e2e_cuda_compare.exe  # CPU vs CUDA end-to-end compare
tcl4_mpi_omp_tests.exe  # (optional) MPI+OpenMP TCL4 smoke test (requires MPI)
```

Generated artifacts
-------------------
- `gamma_test_results.txt`, `integrator_test_results.txt` written by respective tests.
