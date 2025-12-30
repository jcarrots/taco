# Taco C++ Layout

```
taco/
├─ CMakeLists.txt             # build script (taco_tcl + executables/tests)
├─ configs/
│  └─ tcl_driver.yaml         # sample config for `tcl_driver`
├─ examples/
│  ├─ tcl_driver.cpp          # TCL driver (YAML system/bath config; BCF → Γ → F/C/R → MIKX → GW/L4)
│  ├─ tcl4_bench.cpp          # kernel builder benchmark / quick OpenMP sanity
│  └─ TCL4_spin_boson_example.cpp  # spin-boson TCL4 example (prints GW + L4; optional propagation)
├─ tests/
│  ├─ integrator_tests.cpp    # quadrature / convolution tests
│  ├─ gamma_tests.cpp         # Γ(ω,t) integrator accuracy check
│  ├─ spin_boson_tests.cpp    # dumps ρ(t) + Liouvillian for regression analysis
│  ├─ tcl4_tests.cpp          # compares Direct vs Convolution F/C/R builders
│  └─ tcl4_h5_compare.cpp     # MATLAB HDF5 compare tool (requires HDF5)
├─ cpp/
│  ├─ include/taco/
│  │  ├─ bath.hpp             # correlation-function interface (abstract)
│  │  ├─ bath_tabulated.hpp   # tabulated correlation implementation
│  │  ├─ bath_models.hpp      # Ohmic Drude model + helpers (J→C→Γ)
│  │  ├─ correlation_fft.hpp  # FFT-based C(t) builder (built-in radix-2)
│  │  ├─ gamma.hpp            # Γ(ω,t) integrators + streaming accumulator
│  │  ├─ ops.hpp              # operators, states, trace/norm utilities
│  │  ├─ propagate.hpp        # fixed-step propagators (RK4/expm)
│  │  ├─ rk4_dense.hpp        # dense-matrix RK4 utilities (serial/omp)
│  │  ├─ system.hpp           # eigensystem + frequency buckets + spectral slices
│  │  ├─ generator.hpp        # TCL2 superoperator builders (unitary + dissipator)
│  │  ├─ tcl2.hpp             # stateful TCL2 generator API
│  │  ├─ tcl4_kernels.hpp     # TCL4 kernel builders (F, C, R)
   │  │  ├─ tcl4_mikx.hpp        # TCL4 M/I/K/X tensor builder
│  │  ├─ tcl4_assemble.hpp    # TCL4 Liouvillian assembly (work in progress)
│  │  └─ spin_boson.hpp       # spin-boson parameters + model wrapper
│  └─ src/
│     ├─ core/
│     │  ├─ correlation.cpp   # thin TU for correlation_fft
│     │  └─ integrator.hpp    # implementation of quadrature / convolution helpers
│     └─ tcl/
│        ├─ generator.cpp           # TCL2 builder implementation (L unitary/dissipator)
│        ├─ tcl2_generator.cpp      # stateful TCL2Generator implementation
│        └─ tcl4_kernels.cpp        # F/C/R kernel time-series (discrete integrals)
    │        ├─ tcl4_mikx.cpp           # M/I/K/X assembly (explicit contractions)
│        └─ tcl4_assemble.cpp       # placeholder for TCL4 Liouvillian assembly
└─ docs/
    └─ STRUCTURE.md           # this file
    └─ PARALLEL_PLAN.md       # serial/omp/cuda/mpi_omp/mpi_cuda plan (Exec, phases, backends)
```

Key binaries after a Release build live under `build/Release/`:

```
integrator_tests.exe    # quadrature/convolution tests
gamma_tests.exe         # Γ integrator tests
spin_boson_tests.exe    # spin-boson regression dump
tcl_driver.exe          # TCL driver (YAML config; Γ FFT → F/C/R → MIKX → GW/L4)
tcl4_tests.exe          # Direct vs Convolution consistency check
tcl4_bench.exe          # kernel builder benchmark / quick OpenMP sanity
tcl4_spin_boson_example.exe  # Spin-boson TCL4 example (GW->L4 reshuffle + propagation)
tcl4_h5_compare.exe     # (optional) MATLAB HDF5 compare tool (requires HDF5)
```

Generated artifacts
-------------------
- `gamma_test_results.txt`, `integrator_test_results.txt` written by respective tests.
```
