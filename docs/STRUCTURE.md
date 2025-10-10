# Taco C++ Layout

```
taco/
├─ CMakeLists.txt             # build script (taco_tcl + executables/tests)
├─ configs/
│  └─ spin_boson.yaml         # sample configuration for the spin-boson CLI
├─ examples/
│  ├─ tcl2_demo.cpp           # legacy fixed-parameter TCL2 demo
│  ├─ generator_demo.cpp      # shows how to use tcl2::build_tcl2_components
│  └─ spin_boson.cpp          # configurable spin-boson simulator (CLI)
├─ tests/
│  ├─ integrator_tests.cpp    # quadrature / convolution tests
│  ├─ gamma_tests.cpp         # Γ(ω,t) integrator accuracy check
│  └─ spin_boson_tests.cpp    # dumps ρ(t) + Liouvillian for regression analysis
├─ cpp/
│  ├─ include/taco/
│  │  ├─ bath.hpp             # correlation-function interface (abstract)
│  │  ├─ bath_tabulated.hpp   # tabulated correlation implementation
│  │  ├─ bath_models.hpp      # Ohmic Drude model + helpers (J→C→Γ)
│  │  ├─ correlation_fft.hpp  # FFT-based C(t) builder (pocketfft/fallback)
│  │  ├─ gamma.hpp            # Γ(ω,t) integrators + streaming accumulator
│  │  ├─ ops.hpp              # operators, states, trace/norm utilities
│  │  ├─ propagate.hpp        # fixed-step propagators (RK4/expm)
│  │  ├─ system.hpp           # eigensystem + frequency buckets + spectral slices
│  │  ├─ generator.hpp        # TCL2 superoperator builders (unitary + dissipator)
│  │  ├─ tcl2.hpp             # stateful TCL2 generator API
│  │  ├─ tcl4_kernels.hpp     # TCL4 kernel builders (F, C, R)
│  │  └─ spin_boson.hpp       # spin-boson parameters + model wrapper
│  └─ src/
│     ├─ core/
│     │  ├─ correlation.cpp   # thin TU for correlation_fft
│     │  └─ integrator.hpp    # implementation of quadrature / convolution helpers
│     └─ tcl/
│        ├─ generator.cpp           # TCL2 builder implementation (L unitary/dissipator)
│        ├─ tcl2_generator.cpp      # stateful TCL2Generator implementation
│        └─ tcl4_kernels.cpp        # F/C/R kernel time-series (discrete integrals)
├─ configs/
│   └─ spin_boson.yaml        # (duplicated above for clarity)
└─ docs/
    └─ STRUCTURE.md           # this file
```

Key binaries after a Release build live under `build/Release/`:

```
spin_boson.exe          # CLI simulator (configurable)
generator_demo.exe      # Liouvillian builder example
tcl2_demo.exe           # legacy demo
integrator_tests.exe    # quadrature/convolution tests
gamma_tests.exe         # Γ integrator tests
spin_boson_tests.exe    # spin-boson regression dump
```

Generated artifacts
-------------------
- `spin_boson_observables.csv`, `spin_boson_density.csv` (created by `spin_boson.exe`).
- `gamma_test_results.txt`, `integrator_test_results.txt` written by respective tests.
```
