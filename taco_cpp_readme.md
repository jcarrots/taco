taco/cpp/
├─ include/taco/                  # **Public API headers** (installable)
│  ├─ taco.h                      # C ABI (opaque handles)  ← stable
│  ├─ config.hpp                  # feature macros (exported)
│  ├─ types.hpp                   # internal C++ types (PODs, enums)
│  ├─ model.hpp                   # Model, Operator, Bath specs
│  ├─ bath.hpp                    # correlation builders/interfaces
│  ├─ tcl.hpp                     # TCL2/4/6 build interfaces
│  ├─ step.hpp                    # steppers (rk45/etd/krylov) interfaces
│  ├─ linalg.hpp                  # BLAS/CSR helpers, small dense ops
│  ├─ cpu.hpp                     # CPU backend interface
│  ├─ gpu.hpp                     # GPU backend interface (streams, handles)
│  ├─ mpi.hpp                     # MPI runtime interface
│  ├─ io.hpp                      # HDF5 writer interfaces
│  └─ util.hpp                    # logging, timers, error macros
├─ src/                           # **Implementation** (not installed)
│  ├─ core/
│  │  ├─ context.cpp              # taco_ctx lifecycle, feature flags
│  │  ├─ errors.cpp               # status strings, last-error ring
│  │  └─ api_c.cpp                # C ABI entry points (wrap C++)
│  ├─ ops/
│  │  ├─ pauli.cpp                # sx/sy/sz, kron builders
│  │  └─ csr_utils.cpp
│  ├─ bath/
│  │  ├─ drude_lorentz.cpp
│  │  ├─ ohmic.cpp
│  │  └─ custom_ct.cpp
│  ├─ tcl/
│  │  ├─ tcl2.cpp
│  │  ├─ tcl4.cpp
│  │  └─ tcl6.cpp                 # (stub until ready)
│  ├─ step/
│  │  ├─ rk45.cpp                 # PID controller, adapt logic
│  │  ├─ etd.cpp                  # (optional later)
│  │  └─ krylov.cpp               # (optional later)
│  ├─ cpu/
│  │  ├─ lmatvec.cpp              # Liouville matvec (OpenMP)
│  │  └─ propagate.cpp            # CPU loop, observables
│  ├─ gpu/
│  │  ├─ context.cu               # device selection, streams, handles
│  │  ├─ lmatvec.cu               # Liouville matvec (cuBLAS/cuSPARSE)
│  │  └─ propagate.cu             # device-resident trajectory
│  ├─ mpi/
│  │  ├─ farm.cpp                 # task queue, work stealing
│  │  └─ reduce.cpp               # metrics/BLP reductions
│  ├─ io/
│  │  ├─ hdf5_writer.cpp
│  │  └─ csv_metrics.cpp
│  └─ util/
│     ├─ log.cpp                  # spdlog wrappers
│     └─ timers.cpp               # chrono/PAPI/NVTX helpers
├─ bindings/pybind11/
│  └─ pybind_taco.cpp             # exposes minimal API to Python
└─ tests/                         # C++ unit tests (Catch2/GoogleTest)
   ├─ test_tcl2.cpp
   ├─ test_rk45.cpp
   └─ test_io.cpp
