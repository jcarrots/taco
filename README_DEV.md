Dev Loop
========

A fast local build/run loop for the C++ TCL2 core.

Prereqs
-------
- CMake 3.20+
- MSVC (Visual Studio 2022 Build Tools) on Windows, or any C++17 compiler

One-shot commands
-----------------
- Configure: `cmake -S . -B build`
- Build: `cmake --build build --config Debug -j 8`
- Run demo: `build/Debug/tcl2_demo.exe`

Scripted dev loop (Windows PowerShell)
--------------------------------------
- Build Debug: `powershell -ExecutionPolicy Bypass -File scripts/dev.ps1 -Action build -Config Debug`
- Run Debug: `powershell -ExecutionPolicy Bypass -File scripts/dev.ps1 -Action run -Config Debug`
- Clean: `powershell -ExecutionPolicy Bypass -File scripts/dev.ps1 -Action clean`

VS Code
-------
- Tasks: Terminal → Run Task (build/run debug or release)
- Debug: choose "Launch tcl2_demo (Debug)" and press F5

Notes
-----
- Library code: `cpp/include/taco/*.hpp`, `cpp/src/tcl/tcl2.cpp`
- Demo: `examples/tcl2_demo.cpp`
- Eigen is fetched automatically via CMake FetchContent.

FFTW Requirement
----------------
FFTW3 is now required for builds. Install it per platform, then configure CMake so it is found.

- Windows (vcpkg recommended)
  - Install: `vcpkg install fftw3:x64-windows`
  - Configure:
    - `cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows`
  - Notes: if using shared FFTW, ensure `fftw3.dll` is on PATH when running.

- Ubuntu/Debian
  - Install: `sudo apt-get install libfftw3-dev`
  - Configure: `cmake -S . -B build`

- macOS (Homebrew)
  - Install: `brew install fftw`
  - Configure: `cmake -S . -B build -DCMAKE_PREFIX_PATH="$(brew --prefix fftw)"`

- Custom install
  - Provide path via one of:
    - `-DCMAKE_PREFIX_PATH=C:\fftw\install` (must contain FFTW3Config.cmake)
    - `-DFFTW3_DIR=C:\fftw\install\lib\cmake\fftw3`

If configuration fails with "FFTW3 not found", install FFTW3 development files or point CMake to your install using the options above.
