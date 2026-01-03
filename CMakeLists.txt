cmake_minimum_required(VERSION 3.20)

project(taco_tcl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Optional: OpenMP for CPU parallelization
option(TACO_WITH_OPENMP "Enable OpenMP parallelization" ON)
option(TACO_WITH_CUDA "Enable CUDA backend (requires NVIDIA CUDA Toolkit)" OFF)
option(TACO_BUILD_PYTHON "Build Python extension module" ON)
option(TACO_BUILD_GAMMA_TESTS "Build gamma_tests executable" ON)

if (TACO_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
  else()
    message(WARNING "TACO_WITH_CUDA=ON but no CUDA compiler was found; install the CUDA Toolkit (nvcc) or set TACO_WITH_CUDA=OFF.")
  endif()
endif()

# Fetch Eigen (header-only)
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
)
FetchContent_MakeAvailable(eigen)

add_library(taco_tcl STATIC
  cpp/src/tcl/tcl4_kernels.cpp
  cpp/src/tcl/tcl4_mikx.cpp
  cpp/src/tcl/tcl4_assemble.cpp
  cpp/src/tcl/tcl4.cpp
  cpp/src/tcl/tcl2_generator.cpp
  cpp/src/tcl/generator.cpp
  cpp/src/core/correlation.cpp
  cpp/src/core/rk4_dense.cpp
  cpp/src/core/version.cpp
)
target_include_directories(taco_tcl PUBLIC
  cpp/include
)

  if (TACO_WITH_CUDA AND CMAKE_CUDA_COMPILER)
  target_sources(taco_tcl PRIVATE
    cpp/src/backend/cuda/tcl4_fcr_kernels.cu
    cpp/src/backend/cuda/tcl4_kernels_cuda.cu
    cpp/src/backend/cuda/tcl4_mikx_cuda.cu
  )
  target_compile_definitions(taco_tcl PUBLIC TACO_HAS_CUDA=1)
  target_link_libraries(taco_tcl PUBLIC CUDA::cudart CUDA::cufft)
endif()

if (TACO_WITH_OPENMP)
  find_package(OpenMP QUIET)
  if (OpenMP_CXX_FOUND)
    # Use PUBLIC so consumers (executables/pybind) also compile with OpenMP flags
    # (needed for `_OPENMP` and omp_* API usage in those translation units).
    target_link_libraries(taco_tcl PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for taco_tcl")
  else()
    message(STATUS "OpenMP not found; building without OpenMP")
  endif()
endif()

# Eigen provides an interface target when pulled as a subproject
if(TARGET Eigen3::Eigen)
  target_link_libraries(taco_tcl PUBLIC Eigen3::Eigen)
else()
  # Fallback: include directories only
  target_include_directories(taco_tcl PUBLIC ${eigen_SOURCE_DIR})
endif()

if (MSVC)
  target_compile_options(taco_tcl PRIVATE /W4 /permissive-)
else()
  target_compile_options(taco_tcl PRIVATE -Wall -Wextra -Wpedantic)
endif()

# FFT backend: built-in radix-2 FFT (pocketfft disabled)

if (TACO_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)

  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(_taco_native
    cpp/python/_taco_native.cpp
  )
  target_link_libraries(_taco_native PRIVATE taco_tcl)

  set_target_properties(taco_tcl PROPERTIES POSITION_INDEPENDENT_CODE ON)

  set_target_properties(_taco_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/taco"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/taco"
  )
endif()

find_package(yaml-cpp CONFIG QUIET)
if (yaml-cpp_FOUND)
  add_executable(tcl_driver examples/tcl_driver.cpp)
  target_link_libraries(tcl_driver PRIVATE taco_tcl yaml-cpp::yaml-cpp)
else()
  message(STATUS "yaml-cpp not found; skipping tcl_driver")
endif()

add_executable(tcl4_spin_boson_example examples/TCL4_spin_boson_example.cpp)
target_link_libraries(tcl4_spin_boson_example PRIVATE taco_tcl)

add_executable(tcl4_bench examples/tcl4_bench.cpp)
target_link_libraries(tcl4_bench PRIVATE taco_tcl)

add_executable(integrator_tests tests/integrator_tests.cpp)
 target_link_libraries(integrator_tests PRIVATE taco_tcl)
 target_include_directories(integrator_tests PRIVATE cpp/src)
 target_include_directories(integrator_tests PRIVATE cpp/include)

add_executable(rk4_dense_tests tests/rk4_dense_tests.cpp)
 target_link_libraries(rk4_dense_tests PRIVATE taco_tcl)
 target_include_directories(rk4_dense_tests PRIVATE cpp/include)

add_executable(spin_boson_tests tests/spin_boson_tests.cpp)
 target_link_libraries(spin_boson_tests PRIVATE taco_tcl)
 target_include_directories(spin_boson_tests PRIVATE cpp/include)

add_executable(tcl4_tests tests/tcl4_tests.cpp)
 target_link_libraries(tcl4_tests PRIVATE taco_tcl)
 target_include_directories(tcl4_tests PRIVATE cpp/include)

add_executable(tcl4_mikx_tests cpp/tests/tcl4_mikx_tests.cpp)
 target_link_libraries(tcl4_mikx_tests PRIVATE taco_tcl)
 target_include_directories(tcl4_mikx_tests PRIVATE cpp/include)

if (TACO_WITH_CUDA AND CMAKE_CUDA_COMPILER)
  add_executable(cuda_smoke tests/cuda_smoke.cu)
  target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)
endif()

if (TACO_BUILD_GAMMA_TESTS)
  add_executable(gamma_tests tests/gamma_tests.cpp)
  target_link_libraries(gamma_tests PRIVATE taco_tcl)

  # Optional: Boost (header-only) for Ei in S(omega) analytic checks
  find_package(Boost QUIET)
  if (Boost_FOUND)
    target_include_directories(gamma_tests PRIVATE ${Boost_INCLUDE_DIRS})
    target_compile_definitions(gamma_tests PRIVATE HAS_BOOST_MATH=1)
    message(STATUS "Boost found: enabling Ei-based S(omega) checks in gamma_tests")
  else()
    message(STATUS "Boost not found: skipping Ei-based S(omega) checks in gamma_tests")
  endif()
endif()

# Optional HDF5-based comparison tool for MATLAB exports.
find_package(HDF5 COMPONENTS C QUIET)
if (HDF5_FOUND)
  add_executable(tcl4_h5_compare tests/tcl4_h5_compare.cpp)
  target_link_libraries(tcl4_h5_compare PRIVATE taco_tcl)
  target_include_directories(tcl4_h5_compare PRIVATE ${HDF5_INCLUDE_DIRS})
  if (TARGET HDF5::C)
    target_link_libraries(tcl4_h5_compare PRIVATE HDF5::C)
  elseif (TARGET HDF5::HDF5)
    target_link_libraries(tcl4_h5_compare PRIVATE HDF5::HDF5)
  else()
    target_link_libraries(tcl4_h5_compare PRIVATE ${HDF5_LIBRARIES})
  endif()
else()
  message(STATUS "HDF5 not found; skipping tcl4_h5_compare")
endif()


