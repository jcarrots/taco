cmake_minimum_required(VERSION 3.20)

project(taco_tcl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Optional: OpenMP for CPU parallelization
option(TACO_WITH_OPENMP "Enable OpenMP parallelization" ON)

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
  cpp/src/core/correlation.cpp
  cpp/src/tcl/generator.cpp
)
target_include_directories(taco_tcl PUBLIC
  cpp/include
)

if (TACO_WITH_OPENMP)
  find_package(OpenMP QUIET)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(taco_tcl PRIVATE OpenMP::OpenMP_CXX)
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

# FFT backend: prefer pocketfft (BSD); fall back to built-in radix-2
find_package(pocketfft CONFIG QUIET)
if (TARGET pocketfft::pocketfft)
  target_link_libraries(taco_tcl PRIVATE pocketfft::pocketfft)
  target_compile_definitions(taco_tcl PRIVATE TACO_USE_POCKETFFT=1)
elseif (EXISTS "${CMAKE_SOURCE_DIR}/third_party/pocketfft/pocketfft_hdronly.h")
  target_include_directories(taco_tcl PRIVATE ${CMAKE_SOURCE_DIR}/third_party/pocketfft)
  target_compile_definitions(taco_tcl PRIVATE TACO_USE_POCKETFFT=1)
else()
  message(STATUS "pocketfft not found; using built-in radix-2 FFT")
endif()

add_executable(tcl2_demo examples/tcl2_demo.cpp)
target_link_libraries(tcl2_demo PRIVATE taco_tcl)

add_executable(generator_demo examples/generator_demo.cpp)
target_link_libraries(generator_demo PRIVATE taco_tcl)

add_executable(spin_boson examples/spin_boson.cpp)
target_link_libraries(spin_boson PRIVATE taco_tcl)

add_executable(tcl4_driver examples/tcl4_driver.cpp)
target_link_libraries(tcl4_driver PRIVATE taco_tcl)

add_executable(spin_boson_tcl4 examples/spin_boson_tcl4.cpp)
target_link_libraries(spin_boson_tcl4 PRIVATE taco_tcl)

add_executable(tcl4_bench examples/tcl4_bench.cpp)
target_link_libraries(tcl4_bench PRIVATE taco_tcl)

add_executable(integrator_tests tests/integrator_tests.cpp)
 target_link_libraries(integrator_tests PRIVATE taco_tcl)
 target_include_directories(integrator_tests PRIVATE cpp/src)
 target_include_directories(integrator_tests PRIVATE cpp/include)


add_executable(gamma_tests tests/gamma_tests.cpp)
 target_link_libraries(gamma_tests PRIVATE taco_tcl)

add_executable(spin_boson_tests tests/spin_boson_tests.cpp)
 target_link_libraries(spin_boson_tests PRIVATE taco_tcl)
 target_include_directories(spin_boson_tests PRIVATE cpp/include)

add_executable(tcl4_tests tests/tcl4_tests.cpp)
 target_link_libraries(tcl4_tests PRIVATE taco_tcl)
 target_include_directories(tcl4_tests PRIVATE cpp/include)

# Optional: Boost (header-only) for Ei in S(omega) analytic checks
find_package(Boost QUIET)
if (Boost_FOUND)
  target_include_directories(gamma_tests PRIVATE ${Boost_INCLUDE_DIRS})
  target_compile_definitions(gamma_tests PRIVATE HAS_BOOST_MATH=1)
  message(STATUS "Boost found: enabling Ei-based S(omega) checks in gamma_tests")
else()
  message(STATUS "Boost not found: skipping Ei-based S(omega) checks in gamma_tests")
endif()
