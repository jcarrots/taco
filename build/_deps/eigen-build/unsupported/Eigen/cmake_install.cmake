# Install script for directory: C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/taco_tcl")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/AdolcForward"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/AlignedVector3"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/ArpackSupport"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/AutoDiff"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/BVH"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/EulerAngles"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/FFT"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/IterativeSolvers"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/KroneckerProduct"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/LevenbergMarquardt"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/MatrixFunctions"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/MoreVectorization"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/MPRealSupport"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/NonLinearOptimization"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/NumericalDiff"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/OpenGLSupport"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/Polynomials"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/Skyline"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/SparseExtra"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/SpecialFunctions"
    "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "C:/Users/59405/taco/build/_deps/eigen-src/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/59405/taco/build/_deps/eigen-build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

