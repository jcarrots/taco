cmake_minimum_required(VERSION 3.30.0-rc3)
set(CMAKE_MODULE_PATH "C:/Users/59405/taco/build/_deps/eigen-src/cmake")
project(CheckFortran Fortran)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
  "set(CMAKE_Fortran_COMPILER \"${CMAKE_Fortran_COMPILER}\")\n"
  
  )