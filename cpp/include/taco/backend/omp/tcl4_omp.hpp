#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "taco/exec.hpp"
#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_kernels.hpp"

namespace taco::tcl4 {

// OpenMP-oriented CPU orchestration for triple-kernel construction.
TripleKernelSeries compute_triple_kernels_omp(const sys::System& system,
                                              const Eigen::MatrixXcd& gamma_series,
                                              double dt,
                                              int nmax,
                                              FCRMethod method,
                                              Exec exec);

// OpenMP-oriented CPU orchestration for assembling full L4(t) series from precomputed kernels.
std::vector<Eigen::MatrixXcd> build_correction_series_omp(const sys::System& system,
                                                          const TripleKernelSeries& kernels,
                                                          const Tcl4Map& map,
                                                          std::size_t Nt,
                                                          Exec exec);

} // namespace taco::tcl4

