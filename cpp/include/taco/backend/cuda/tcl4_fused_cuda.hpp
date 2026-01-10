#pragma once

#include <cstddef>

#include <Eigen/Dense>

#include "taco/exec.hpp"
#include "taco/system.hpp"
#include "taco/tcl4_kernels.hpp"

#ifdef TACO_HAS_CUDA
namespace taco::tcl4 {

// Fused CUDA path: compute F/C/R -> M/I/K/X -> GW -> L4 on GPU with a single final D2H copy.
Eigen::MatrixXcd build_TCL4_generator_cuda_fused(const sys::System& system,
                                                 const Eigen::MatrixXcd& gamma_series,
                                                 double dt,
                                                 std::size_t time_index,
                                                 FCRMethod method,
                                                 const Exec& exec);

} // namespace taco::tcl4
#endif
