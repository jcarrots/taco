#pragma once

#include "taco/exec.hpp"
#include "taco/tcl4.hpp"

namespace taco::tcl4 {

TripleKernelSeries compute_triple_kernels_cuda(const sys::System& system,
                                               const Eigen::MatrixXcd& gamma_series,
                                               double dt,
                                               int nmax,
                                               FCRMethod method,
                                               const Exec& exec);

} // namespace taco::tcl4
