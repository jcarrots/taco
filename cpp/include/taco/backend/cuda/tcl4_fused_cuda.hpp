#pragma once

#include <cstddef>
#include <vector>

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

// Batched fused CUDA path: compute L4 for multiple time indices with a single H2D/D2H.
// If time_indices is empty, computes all time indices [0..Nt-1].
std::vector<Eigen::MatrixXcd> build_TCL4_generator_cuda_fused_batch(const sys::System& system,
                                                                    const Eigen::MatrixXcd& gamma_series,
                                                                    double dt,
                                                                    const std::vector<std::size_t>& time_indices,
                                                                    FCRMethod method,
                                                                    const Exec& exec);

// Convenience: compute L4 for all time indices.
std::vector<Eigen::MatrixXcd> build_correction_series_cuda_fused(const sys::System& system,
                                                                 const Eigen::MatrixXcd& gamma_series,
                                                                 double dt,
                                                                 FCRMethod method,
                                                                 const Exec& exec);

} // namespace taco::tcl4
#endif
