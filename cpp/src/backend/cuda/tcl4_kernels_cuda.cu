#include "taco/backend/cuda/tcl4_kernels_cuda.hpp"

#include <stdexcept>

namespace taco::tcl4 {

TripleKernelSeries compute_triple_kernels_cuda(const sys::System& /*system*/,
                                               const Eigen::MatrixXcd& /*gamma_series*/,
                                               double /*dt*/,
                                               int /*nmax*/,
                                               FCRMethod /*method*/,
                                               const Exec& /*exec*/)
{
    throw std::runtime_error("compute_triple_kernels_cuda: not implemented");
}

} // namespace taco::tcl4

