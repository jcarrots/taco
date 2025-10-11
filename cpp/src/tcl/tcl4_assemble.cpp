#include "taco/tcl4_assemble.hpp"

#include <stdexcept>

namespace taco::tcl4 {

Eigen::MatrixXcd assemble_liouvillian(const MikxTensors& tensors,
                                      const std::vector<Eigen::MatrixXcd>& /*coupling_ops*/)
{
    if (tensors.N == 0) {
        throw std::invalid_argument("assemble_liouvillian: empty tensors");
    }

    throw std::logic_error("assemble_liouvillian is not implemented yet (Phase 3 pending)");
}

} // namespace taco::tcl4

