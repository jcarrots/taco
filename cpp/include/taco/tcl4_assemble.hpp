#pragma once

#include <vector>
#include <Eigen/Dense>

#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"

namespace taco::tcl4 {

Eigen::MatrixXcd assemble_liouvillian(const MikxTensors& tensors,
                                      const std::vector<Eigen::MatrixXcd>& coupling_ops);

} // namespace taco::tcl4

