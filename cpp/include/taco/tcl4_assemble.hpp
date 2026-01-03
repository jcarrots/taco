#pragma once

#include <vector>
#include <Eigen/Dense>

#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"

namespace taco::tcl4 {

// Assemble the raw TCL4 tensor GW in MATLAB NAKZWAN indexing:
//   GW has outer indices (n,i,m,j) and is stored as a matrix with
//     row = (n,i), col = (m,j) under column-major pair flattening.
// This matches MATLAB's `NAKZWAN_v9` output (before any G2D reshuffle).
Eigen::MatrixXcd assemble_liouvillian(const MikxTensors& tensors,
                                      const std::vector<Eigen::MatrixXcd>& coupling_ops);

// Convert GW (row=(n,i), col=(m,j)) to the Liouvillian superoperator L4 that acts on vec(rho):
//   L4(n,m,i,j) = GW(n,i,m,j)
// i.e., "reshuffle" indices (n,i,m,j) -> (n,m,i,j) by swapping the middle indices.
Eigen::MatrixXcd gw_to_liouvillian(const Eigen::MatrixXcd& GW, std::size_t N);

} // namespace taco::tcl4

