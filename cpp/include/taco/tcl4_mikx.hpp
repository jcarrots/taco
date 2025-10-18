#pragma once

#include <vector>
#include <Eigen/Dense>

#include "taco/tcl4.hpp"

namespace taco::tcl4 {

struct MikxTensors {
    Eigen::MatrixXcd M; // size N^2 x N^2 (flattened)
    Eigen::MatrixXcd I;
    Eigen::MatrixXcd K;
    std::vector<std::complex<double>> X; // stored as length N^6 (row-major)
    int N{0};
};

MikxTensors build_mikx(const Tcl4Map& map,
                       const TripleKernelSeries& kernels,
                       std::size_t time_index);

} // namespace taco::tcl4
