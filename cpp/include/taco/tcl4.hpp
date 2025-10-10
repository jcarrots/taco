#pragma once

#include <vector>

#include <Eigen/Dense>

#include "taco/system.hpp"

namespace taco::tcl4 {

struct Tcl4Map {
    int N{0};
    int nf{0};
    std::vector<double> time_grid;
    std::vector<double> omegas;
    Eigen::MatrixXi pair_to_freq;
    std::vector<std::pair<int,int>> freq_to_pair;
};

Tcl4Map build_map(const sys::System& system, const std::vector<double>& time_grid);

struct TripleKernelSeries {
    std::vector<std::vector<std::vector<Eigen::VectorXcd>>> F;
    std::vector<std::vector<std::vector<Eigen::VectorXcd>>> C;
    std::vector<std::vector<std::vector<Eigen::VectorXcd>>> R;
};

TripleKernelSeries compute_triple_kernels(const sys::System& system,
                                          const Eigen::MatrixXcd& gamma_series,
                                          double dt,
                                          int nmax);

} // namespace taco::tcl4

