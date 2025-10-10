#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

namespace taco::tcl4 {

enum class SpectralOp {
    Identity,
    Transpose,
    Conjugate,
    Hermitian
};

struct FCRSeries {
    std::vector<Eigen::MatrixXcd> F;  // Γ4 kernel F(t)
    std::vector<Eigen::MatrixXcd> C;  // Γ4 kernel C(t)
    std::vector<Eigen::MatrixXcd> R;  // Γ4 kernel R(t)
};

FCRSeries compute_FCR_time_series(const std::vector<Eigen::MatrixXcd>& G1,
                                  const std::vector<Eigen::MatrixXcd>& G2,
                                  double omega,
                                  double dt,
                                  SpectralOp op2 = SpectralOp::Transpose);

} // namespace taco::tcl4

