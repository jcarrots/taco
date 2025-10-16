#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

namespace taco::tcl4 {

// Method selector for F/C/R kernel construction
enum class FCRMethod {
    Convolution, // default
    Direct
};

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

// -------- Scalar-series (1×1 per time) fast-path --------
// Operate directly on Γ columns as vectors to avoid building vectors of 1×1 matrices.
Eigen::VectorXcd compute_F_series_direct(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                         const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                         double omega,
                                         double dt);

Eigen::VectorXcd compute_C_series_direct(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                         const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                         double omega,
                                         double dt);

Eigen::VectorXcd compute_R_series_direct(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                         const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                         double omega,
                                         double dt);

Eigen::VectorXcd compute_F_series(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                  double omega,
                                  double dt,
                                  FCRMethod method = FCRMethod::Convolution);

Eigen::VectorXcd compute_C_series(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                  double omega,
                                  double dt,
                                  FCRMethod method = FCRMethod::Convolution);

Eigen::VectorXcd compute_R_series(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                  double omega,
                                  double dt,
                                  FCRMethod method = FCRMethod::Convolution);

// -------- Matrix-series (general multi-channel) path --------
// Split series builders (Direct path)
std::vector<Eigen::MatrixXcd> compute_F_series_direct(const std::vector<Eigen::MatrixXcd>& G1,
                                                      const std::vector<Eigen::MatrixXcd>& G2,
                                                      double omega,
                                                      double dt);

std::vector<Eigen::MatrixXcd> compute_C_series_direct(const std::vector<Eigen::MatrixXcd>& G1,
                                                      const std::vector<Eigen::MatrixXcd>& G2_conj,
                                                      double omega,
                                                      double dt);

std::vector<Eigen::MatrixXcd> compute_R_series_direct(const std::vector<Eigen::MatrixXcd>& G1,
                                                      const std::vector<Eigen::MatrixXcd>& G2,
                                                      double omega,
                                                      double dt);

// Direct (time-domain) method
FCRSeries compute_FCR_time_series_direct(const std::vector<Eigen::MatrixXcd>& G1,
                                         const std::vector<Eigen::MatrixXcd>& G2,
                                         double omega,
                                         double dt,
                                         SpectralOp op2 = SpectralOp::Transpose);

// Convolution/FFT method (fast path). Initially delegates to direct until optimized path is implemented.
FCRSeries compute_FCR_time_series_convolution(const std::vector<Eigen::MatrixXcd>& G1,
                                              const std::vector<Eigen::MatrixXcd>& G2,
                                              double omega,
                                              double dt,
                                              SpectralOp op2 = SpectralOp::Transpose);

// Unified wrapper with method selection (default = Convolution)
FCRSeries compute_FCR_time_series(const std::vector<Eigen::MatrixXcd>& G1,
                                  const std::vector<Eigen::MatrixXcd>& G2,
                                  double omega,
                                  double dt,
                                  SpectralOp op2 = SpectralOp::Transpose,
                                  FCRMethod method = FCRMethod::Convolution);

// Split series builders (method-selecting wrappers)
std::vector<Eigen::MatrixXcd> compute_F_series(const std::vector<Eigen::MatrixXcd>& G1,
                                               const std::vector<Eigen::MatrixXcd>& G2,
                                               double omega,
                                               double dt,
                                               FCRMethod method = FCRMethod::Convolution);

std::vector<Eigen::MatrixXcd> compute_C_series(const std::vector<Eigen::MatrixXcd>& G1,
                                               const std::vector<Eigen::MatrixXcd>& G2_conj,
                                               double omega,
                                               double dt,
                                               FCRMethod method = FCRMethod::Convolution);

std::vector<Eigen::MatrixXcd> compute_R_series(const std::vector<Eigen::MatrixXcd>& G1,
                                               const std::vector<Eigen::MatrixXcd>& G2,
                                               double omega,
                                               double dt,
                                               FCRMethod method = FCRMethod::Convolution);

} // namespace taco::tcl4
