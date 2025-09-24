#include "taco/tcl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Eigenvalues>

namespace taco::tcl {

namespace {

inline std::size_t pair_index(std::size_t alpha, std::size_t beta, std::size_t width) noexcept {
    return alpha * width + beta;
}

inline bool nearly_equal(double a, double b, double tol) noexcept {
    return std::abs(a - b) <= tol;
}

}  // namespace

TCL2Generator::TCL2Generator(const Eigen::MatrixXcd& hamiltonian,
                             std::vector<JumpOperator> jump_ops,
                             const bath::CorrelationFunction& correlation,
                             const GeneratorOptions& options)
    : bath_(correlation), opts_(options) {
    if (hamiltonian.rows() != hamiltonian.cols()) {
        throw std::invalid_argument("Hamiltonian must be square");
    }

    dim_ = static_cast<std::size_t>(hamiltonian.rows());
    if (dim_ == 0) {
        throw std::invalid_argument("Hamiltonian dimension must be positive");
    }

    lab_jump_ops_ = std::move(jump_ops);
    jump_ops_ = lab_jump_ops_.size();
    if (jump_ops_ == 0) {
        throw std::invalid_argument("At least one jump operator is required");
    }

    if (bath_.rank() != jump_ops_) {
        throw std::invalid_argument("Bath correlation rank must match number of jump operators");
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hamiltonian);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to diagonalize Hamiltonian");
    }

    eps_ = solver.eigenvalues();
    U_ = solver.eigenvectors();
    U_dag_ = U_.adjoint();

    H_ls_eig_ = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                       static_cast<Eigen::Index>(dim_));

    std::vector<Eigen::MatrixXcd> eigen_jump_ops;
    eigen_jump_ops.reserve(jump_ops_);
    for (const auto& op : lab_jump_ops_) {
        if (op.matrix.rows() != static_cast<Eigen::Index>(dim_) ||
            op.matrix.cols() != static_cast<Eigen::Index>(dim_)) {
            throw std::invalid_argument("Jump operator dimension mismatch");
        }
        eigen_jump_ops.emplace_back(U_dag_ * op.matrix * U_);
    }

    buckets_.clear();
    buckets_.reserve(dim_ * dim_);

    for (Eigen::Index m = 0; m < static_cast<Eigen::Index>(dim_); ++m) {
        for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(dim_); ++n) {
            const double omega = eps_(m) - eps_(n);
            std::size_t idx = bucket_index_for(omega);
            for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
                const complex value = eigen_jump_ops[alpha](m, n);
                if (std::abs(value) <= opts_.transition_cutoff) {
                    continue;
                }
                if (idx == buckets_.size()) {
                    FrequencyBucket bucket;
                    bucket.omega = omega;
                    bucket.A.assign(jump_ops_,
                                    Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                           static_cast<Eigen::Index>(dim_)));
                    bucket.A_dag.assign(jump_ops_,
                                        Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                               static_cast<Eigen::Index>(dim_)));
                    bucket.A_dag_A.assign(jump_ops_ * jump_ops_,
                                          Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                                 static_cast<Eigen::Index>(dim_)));
                    bucket.G = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                                      static_cast<Eigen::Index>(jump_ops_));
                    bucket.gamma = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                                         static_cast<Eigen::Index>(jump_ops_));
                    bucket.H_ls = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                         static_cast<Eigen::Index>(dim_));
                    buckets_.push_back(std::move(bucket));
                    idx = buckets_.size() - 1;
                }
                buckets_[idx].A[alpha](m, n) = value;
            }
        }
    }

    rebuild_bucket_caches();
    reset(0.0);
}

void TCL2Generator::reset(double t0) {
    current_time_ = t0;
    H_ls_eig_.setZero(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    for (auto& bucket : buckets_) {
        bucket.G.setZero();
        bucket.gamma.setZero();
        bucket.H_ls.setZero();
    }
    rebuild_lamb_shift();
}

void TCL2Generator::advance(double t1) {
    if (t1 < current_time_) {
        throw std::invalid_argument("advance: target time must be non-decreasing");
    }

    const double interval = t1 - current_time_;
    if (std::abs(interval) <= std::numeric_limits<double>::epsilon() *
                                  std::max(1.0, std::abs(current_time_))) {
        current_time_ = t1;
        return;
    }

    for (auto& bucket : buckets_) {
        bucket.H_ls.setZero();
        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            for (std::size_t beta = 0; beta < jump_ops_; ++beta) {
                const complex delta = weighted_integral(bucket.omega, current_time_, t1, alpha, beta);
                bucket.G(alpha, beta) += delta;

                double gamma_val = 2.0 * bucket.G(alpha, beta).real();
                if (std::abs(gamma_val) <= opts_.gamma_cutoff) {
                    gamma_val = 0.0;
                }
                bucket.gamma(alpha, beta) = gamma_val;

                double s = bucket.G(alpha, beta).imag();
                if (std::abs(s) <= opts_.gamma_cutoff) {
                    continue;
                }

                const auto index = pair_index(alpha, beta, jump_ops_);
                bucket.H_ls.noalias() += s * bucket.A_dag_A[index];
            }
        }
    }

    rebuild_lamb_shift();
    current_time_ = t1;
}

void TCL2Generator::apply(const Eigen::MatrixXcd& rho_lab, Eigen::MatrixXcd& drho_lab) const {
    if (rho_lab.rows() != static_cast<Eigen::Index>(dim_) ||
        rho_lab.cols() != static_cast<Eigen::Index>(dim_)) {
        throw std::invalid_argument("Density matrix dimension mismatch");
    }

    Eigen::MatrixXcd rho_eig = U_dag_ * rho_lab * U_;
    Eigen::MatrixXcd drho_eig = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                       static_cast<Eigen::Index>(dim_));
    const complex neg_i(0.0, -1.0);

    for (Eigen::Index m = 0; m < static_cast<Eigen::Index>(dim_); ++m) {
        for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(dim_); ++n) {
            const double delta = eps_(m) - eps_(n);
            drho_eig(m, n) += neg_i * delta * rho_eig(m, n);
        }
    }

    if (H_ls_eig_.rows() == static_cast<Eigen::Index>(dim_) &&
        H_ls_eig_.cols() == static_cast<Eigen::Index>(dim_)) {
        Eigen::MatrixXcd comm = H_ls_eig_ * rho_eig - rho_eig * H_ls_eig_;
        drho_eig += neg_i * comm;
    }

    for (const auto& bucket : buckets_) {
        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            const auto& A_alpha_dag = bucket.A_dag[alpha];
            for (std::size_t beta = 0; beta < jump_ops_; ++beta) {
                double gamma = bucket.gamma(alpha, beta);
                if (std::abs(gamma) <= opts_.gamma_cutoff) {
                    continue;
                }
                const auto& A_beta = bucket.A[beta];
                const auto& A_dag_A = bucket.A_dag_A[pair_index(alpha, beta, jump_ops_)];
                drho_eig += gamma * (A_beta * rho_eig * A_alpha_dag -
                                     0.5 * (A_dag_A * rho_eig + rho_eig * A_dag_A));
            }
        }
    }

    drho_lab = U_ * drho_eig * U_dag_;
}

TCL2Generator::complex TCL2Generator::weighted_integral(double omega,
                                                        double a,
                                                        double b,
                                                        std::size_t alpha,
                                                        std::size_t beta) const {
    if (b <= a) {
        return complex{0.0, 0.0};
    }

    double interval = b - a;
    std::size_t segments = opts_.integration_min_subdivisions;
    if (opts_.integration_step_hint > 0.0) {
        const double suggested = interval / opts_.integration_step_hint;
        segments = std::max<std::size_t>(segments,
                                         static_cast<std::size_t>(std::ceil(suggested)));
    }

    segments = std::min<std::size_t>(segments, opts_.integration_max_subdivisions);
    if (segments < 2) {
        segments = 2;
    }
    if (segments % 2 != 0) {
        ++segments;
    }

    const double h = (b - a) / static_cast<double>(segments);
    auto integrand = [&](double tau) -> complex {
        const complex phase = std::exp(complex{0.0, omega * tau});
        return phase * bath_(tau, alpha, beta);
    };

    complex sum = integrand(a) + integrand(b);
    for (std::size_t i = 1; i < segments; ++i) {
        const double tau = a + h * static_cast<double>(i);
        const complex val = integrand(tau);
        sum += (i % 2 == 0 ? 2.0 : 4.0) * val;
    }

    return sum * (h / 3.0);
}

std::size_t TCL2Generator::bucket_index_for(double omega) const {
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
        const double tol = std::max(opts_.frequency_tolerance,
                                    std::numeric_limits<double>::epsilon());
        if (nearly_equal(buckets_[i].omega, omega, tol)) {
            return i;
        }
    }
    return buckets_.size();
}

void TCL2Generator::rebuild_bucket_caches() {
    const std::size_t pair_count = jump_ops_ * jump_ops_;
    for (auto& bucket : buckets_) {
        if (bucket.A.size() != jump_ops_) {
            bucket.A.assign(jump_ops_,
                            Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                   static_cast<Eigen::Index>(dim_)));
        }
        if (bucket.A_dag.size() != jump_ops_) {
            bucket.A_dag.assign(jump_ops_,
                                Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                       static_cast<Eigen::Index>(dim_)));
        }
        if (bucket.A_dag_A.size() != pair_count) {
            bucket.A_dag_A.assign(pair_count,
                                  Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                         static_cast<Eigen::Index>(dim_)));
        }

        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            bucket.A_dag[alpha] = bucket.A[alpha].adjoint();
        }
        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            for (std::size_t beta = 0; beta < jump_ops_; ++beta) {
                const auto index = pair_index(alpha, beta, jump_ops_);
                bucket.A_dag_A[index] = bucket.A_dag[alpha] * bucket.A[beta];
            }
        }

        bucket.G = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                          static_cast<Eigen::Index>(jump_ops_));
        bucket.gamma = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                             static_cast<Eigen::Index>(jump_ops_));
        bucket.H_ls = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                             static_cast<Eigen::Index>(dim_));
    }
}

void TCL2Generator::rebuild_lamb_shift() {
    H_ls_eig_ = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                       static_cast<Eigen::Index>(dim_));
    for (const auto& bucket : buckets_) {
        if (bucket.H_ls.rows() == static_cast<Eigen::Index>(dim_) &&
            bucket.H_ls.cols() == static_cast<Eigen::Index>(dim_)) {
            H_ls_eig_ += bucket.H_ls;
        }
    }
}

}  // namespace taco::tcl
