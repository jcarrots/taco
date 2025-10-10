// Core TCL2 implementation: builds frequency buckets, integrates bath correlations,
// and applies Hamiltonian + dissipative dynamics in the eigen basis.
#include "taco/tcl2.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// Eigen decomposition (self-adjoint) and dense linear algebra
#include <Eigen/Eigenvalues>

namespace taco::tcl {

namespace {
// Helper utilities local to this translation unit

// Flatten a 2D pair (alpha,beta) into a single index for compact vectors
inline std::size_t pair_index(std::size_t alpha, std::size_t beta, std::size_t width) noexcept {
    return alpha * width + beta;
}

// Compare two floating point values with a symmetric absolute tolerance
inline bool nearly_equal(double a, double b, double tol) noexcept {
    return std::abs(a - b) <= tol;
}

}  // namespace

TCL2Generator::TCL2Generator(const Eigen::MatrixXcd& hamiltonian,          // system Hamiltonian (Hermitian)
                             std::vector<JumpOperator> jump_ops,            // lab-basis jump operators
                             const bath::CorrelationFunction& correlation,  // bath correlation interface
                             const GeneratorOptions& options)               // numerical thresholds / quadrature
    : bath_(correlation), opts_(options) {
    // Validate Hamiltonian shape
    if (hamiltonian.rows() != hamiltonian.cols()) {
        throw std::invalid_argument("Hamiltonian must be square");
    }

    // Cache dimension (Hilbert space size)
    dim_ = static_cast<std::size_t>(hamiltonian.rows());
    if (dim_ == 0) {
        throw std::invalid_argument("Hamiltonian dimension must be positive");
    }

    // Take ownership of jump operators and validate count
    lab_jump_ops_ = std::move(jump_ops);
    jump_ops_ = lab_jump_ops_.size();
    if (jump_ops_ == 0) {
        throw std::invalid_argument("At least one jump operator is required");
    }

    // Bath rank (number of channels) must match number of jump operators
    if (bath_.rank() != jump_ops_) {
        throw std::invalid_argument("Bath correlation rank must match number of jump operators");
    }

    // Diagonalize Hamiltonian (self-adjoint decomposition H = U diag(eps) U†)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hamiltonian);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to diagonalize Hamiltonian");
    }

    // Cache eigenvalues and eigenvectors; precompute U†
    eps_ = solver.eigenvalues();
    U_ = solver.eigenvectors();
    U_dag_ = U_.adjoint();

    // Initialize total Lamb-shift accumulator in eigen basis
    H_ls_eig_ = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                       static_cast<Eigen::Index>(dim_));

    // Transform jump operators to the instantaneous eigen basis
    std::vector<Eigen::MatrixXcd> eigen_jump_ops;
    eigen_jump_ops.reserve(jump_ops_);
    for (const auto& op : lab_jump_ops_) {
        // Dimension check for each jump operator
        if (op.matrix.rows() != static_cast<Eigen::Index>(dim_) ||
            op.matrix.cols() != static_cast<Eigen::Index>(dim_)) {
            throw std::invalid_argument("Jump operator dimension mismatch");
        }
        // A_eig = U† A_lab U
        eigen_jump_ops.emplace_back(U_dag_ * op.matrix * U_);
    }

    // Build frequency buckets keyed by Bohr frequencies ω = ε_m - ε_n
    buckets_.clear();
    buckets_.reserve(dim_ * dim_);

    for (Eigen::Index m = 0; m < static_cast<Eigen::Index>(dim_); ++m) {
        for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(dim_); ++n) {
            const double omega = eps_(m) - eps_(n);             // transition frequency
            std::size_t idx = bucket_index_for(omega);          // existing bucket index or sentinel
            for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
                const complex value = eigen_jump_ops[alpha](m, n);  // matrix element for channel alpha
                if (std::abs(value) <= opts_.transition_cutoff) {   // skip negligible transitions
                    continue;
                }
                // Lazily create bucket on first significant transition at this ω
                if (idx == buckets_.size()) {
                    FrequencyBucket bucket;
                    bucket.omega = omega;  // key frequency
                    // Allocate channel matrices for A, A† and pairwise A†A
                    bucket.A.assign(jump_ops_,
                                    Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                           static_cast<Eigen::Index>(dim_)));
                    bucket.A_dag.assign(jump_ops_,
                                        Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                               static_cast<Eigen::Index>(dim_)));
                    bucket.A_dag_A.assign(jump_ops_ * jump_ops_,
                                          Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                                 static_cast<Eigen::Index>(dim_)));
                    // Initialize kernel accumulator G(α,β) and decay rates γ(α,β)
                    bucket.G = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                                      static_cast<Eigen::Index>(jump_ops_));
                    bucket.gamma = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                                         static_cast<Eigen::Index>(jump_ops_));
                    // Lamb shift contribution matrix at this ω
                    bucket.H_ls = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                         static_cast<Eigen::Index>(dim_));
                    buckets_.push_back(std::move(bucket));
                    idx = buckets_.size() - 1;  // update index to new bucket
                }
                // Store non-negligible transition amplitude for channel alpha at (m,n)
                buckets_[idx].A[alpha](m, n) = value;
            }
        }
    }

    // Compute A†, A†A and reset kernel accumulators
    rebuild_bucket_caches();
    // Reset time and clear accumulated G/γ/Hls
    reset(0.0);
}

// Reset generator to time t0 and clear accumulated kernels and Lamb shift
void TCL2Generator::reset(double t0) {
    current_time_ = t0;  // set current time
    H_ls_eig_.setZero(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));  // clear total H_ls
    for (auto& bucket : buckets_) {
        bucket.G.setZero();      // clear kernel accumulators
        bucket.gamma.setZero();  // clear decay rates
        bucket.H_ls.setZero();   // clear per-bucket Lamb shift
    }
    rebuild_lamb_shift();  // recompute total H_ls (becomes zero)
}

// Advance kernel accumulators from current_time_ to t1
void TCL2Generator::advance(double t1) {
    if (t1 < current_time_) {
        throw std::invalid_argument("advance: target time must be non-decreasing");
    }

    const double interval = t1 - current_time_;
    // If interval is numerically tiny, just update time and exit
    if (std::abs(interval) <= std::numeric_limits<double>::epsilon() *
                                  std::max(1.0, std::abs(current_time_))) {
        current_time_ = t1;
        return;
    }

    // Update each bucket's kernel integrals and Lamb shift contribution
    for (auto& bucket : buckets_) {
        bucket.H_ls.setZero();  // rebuild H_ls from fresh G
        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            for (std::size_t beta = 0; beta < jump_ops_; ++beta) {
                // Incremental integral ΔG over (current_time_, t1]
                const complex delta = weighted_integral(bucket.omega, current_time_, t1, alpha, beta);
                bucket.G(alpha, beta) += delta;  // accumulate kernel

                // Decay rate: γ = 2 Re G (apply cutoff to drop tiny noise)
                double gamma_val = 2.0 * bucket.G(alpha, beta).real();
                if (std::abs(gamma_val) <= opts_.gamma_cutoff) {
                    gamma_val = 0.0;
                }
                bucket.gamma(alpha, beta) = gamma_val;

                // Lamb shift coefficient: Im G; skip if negligible
                double s = bucket.G(alpha, beta).imag();
                if (std::abs(s) <= opts_.gamma_cutoff) {
                    continue;
                }

                // Accumulate s(α,β) * A_α† A_β into the per-bucket H_ls
                const auto index = pair_index(alpha, beta, jump_ops_);
                bucket.H_ls.noalias() += s * bucket.A_dag_A[index];
            }
        }
    }

    rebuild_lamb_shift();  // H_ls_eig_ = sum_ω H_ls(ω)
    current_time_ = t1;    // commit new time
}

// Apply TCL2 generator to compute dρ/dt at the current time
void TCL2Generator::apply(const Eigen::MatrixXcd& rho_lab, Eigen::MatrixXcd& drho_lab) const {
    // Check shape of input density matrix
    if (rho_lab.rows() != static_cast<Eigen::Index>(dim_) ||
        rho_lab.cols() != static_cast<Eigen::Index>(dim_)) {
        throw std::invalid_argument("Density matrix dimension mismatch");
    }

    // Transform to eigen basis for efficient application of generators
    Eigen::MatrixXcd rho_eig = U_dag_ * rho_lab * U_;
    Eigen::MatrixXcd drho_eig = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                                       static_cast<Eigen::Index>(dim_));
    const complex neg_i(0.0, -1.0);

    // Coherent evolution: -i[H_diag, ρ]
    for (Eigen::Index m = 0; m < static_cast<Eigen::Index>(dim_); ++m) {
        for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(dim_); ++n) {
            const double delta = eps_(m) - eps_(n);      // energy difference
            drho_eig(m, n) += neg_i * delta * rho_eig(m, n);
        }
    }

    // Lamb shift: -i[H_ls, ρ]
    if (H_ls_eig_.rows() == static_cast<Eigen::Index>(dim_) &&
        H_ls_eig_.cols() == static_cast<Eigen::Index>(dim_)) {
        Eigen::MatrixXcd comm = H_ls_eig_ * rho_eig - rho_eig * H_ls_eig_;
        drho_eig += neg_i * comm;
    }

    // Dissipator: sum_ω,α,β γ_{αβ}(ω) [ A_β ρ A_α† - 1/2{A_α†A_β, ρ} ]
    for (const auto& bucket : buckets_) {
        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            const auto& A_alpha_dag = bucket.A_dag[alpha];
            for (std::size_t beta = 0; beta < jump_ops_; ++beta) {
                double gamma = bucket.gamma(alpha, beta);
                if (std::abs(gamma) <= opts_.gamma_cutoff) {
                    continue;  // skip negligible rates
                }
                const auto& A_beta = bucket.A[beta];
                const auto& A_dag_A = bucket.A_dag_A[pair_index(alpha, beta, jump_ops_)];
                drho_eig += gamma * (A_beta * rho_eig * A_alpha_dag -
                                     0.5 * (A_dag_A * rho_eig + rho_eig * A_dag_A));
            }
        }
    }

    // Transform derivative back to lab basis
    drho_lab = U_ * drho_eig * U_dag_;
}

// Numerically integrate ∫_a^b e^{iωτ} C_{αβ}(τ) dτ using composite Simpson's rule
complex TCL2Generator::weighted_integral(double omega,
                                                        double a,
                                                        double b,
                                                        std::size_t alpha,
                                                        std::size_t beta) const {
    if (b <= a) {
        return complex{0.0, 0.0};  // empty interval
    }

    // Choose segment count based on step hint and bounds
    double interval = b - a;
    std::size_t segments = opts_.integration_min_subdivisions;
    if (opts_.integration_step_hint > 0.0) {
        const double suggested = interval / opts_.integration_step_hint;
        segments = std::max<std::size_t>(segments,
                                         static_cast<std::size_t>(std::ceil(suggested)));
    }

    // Clamp and make even (Simpson requires even number of subintervals)
    segments = std::min<std::size_t>(segments, opts_.integration_max_subdivisions);
    if (segments < 2) {
        segments = 2;
    }
    if (segments % 2 != 0) {
        ++segments;
    }

    const double h = (b - a) / static_cast<double>(segments);  // subinterval width
    auto integrand = [&](double tau) -> complex {
        const complex phase = std::exp(complex{0.0, omega * tau});
        return phase * bath_(tau, alpha, beta);  // oscillatory weight times correlation
    };

    // Composite Simpson's rule: h/3 [f(a)+f(b) + 4 Σ f(a+(2k-1)h) + 2 Σ f(a+2kh)]
    complex sum = integrand(a) + integrand(b);
    for (std::size_t i = 1; i < segments; ++i) {
        const double tau = a + h * static_cast<double>(i);
        const complex val = integrand(tau);
        sum += (i % 2 == 0 ? 2.0 : 4.0) * val;
    }

    return sum * (h / 3.0);
}

// Find an existing bucket with matching ω within tolerance, else return sentinel = buckets_.size()
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

// Recompute A† and A†A caches for each bucket and reset G/γ/H_ls
void TCL2Generator::rebuild_bucket_caches() {
    const std::size_t pair_count = jump_ops_ * jump_ops_;
    for (auto& bucket : buckets_) {
        // Ensure container sizes are correct
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

        // A† per channel
        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            bucket.A_dag[alpha] = bucket.A[alpha].adjoint();
        }
        // Pairwise products A†_α A_β
        for (std::size_t alpha = 0; alpha < jump_ops_; ++alpha) {
            for (std::size_t beta = 0; beta < jump_ops_; ++beta) {
                const auto index = pair_index(alpha, beta, jump_ops_);
                bucket.A_dag_A[index] = bucket.A_dag[alpha] * bucket.A[beta];
            }
        }

        // Reset kernel accumulators and Lamb shift contribution
        bucket.G = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                          static_cast<Eigen::Index>(jump_ops_));
        bucket.gamma = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(jump_ops_),
                                             static_cast<Eigen::Index>(jump_ops_));
        bucket.H_ls = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim_),
                                             static_cast<Eigen::Index>(dim_));
    }
}

// Sum per-bucket Lamb-shift contributions into H_ls_eig_
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
