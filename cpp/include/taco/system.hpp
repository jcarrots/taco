// system.hpp — Eigensystem, basis transforms, Bohr frequencies, and spectral buckets
// Header-only utilities to build a reusable "system" description for TCL generators.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace taco::sys {

// Numeric equality with absolute tolerance
inline bool nearly_equal(double a, double b, double tol) noexcept {
    return std::abs(a - b) <= tol;
}

// --------------------------- Eigensystem + transforms ------------------------

struct Eigensystem {
    std::size_t dim{0};
    Eigen::VectorXd eps;      // eigenvalues (ascending)
    Eigen::MatrixXcd U;       // eigenvectors (columns)
    Eigen::MatrixXcd U_dag;   // U†

    Eigensystem() = default;

    explicit Eigensystem(const Eigen::MatrixXcd& H) {
        build(H);
    }

    void build(const Eigen::MatrixXcd& H) {
        if (H.rows() != H.cols()) {
            throw std::invalid_argument("Eigensystem: Hamiltonian must be square");
        }
        if (H.rows() == 0) {
            throw std::invalid_argument("Eigensystem: Hamiltonian dimension must be > 0");
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigensystem: eigen decomposition failed");
        }
        dim  = static_cast<std::size_t>(H.rows());
        eps  = solver.eigenvalues();
        U    = solver.eigenvectors();
        U_dag= U.adjoint();
    }

    // Transform operator between lab and eigen basis
    Eigen::MatrixXcd to_eigen(const Eigen::MatrixXcd& A_lab) const {
        if (A_lab.rows() != static_cast<Eigen::Index>(dim) ||
            A_lab.cols() != static_cast<Eigen::Index>(dim)) {
            throw std::invalid_argument("to_eigen: dimension mismatch");
        }
        return U_dag * A_lab * U;
    }

    Eigen::MatrixXcd to_lab(const Eigen::MatrixXcd& A_eig) const {
        if (A_eig.rows() != static_cast<Eigen::Index>(dim) ||
            A_eig.cols() != static_cast<Eigen::Index>(dim)) {
            throw std::invalid_argument("to_lab: dimension mismatch");
        }
        return U * A_eig * U_dag;
    }

    // Transform density matrix between bases (same as operator transforms)
    Eigen::MatrixXcd rho_to_eigen(const Eigen::MatrixXcd& rho_lab) const {
        return to_eigen(rho_lab);
    }
    Eigen::MatrixXcd rho_to_lab(const Eigen::MatrixXcd& rho_eig) const {
        return to_lab(rho_eig);
    }
};

// ------------------------------ Bohr frequencies ----------------------------

// Precompute all Bohr frequencies ω_{mn} = ε_m - ε_n
struct BohrFrequencies {
    std::size_t dim{0};
    Eigen::VectorXd eps;      // copy of eigenvalues
    Eigen::MatrixXd omega;    // dim x dim

    BohrFrequencies() = default;
    explicit BohrFrequencies(const Eigen::VectorXd& eigenvals) { build(eigenvals); }

    void build(const Eigen::VectorXd& eigenvals) {
        eps = eigenvals;
        dim = static_cast<std::size_t>(eps.size());
        omega.resize(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
        for (Eigen::Index m = 0; m < static_cast<Eigen::Index>(dim); ++m) {
            for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(dim); ++n) {
                omega(m, n) = eps(m) - eps(n);
            }
        }
    }
};

// --------------------------- Frequency buckets (ω) --------------------------

struct FrequencyBucket {
    double omega{0.0};                         // representative frequency
    std::vector<std::pair<int,int>> pairs;     // list of (m,n) transitions with ε_m - ε_n ≈ omega
};

struct FrequencyIndex {
    double tol{1e-9};
    std::vector<FrequencyBucket> buckets;

    // Return existing bucket index or buckets.size() if not found
    std::size_t find_bucket(double w) const {
        for (std::size_t i = 0; i < buckets.size(); ++i) {
            if (nearly_equal(buckets[i].omega, w, tol)) return i;
        }
        return buckets.size();
    }

    // Lookup table: for each (m,n) store bucket index or -1 if none
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> bucket_map(const BohrFrequencies& bf) const {
        const auto D = static_cast<Eigen::Index>(bf.dim);
        Eigen::MatrixXi map(D, D);
        for (Eigen::Index m = 0; m < D; ++m) {
            for (Eigen::Index n = 0; n < D; ++n) {
                const double w = bf.omega(m, n);
                const std::size_t idx = find_bucket(w);
                map(m, n) = (idx < buckets.size()) ? static_cast<int>(idx) : -1;
            }
        }
        return map;
    }
};

// Build frequency buckets by scanning all (m,n) and grouping by |Δε - ω| ≤ tol
inline FrequencyIndex build_frequency_buckets(const BohrFrequencies& bf, double tol = 1e-9) {
    FrequencyIndex FI; FI.tol = std::max(tol, std::numeric_limits<double>::epsilon());
    const auto D = static_cast<Eigen::Index>(bf.dim);
    for (Eigen::Index m = 0; m < D; ++m) {
        for (Eigen::Index n = 0; n < D; ++n) {
            const double w = bf.omega(m, n);
            std::size_t idx = FI.find_bucket(w);
            if (idx == FI.buckets.size()) {
                FrequencyBucket B;
                B.omega = w;
                FI.buckets.push_back(std::move(B));
                idx = FI.buckets.size() - 1;
            }
            FI.buckets[idx].pairs.emplace_back(static_cast<int>(m), static_cast<int>(n));
        }
    }
    // Optional: sort buckets by omega for consistency
    std::sort(FI.buckets.begin(), FI.buckets.end(), [](const FrequencyBucket& a, const FrequencyBucket& b){ return a.omega < b.omega; });
    return FI;
}

// ------------------------- Spectral decomposition A(ω) ----------------------

// Decompose A (in eigen basis) into frequency-resolved components A(ω):
//   [A(ω)]_{mn} = A_{mn} if ε_m - ε_n ≈ ω, else 0
inline std::vector<Eigen::MatrixXcd>
decompose_operator_by_frequency(const Eigen::MatrixXcd& A_eig,
                                const BohrFrequencies& bf,
                                const FrequencyIndex& FI)
{
    if (A_eig.rows() != static_cast<Eigen::Index>(bf.dim) ||
        A_eig.cols() != static_cast<Eigen::Index>(bf.dim)) {
        throw std::invalid_argument("decompose_operator_by_frequency: dimension mismatch");
    }
    const auto D = static_cast<Eigen::Index>(bf.dim);
    std::vector<Eigen::MatrixXcd> parts(FI.buckets.size(), Eigen::MatrixXcd::Zero(D, D));
    for (std::size_t b = 0; b < FI.buckets.size(); ++b) {
        for (const auto& mn : FI.buckets[b].pairs) {
            const int m = mn.first;
            const int n = mn.second;
            parts[b](m, n) = A_eig(m, n);
        }
    }
    return parts;
}

// Bulk decompose multiple channels {A_α}
inline std::vector<std::vector<Eigen::MatrixXcd>>
decompose_operators_by_frequency(const std::vector<Eigen::MatrixXcd>& A_eigs,
                                 const BohrFrequencies& bf,
                                 const FrequencyIndex& FI)
{
    std::vector<std::vector<Eigen::MatrixXcd>> out;
    out.reserve(A_eigs.size());
    for (const auto& A : A_eigs) {
        out.emplace_back(decompose_operator_by_frequency(A, bf, FI));
    }
    return out;
}

// ------------------------------ System builder ------------------------------

// Bundle Hamiltonian eigensystem + frequency buckets and spectral parts
struct System {
    Eigensystem eig;
    BohrFrequencies bf;
    FrequencyIndex fidx;

    // For each channel α: A_lab[α] (input), A_eig[α] (derived), and A_eig_parts[α][b]
    std::vector<Eigen::MatrixXcd> A_lab;
    std::vector<Eigen::MatrixXcd> A_eig;
    std::vector<std::vector<Eigen::MatrixXcd>> A_eig_parts;

    // Build from H and lab-basis operators; frequency tolerance controls bucket grouping.
    void build(const Eigen::MatrixXcd& H,
               const std::vector<Eigen::MatrixXcd>& jump_ops_lab,
               double freq_tol = 1e-9)
    {
        eig = Eigensystem(H);
        bf  = BohrFrequencies(eig.eps);
        fidx= build_frequency_buckets(bf, freq_tol);

        A_lab = jump_ops_lab;
        A_eig.clear();
        A_eig.reserve(A_lab.size());
        for (const auto& A : A_lab) A_eig.push_back(eig.to_eigen(A));
        A_eig_parts = decompose_operators_by_frequency(A_eig, bf, fidx);
    }
};

} // namespace taco::sys

