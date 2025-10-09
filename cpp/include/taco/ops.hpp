// ops.hpp — Common small operators: Pauli, ladder, basis/projectors, Kronecker helper

#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <cstddef>
#include <stdexcept>
#include <cmath>

namespace taco::ops {

using Matrix = Eigen::MatrixXcd;
using Vector = Eigen::VectorXcd;

// --------------------------- 2×2 Pauli and identity -------------------------

inline Matrix I2() {
    Matrix M(2,2);
    M.setZero();
    M(0,0) = 1.0; M(1,1) = 1.0;
    return M;
}

inline Matrix sigma_x() {
    Matrix M(2,2);
    M << 0.0, 1.0,
         1.0, 0.0;
    return M;
}

inline Matrix sigma_y() {
    Matrix M(2,2);
    M << 0.0, std::complex<double>(0.0, -1.0),
         std::complex<double>(0.0, +1.0), 0.0;
    return M;
}

inline Matrix sigma_z() {
    Matrix M(2,2);
    M << 1.0,  0.0,
         0.0, -1.0;
    return M;
}

inline Matrix sigma_plus() { // |1><0|
    Matrix M(2,2);
    M.setZero();
    M(1,0) = 1.0;
    return M;
}

inline Matrix sigma_minus() { // |0><1|
    Matrix M(2,2);
    M.setZero();
    M(0,1) = 1.0;
    return M;
}

// --------------------------- Basis/projector helpers ------------------------

// |i><j| in N-dimensional Hilbert space
inline Matrix basis_op(std::size_t N, std::size_t i, std::size_t j) {
    if (N == 0) throw std::invalid_argument("basis_op: N must be > 0");
    if (i >= N || j >= N) throw std::out_of_range("basis_op: index out of range");
    Matrix M(N, N); M.setZero();
    M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = 1.0;
    return M;
}

inline Matrix projector(std::size_t N, std::size_t i) { // |i><i|
    return basis_op(N, i, i);
}

// --------------------------- Harmonic ladder operators ----------------------

// a |n> = sqrt(n) |n-1>  →  a_{n-1, n} = sqrt(n)
inline Matrix a(std::size_t N) {
    if (N == 0) throw std::invalid_argument("a: N must be > 0");
    Matrix M(N, N); M.setZero();
    for (std::size_t n = 1; n < N; ++n) {
        M(static_cast<Eigen::Index>(n-1), static_cast<Eigen::Index>(n)) = std::sqrt(static_cast<double>(n));
    }
    return M;
}

// a† |n> = sqrt(n+1) |n+1>  →  adag_{n+1, n} = sqrt(n+1)
inline Matrix adag(std::size_t N) {
    if (N == 0) throw std::invalid_argument("adag: N must be > 0");
    Matrix M(N, N); M.setZero();
    for (std::size_t n = 0; n + 1 < N; ++n) {
        M(static_cast<Eigen::Index>(n+1), static_cast<Eigen::Index>(n)) = std::sqrt(static_cast<double>(n+1));
    }
    return M;
}

// Number operator: n = a† a (diagonal with entries 0,1,2,...)
inline Matrix number(std::size_t N) {
    if (N == 0) throw std::invalid_argument("number: N must be > 0");
    Matrix M(N, N); M.setZero();
    for (std::size_t n = 0; n < N; ++n) {
        M(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)) = static_cast<double>(n);
    }
    return M;
}

// --------------------------- Kronecker helper -------------------------------

inline Matrix kron(const Matrix& A, const Matrix& B) {
    const auto expr = Eigen::kroneckerProduct(A, B);
    Matrix K(expr.rows(), expr.cols());
    K = expr;
    return K;
}

// --------------------------- Trace utilities --------------------------------

inline std::complex<double> tr(const Matrix& A) {
    return A.trace();
}

// For Hermitian matrices, the trace should be real (return real part safely)
inline double tr_real_hermitian(const Matrix& A) {
    return A.trace().real();
}

// Hilbert–Schmidt inner product: ⟨A,B⟩ = Tr(A† B)
inline std::complex<double> hs_inner(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("hs_inner: dimension mismatch");
    }
    return (A.adjoint() * B).trace();
}

// Purity: Tr(rho^2)
inline double purity(const Matrix& rho) {
    return (rho * rho).trace().real();
}

// --------------------------- Partial trace (bipartite) ----------------------

// rho is (dA*dB) x (dA*dB). Partial trace over B → rho_A of size dA x dA
inline Matrix ptrace_B(const Matrix& rho, std::size_t dA, std::size_t dB) {
    if (dA == 0 || dB == 0) throw std::invalid_argument("ptrace_B: dims must be > 0");
    const std::size_t N = dA * dB;
    if (rho.rows() != static_cast<Eigen::Index>(N) || rho.cols() != static_cast<Eigen::Index>(N)) {
        throw std::invalid_argument("ptrace_B: rho dimension mismatch");
    }
    Matrix out(dA, dA); out.setZero();
    for (std::size_t iA = 0; iA < dA; ++iA) {
        for (std::size_t kA = 0; kA < dA; ++kA) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t b = 0; b < dB; ++b) {
                const Eigen::Index i = static_cast<Eigen::Index>(iA * dB + b);
                const Eigen::Index k = static_cast<Eigen::Index>(kA * dB + b);
                sum += rho(i, k);
            }
            out(static_cast<Eigen::Index>(iA), static_cast<Eigen::Index>(kA)) = sum;
        }
    }
    return out;
}

// Partial trace over A → rho_B of size dB x dB
inline Matrix ptrace_A(const Matrix& rho, std::size_t dA, std::size_t dB) {
    if (dA == 0 || dB == 0) throw std::invalid_argument("ptrace_A: dims must be > 0");
    const std::size_t N = dA * dB;
    if (rho.rows() != static_cast<Eigen::Index>(N) || rho.cols() != static_cast<Eigen::Index>(N)) {
        throw std::invalid_argument("ptrace_A: rho dimension mismatch");
    }
    Matrix out(dB, dB); out.setZero();
    for (std::size_t iB = 0; iB < dB; ++iB) {
        for (std::size_t kB = 0; kB < dB; ++kB) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t a = 0; a < dA; ++a) {
                const Eigen::Index i = static_cast<Eigen::Index>(a * dB + iB);
                const Eigen::Index k = static_cast<Eigen::Index>(a * dB + kB);
                sum += rho(i, k);
            }
            out(static_cast<Eigen::Index>(iB), static_cast<Eigen::Index>(kB)) = sum;
        }
    }
    return out;
}

// --------------------------- State constructors -----------------------------

inline Vector ket(std::size_t N, std::size_t i) {
    if (N == 0) throw std::invalid_argument("ket: N must be > 0");
    if (i >= N) throw std::out_of_range("ket: index out of range");
    Vector v(static_cast<Eigen::Index>(N)); v.setZero();
    v(static_cast<Eigen::Index>(i)) = 1.0;
    return v;
}

inline Vector bra(std::size_t N, std::size_t i) { return ket(N, i).conjugate(); }

inline Matrix outer(const Vector& psi, const Vector& phi_dag) {
    if (psi.size() != phi_dag.size()) throw std::invalid_argument("outer: dimension mismatch");
    return psi * phi_dag.adjoint();
}

inline Matrix rho_pure(const Vector& psi_raw) {
    Vector psi = psi_raw;
    const double nrm = psi.norm();
    if (nrm > 0.0) psi /= nrm;
    return psi * psi.adjoint();
}

// Common qubit states
inline Vector ket0() { Vector v(2); v << 1.0, 0.0; return v; }
inline Vector ket1() { Vector v(2); v << 0.0, 1.0; return v; }
inline Matrix rho_qubit_0() { return rho_pure(ket0()); }
inline Matrix rho_qubit_1() { return rho_pure(ket1()); }

inline Vector ket_plus_x() { return (ket0() + ket1()) / std::sqrt(2.0); }
inline Vector ket_minus_x() { return (ket0() - ket1()) / std::sqrt(2.0); }
inline Vector ket_plus_y() { return (ket0() + std::complex<double>(0.0,1.0)*ket1()) / std::sqrt(2.0); }
inline Vector ket_minus_y() { return (ket0() - std::complex<double>(0.0,1.0)*ket1()) / std::sqrt(2.0); }

inline Matrix rho_plus_x() { return rho_pure(ket_plus_x()); }
inline Matrix rho_minus_x() { return rho_pure(ket_minus_x()); }
inline Matrix rho_plus_y() { return rho_pure(ket_plus_y()); }
inline Matrix rho_minus_y() { return rho_pure(ket_minus_y()); }

// --------------------------- Algebraic helpers ------------------------------

inline Matrix comm(const Matrix& A, const Matrix& B) { return A*B - B*A; }
inline Matrix anti(const Matrix& A, const Matrix& B) { return A*B + B*A; }

inline Matrix hermitize(const Matrix& A) { return 0.5 * (A + A.adjoint()); }

inline Matrix renormalize_trace(const Matrix& A) {
    const auto trA = A.trace();
    Matrix B = A;
    if (std::abs(trA) > 0.0) B /= trA;
    return B;
}

inline Matrix hermitize_and_normalize(const Matrix& rho) {
    return renormalize_trace(hermitize(rho));
}

// --------------------------- Validity checks --------------------------------

inline bool is_hermitian(const Matrix& A, double tol = 1e-12) {
    return (A - A.adjoint()).cwiseAbs().maxCoeff() <= tol;
}

inline bool is_psd(const Matrix& H, double tol = 1e-12) {
    // Check Hermitian part eigenvalues ≥ -tol
    Matrix S = hermitize(H);
    Eigen::SelfAdjointEigenSolver<Matrix> es(S);
    if (es.info() != Eigen::Success) return false;
    return es.eigenvalues().minCoeff() >= -tol;
}

inline bool is_density(const Matrix& rho, double tol = 1e-10) {
    if (rho.rows() != rho.cols()) return false;
    if (!is_hermitian(rho, tol*10)) return false;
    if (!is_psd(rho, tol*10)) return false;
    const double tr = rho.trace().real();
    return std::abs(tr - 1.0) <= tol;
}

// --------------------------- Bloch conversions (qubit) ----------------------

inline Eigen::Vector3d bloch_from_rho(const Matrix& rho) {
    if (rho.rows() != 2 || rho.cols() != 2) throw std::invalid_argument("bloch_from_rho: requires 2x2 rho");
    Eigen::Vector3d r;
    r(0) = (rho * sigma_x()).trace().real();
    r(1) = (rho * sigma_y()).trace().real();
    r(2) = (rho * sigma_z()).trace().real();
    return r;
}

inline Matrix rho_from_bloch(const Eigen::Vector3d& r, double tol = 1e-12) {
    if (r.norm() > 1.0 + tol) throw std::invalid_argument("rho_from_bloch: |r| must be ≤ 1");
    Matrix M = 0.5*(I2() + r(0)*sigma_x() + r(1)*sigma_y() + r(2)*sigma_z());
    return hermitize_and_normalize(M);
}

// --------------------------- Vec/unvec and superops -------------------------

inline Vector vec(const Matrix& A) {
    Vector v(static_cast<Eigen::Index>(A.size()));
    // Column-major flattening (Eigen default)
    Eigen::Map<const Vector> m(A.data(), v.size());
    v = m;
    return v;
}

inline Matrix unvec(const Vector& v, std::size_t N) {
    if (v.size() != static_cast<Eigen::Index>(N*N)) throw std::invalid_argument("unvec: size mismatch");
    Matrix A(N, N);
    Eigen::Map<const Matrix> m(v.data(), static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(N));
    A = m;
    return A;
}

// Left and right multiplication superoperators in vec form
inline Matrix super_left(const Matrix& A) {
    const Eigen::Index N = A.rows();
    return kron(Matrix::Identity(N, N), A);
}

inline Matrix super_right(const Matrix& B) {
    const Eigen::Index N = B.rows();
    return kron(B.transpose(), Matrix::Identity(N, N));
}

// --------------------------- Fidelity / distances ---------------------------

inline Matrix sqrt_psd(const Matrix& H, double tol = 1e-12) {
    Matrix S = hermitize(H);
    Eigen::SelfAdjointEigenSolver<Matrix> es(S);
    if (es.info() != Eigen::Success) throw std::runtime_error("sqrt_psd: eigensolve failed");
    Eigen::VectorXd d = es.eigenvalues();
    for (Eigen::Index i = 0; i < d.size(); ++i) d(i) = (d(i) > tol) ? std::sqrt(d(i)) : 0.0;
    return es.eigenvectors() * d.asDiagonal() * es.eigenvectors().adjoint();
}

// Uhlmann fidelity F(ρ,σ) = (Tr sqrt( sqrtρ σ sqrtρ ))^2
inline double fidelity(const Matrix& rho, const Matrix& sigma) {
    Matrix sr = sqrt_psd(rho);
    Matrix M  = sr * sigma * sr;
    Matrix sM = sqrt_psd(M);
    const double t = sM.trace().real();
    return t * t;
}

// Trace distance D(ρ,σ) = 1/2 ||ρ-σ||_1 (for Hermitian difference)
inline double trace_distance(const Matrix& rho, const Matrix& sigma) {
    Matrix H = hermitize(rho - sigma);
    Eigen::SelfAdjointEigenSolver<Matrix> es(H);
    if (es.info() != Eigen::Success) throw std::runtime_error("trace_distance: eigensolve failed");
    const auto& d = es.eigenvalues();
    double s = 0.0;
    for (Eigen::Index i = 0; i < d.size(); ++i) s += std::abs(d(i));
    return 0.5 * s;
}

// --------------------------- Matrix/vector norms ----------------------------

inline double fro_norm(const Matrix& A) { return A.norm(); }

inline Eigen::VectorXd singular_values(const Matrix& A) {
    // Use SVD for general complex matrices
    Eigen::JacobiSVD<Matrix> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.singularValues();
}

inline double op_norm2(const Matrix& A) { // spectral/operator 2-norm = max singular value
    if (A.rows() == 0 || A.cols() == 0) return 0.0;
    // Fast path for Hermitian: max |eigenvalue|
    if (is_hermitian(A)) {
        Eigen::SelfAdjointEigenSolver<Matrix> es(A);
        if (es.info() != Eigen::Success) throw std::runtime_error("op_norm2: eigensolve failed");
        return std::max(std::abs(es.eigenvalues().minCoeff()), std::abs(es.eigenvalues().maxCoeff()));
    }
    return singular_values(A).maxCoeff();
}

inline double trace_norm(const Matrix& A) { // nuclear norm = sum singular values
    if (A.rows() == 0 || A.cols() == 0) return 0.0;
    return singular_values(A).sum();
}

inline double schatten_p_norm(const Matrix& A, double p) {
    if (!(p >= 1.0)) throw std::invalid_argument("schatten_p_norm: p must be >= 1");
    if (!std::isfinite(p)) return op_norm2(A);
    Eigen::VectorXd s = singular_values(A);
    if (s.size() == 0) return 0.0;
    double acc = 0.0;
    for (Eigen::Index i = 0; i < s.size(); ++i) acc += std::pow(s(i), p);
    return std::pow(acc, 1.0 / p);
}

inline double op_norm1(const Matrix& A) { // induced 1-norm = max column sum
    if (A.rows() == 0 || A.cols() == 0) return 0.0;
    double mx = 0.0;
    for (Eigen::Index j = 0; j < A.cols(); ++j) {
        double s = 0.0;
        for (Eigen::Index i = 0; i < A.rows(); ++i) s += std::abs(A(i,j));
        if (s > mx) mx = s;
    }
    return mx;
}

inline double op_norm_inf(const Matrix& A) { // induced ∞-norm = max row sum
    if (A.rows() == 0 || A.cols() == 0) return 0.0;
    double mx = 0.0;
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        double s = 0.0;
        for (Eigen::Index j = 0; j < A.cols(); ++j) s += std::abs(A(i,j));
        if (s > mx) mx = s;
    }
    return mx;
}

inline double max_abs_entry(const Matrix& A) {
    if (A.size() == 0) return 0.0;
    return A.cwiseAbs().maxCoeff();
}

inline double comm_norm_fro(const Matrix& A, const Matrix& B) { return fro_norm(comm(A,B)); }

inline double v_norm2(const Vector& v) { return v.norm(); }
inline double v_norm1(const Vector& v) { return v.lpNorm<1>(); }
inline double v_norm_inf(const Vector& v) { return v.lpNorm<Eigen::Infinity>(); }

} // namespace taco::ops
