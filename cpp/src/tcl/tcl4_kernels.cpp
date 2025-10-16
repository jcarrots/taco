#include "taco/tcl4_kernels.hpp"

#include <stdexcept>
#include <vector>
#include <complex>

#include "taco/correlation_fft.hpp"

namespace taco::tcl4 {
namespace {

using Matrix = Eigen::MatrixXcd;
using cd     = std::complex<double>;

// -------------------- Scalar helpers --------------------

Eigen::VectorXcd prefix_int_left_vec(const Eigen::Ref<const Eigen::VectorXcd>& y,
                                     double dt)
{
    Eigen::VectorXcd out(y.size());
    cd accum{0.0, 0.0};
    for (Eigen::Index k = 0; k < y.size(); ++k) {
        accum += y(k);
        out(k) = accum * dt;
    }
    return out;
}

Eigen::VectorXcd causal_conv_fft(const Eigen::Ref<const Eigen::VectorXcd>& f,
                                 const Eigen::Ref<const Eigen::VectorXcd>& g,
                                 double dt)
{
    const std::size_t N = static_cast<std::size_t>(f.size());
    Eigen::VectorXcd out(static_cast<Eigen::Index>(N));
    out.setZero();
    if (N == 0) return out;

    const std::size_t L = 2 * N - 1;
    std::size_t Nfft = bcf::next_pow2(L);
    if (Nfft < 2) Nfft = 2;

    std::vector<cd> F(Nfft, cd{0.0, 0.0});
    std::vector<cd> G(Nfft, cd{0.0, 0.0});
    for (std::size_t k = 0; k < N; ++k) {
        F[k] = f(static_cast<Eigen::Index>(k));
        G[k] = g(static_cast<Eigen::Index>(k));
    }

    bcf::FFTPlan plan(Nfft);
    plan.exec_forward(F);
    plan.exec_forward(G);
    for (std::size_t k = 0; k < Nfft; ++k) F[k] *= G[k];
    plan.exec_inverse(F);

    for (std::size_t k = 0; k < N; ++k) {
        out(static_cast<Eigen::Index>(k)) = dt * F[k];
    }
    return out;
}

Eigen::VectorXcd compute_F_series_convolution_vec(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                                  double omega,
                                                  double dt)
{
    const std::size_t Nt = static_cast<std::size_t>(g1.size());
    Eigen::VectorXcd out(static_cast<Eigen::Index>(Nt));
    out.setZero();
    if (Nt == 0) return out;

    Eigen::ArrayXcd phase_plus(Nt), phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus(static_cast<Eigen::Index>(k))  = std::exp(cd{0.0, omega * tk});
        phase_minus(static_cast<Eigen::Index>(k)) = std::exp(cd{0.0, -omega * tk});
    }

    Eigen::VectorXcd g1_phase = (g1.array() * phase_minus).matrix();
    Eigen::VectorXcd prefix   = prefix_int_left_vec((g2.array() * phase_plus).matrix(), dt);
    Eigen::VectorXcd term1    = g1_phase.cwiseProduct(prefix);
    Eigen::VectorXcd term2    = causal_conv_fft(g1_phase, g2, dt);
    return term1 - term2;
}

Eigen::VectorXcd compute_C_series_convolution_vec(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                                  double omega,
                                                  double dt)
{
    const std::size_t Nt = static_cast<std::size_t>(g1.size());
    Eigen::VectorXcd out(static_cast<Eigen::Index>(Nt));
    out.setZero();
    if (Nt == 0) return out;

    Eigen::ArrayXcd phase_plus(Nt), phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus(static_cast<Eigen::Index>(k))  = std::exp(cd{0.0, omega * tk});
        phase_minus(static_cast<Eigen::Index>(k)) = std::exp(cd{0.0, -omega * tk});
    }

    Eigen::VectorXcd g1_phase = (g1.array() * phase_minus).matrix();
    Eigen::VectorXcd g2_conj  = g2.conjugate();
    Eigen::VectorXcd prefix   = prefix_int_left_vec((g2_conj.array() * phase_plus).matrix(), dt);
    Eigen::VectorXcd term1    = g1_phase.cwiseProduct(prefix);
    Eigen::VectorXcd term2    = causal_conv_fft(g1_phase, g2_conj, dt);
    return term1 - term2;
}

Eigen::VectorXcd compute_R_series_convolution_vec(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                                  double omega,
                                                  double dt)
{
    const std::size_t Nt = static_cast<std::size_t>(g1.size());
    Eigen::VectorXcd out(static_cast<Eigen::Index>(Nt));
    out.setZero();
    if (Nt == 0) return out;

    Eigen::ArrayXcd phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_minus(static_cast<Eigen::Index>(k)) = std::exp(cd{0.0, -omega * tk});
    }

    Eigen::VectorXcd prefix = prefix_int_left_vec((g2.array() * phase_minus).matrix(), dt);
    Eigen::VectorXcd term1  = g1.cwiseProduct(prefix);
    Eigen::VectorXcd P      = g1.cwiseProduct(g2);
    Eigen::VectorXcd prefixP= prefix_int_left_vec((P.array() * phase_minus).matrix(), dt);
    return term1 - prefixP;
}

} // namespace

// --------------------- Scalar-series (1x1) direct builders ------------------
Eigen::VectorXcd compute_F_series_direct(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                         const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                         double omega,
                                         double dt)
{
    const std::size_t Nt = static_cast<std::size_t>(g1.size());
    if (g2.size() != static_cast<Eigen::Index>(Nt)) {
        throw std::invalid_argument("compute_F_series_direct(vec): mismatched series lengths");
    }
    Eigen::VectorXcd F(static_cast<Eigen::Index>(Nt));
    F.setZero();
    if (Nt == 0) return F;
    const cd I{0.0, 1.0};
    std::vector<cd> phase_plus(Nt), phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus[k]  = std::exp(I * omega * tk);
        phase_minus[k] = std::exp(-I * omega * tk);
    }
    cd prefix_G2_op_phase{0.0, 0.0};
    for (std::size_t k = 0; k < Nt; ++k) {
        prefix_G2_op_phase += g2(static_cast<Eigen::Index>(k)) * phase_plus[k] * dt;
        const cd firstTermF = g1(static_cast<Eigen::Index>(k)) * prefix_G2_op_phase * phase_minus[k];
        cd secondTermF{0.0, 0.0};
        for (std::size_t m = 0; m <= k; ++m) {
            secondTermF += g1(static_cast<Eigen::Index>(k - m)) * g2(static_cast<Eigen::Index>(m)) * phase_minus[k - m];
        }
        F(static_cast<Eigen::Index>(k)) = firstTermF - dt * secondTermF;
    }
    return F;
}

Eigen::VectorXcd compute_C_series_direct(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                         const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                         double omega,
                                         double dt)
{
    const std::size_t Nt = static_cast<std::size_t>(g1.size());
    if (g2.size() != static_cast<Eigen::Index>(Nt)) {
        throw std::invalid_argument("compute_C_series_direct(vec): mismatched series lengths");
    }
    Eigen::VectorXcd C(static_cast<Eigen::Index>(Nt));
    C.setZero();
    if (Nt == 0) return C;
    const cd I{0.0, 1.0};
    std::vector<cd> phase_plus(Nt), phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus[k]  = std::exp(I * omega * tk);
        phase_minus[k] = std::exp(-I * omega * tk);
    }
    cd prefix_G2_conj_phase{0.0, 0.0};
    for (std::size_t k = 0; k < Nt; ++k) {
        prefix_G2_conj_phase += std::conj(g2(static_cast<Eigen::Index>(k))) * phase_plus[k] * dt;
        const cd firstTermC = g1(static_cast<Eigen::Index>(k)) * prefix_G2_conj_phase * phase_minus[k];
        cd secondTermC{0.0, 0.0};
        for (std::size_t m = 0; m <= k; ++m) {
            secondTermC += g1(static_cast<Eigen::Index>(k - m)) * std::conj(g2(static_cast<Eigen::Index>(m))) * phase_minus[k - m];
        }
        C(static_cast<Eigen::Index>(k)) = firstTermC - dt * secondTermC;
    }
    return C;
}

Eigen::VectorXcd compute_R_series_direct(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                         const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                         double omega,
                                         double dt)
{
    const std::size_t Nt = static_cast<std::size_t>(g1.size());
    if (g2.size() != static_cast<Eigen::Index>(Nt)) {
        throw std::invalid_argument("compute_R_series_direct(vec): mismatched series lengths");
    }
    Eigen::VectorXcd R(static_cast<Eigen::Index>(Nt));
    R.setZero();
    if (Nt == 0) return R;
    const cd I{0.0, 1.0};
    std::vector<cd> phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_minus[k] = std::exp(-I * omega * tk);
    }
    cd prefix_G2_phase_minus{0.0, 0.0};
    cd prefix_P_phase_minus{0.0, 0.0};
    for (std::size_t k = 0; k < Nt; ++k) {
        prefix_G2_phase_minus += g2(static_cast<Eigen::Index>(k)) * phase_minus[k] * dt;
        const cd term1R = g1(static_cast<Eigen::Index>(k)) * prefix_G2_phase_minus;
        const cd Pk     = g1(static_cast<Eigen::Index>(k)) * g2(static_cast<Eigen::Index>(k));
        prefix_P_phase_minus += Pk * phase_minus[k] * dt;
        const cd term2R = prefix_P_phase_minus;
        R(static_cast<Eigen::Index>(k)) = term1R - term2R;
    }
    return R;
}

Eigen::VectorXcd compute_F_series(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                  double omega,
                                  double dt,
                                  FCRMethod method)
{
    switch (method) {
        case FCRMethod::Direct:      return compute_F_series_direct(g1, g2, omega, dt);
        case FCRMethod::Convolution: return compute_F_series_convolution_vec(g1, g2, omega, dt);
        default:                     return compute_F_series_direct(g1, g2, omega, dt);
    }
}

Eigen::VectorXcd compute_C_series(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                  double omega,
                                  double dt,
                                  FCRMethod method)
{
    switch (method) {
        case FCRMethod::Direct:      return compute_C_series_direct(g1, g2, omega, dt);
        case FCRMethod::Convolution: return compute_C_series_convolution_vec(g1, g2, omega, dt);
        default:                     return compute_C_series_direct(g1, g2, omega, dt);
    }
}

Eigen::VectorXcd compute_R_series(const Eigen::Ref<const Eigen::VectorXcd>& g1,
                                  const Eigen::Ref<const Eigen::VectorXcd>& g2,
                                  double omega,
                                  double dt,
                                  FCRMethod method)
{
    switch (method) {
        case FCRMethod::Direct:      return compute_R_series_direct(g1, g2, omega, dt);
        case FCRMethod::Convolution: return compute_R_series_convolution_vec(g1, g2, omega, dt);
        default:                     return compute_R_series_direct(g1, g2, omega, dt);
    }
}

// --------------------- Matrix-series (general) direct builders --------------
std::vector<Matrix> compute_F_series_direct(const std::vector<Matrix>& G1,
                                            const std::vector<Matrix>& G2,
                                            double omega,
                                            double dt)
{
    const std::size_t Nt = G1.size();
    if (Nt == 0 || G2.size() != Nt) {
        throw std::invalid_argument("compute_F_series_direct: mismatched time-series lengths");
    }
    const auto rows = G1.front().rows();
    const auto cols = G1.front().cols();
    for (std::size_t k = 0; k < Nt; ++k) {
        if (G1[k].rows() != rows || G1[k].cols() != cols ||
            G2[k].rows() != rows || G2[k].cols() != cols) {
            throw std::invalid_argument("compute_F_series_direct: inconsistent matrix dimensions");
        }
    }

    const cd I{0.0, 1.0};
    std::vector<cd> phase_plus(Nt), phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus[k]  = std::exp(I * omega * tk);
        phase_minus[k] = std::exp(-I * omega * tk);
    }

    std::vector<Matrix> F(Nt, Matrix::Zero(rows, cols));
    Matrix prefix_G2_op_phase = Matrix::Zero(rows, cols);
    for (std::size_t k = 0; k < Nt; ++k) {
        const Matrix& G1k = G1[k];
        prefix_G2_op_phase += G2[k] * phase_plus[k] * dt;
        Matrix firstTermF = G1k * prefix_G2_op_phase * phase_minus[k];
        Matrix secondTermF = Matrix::Zero(rows, cols);
        for (std::size_t m = 0; m <= k; ++m) {
            const Matrix& G1km = G1[k - m];
            secondTermF += G1km * G2[m] * phase_minus[k - m];
        }
        F[k] = firstTermF - secondTermF * dt;
    }
    return F;
}

std::vector<Matrix> compute_C_series_direct(const std::vector<Matrix>& G1,
                                            const std::vector<Matrix>& G2_conj,
                                            double omega,
                                            double dt)
{
    const std::size_t Nt = G1.size();
    if (Nt == 0 || G2_conj.size() != Nt) {
        throw std::invalid_argument("compute_C_series_direct: mismatched time-series lengths");
    }
    const auto rows = G1.front().rows();
    const auto cols = G1.front().cols();
    for (std::size_t k = 0; k < Nt; ++k) {
        if (G1[k].rows() != rows || G1[k].cols() != cols ||
            G2_conj[k].rows() != rows || G2_conj[k].cols() != cols) {
            throw std::invalid_argument("compute_C_series_direct: inconsistent matrix dimensions");
        }
    }

    const cd I{0.0, 1.0};
    std::vector<cd> phase_plus(Nt), phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus[k]  = std::exp(I * omega * tk);
        phase_minus[k] = std::exp(-I * omega * tk);
    }

    std::vector<Matrix> C(Nt, Matrix::Zero(rows, cols));
    Matrix prefix_G2_conj_phase = Matrix::Zero(rows, cols);
    for (std::size_t k = 0; k < Nt; ++k) {
        const Matrix& G1k = G1[k];
        prefix_G2_conj_phase += G2_conj[k] * phase_plus[k] * dt;
        Matrix firstTermC = G1k * prefix_G2_conj_phase * phase_minus[k];
        Matrix secondTermC = Matrix::Zero(rows, cols);
        for (std::size_t m = 0; m <= k; ++m) {
            const Matrix& G1km = G1[k - m];
            secondTermC += G1km * G2_conj[m] * phase_minus[k - m];
        }
        C[k] = firstTermC - secondTermC * dt;
    }
    return C;
}

std::vector<Matrix> compute_R_series_direct(const std::vector<Matrix>& G1,
                                            const std::vector<Matrix>& G2,
                                            double omega,
                                            double dt)
{
    const std::size_t Nt = G1.size();
    if (Nt == 0 || G2.size() != Nt) {
        throw std::invalid_argument("compute_R_series_direct: mismatched time-series lengths");
    }
    const auto rows = G1.front().rows();
    const auto cols = G1.front().cols();
    for (std::size_t k = 0; k < Nt; ++k) {
        if (G1[k].rows() != rows || G1[k].cols() != cols ||
            G2[k].rows() != rows || G2[k].cols() != cols) {
            throw std::invalid_argument("compute_R_series_direct: inconsistent matrix dimensions");
        }
    }

    const cd I{0.0, 1.0};
    std::vector<cd> phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_minus[k] = std::exp(-I * omega * tk);
    }

    std::vector<Matrix> R(Nt, Matrix::Zero(rows, cols));
    Matrix prefix_G2_phase_minus = Matrix::Zero(rows, cols);
    Matrix prefix_P_phase_minus  = Matrix::Zero(rows, cols);
    for (std::size_t k = 0; k < Nt; ++k) {
        const Matrix& G1k = G1[k];
        prefix_G2_phase_minus += G2[k] * phase_minus[k] * dt;
        Matrix term1R = G1k * prefix_G2_phase_minus;
        Matrix Pk = G1k * G2[k];
        prefix_P_phase_minus += Pk * phase_minus[k] * dt;
        Matrix term2R = prefix_P_phase_minus;
        R[k] = term1R - term2R;
    }
    return R;
}

FCRSeries compute_FCR_time_series_direct(const std::vector<Matrix>& G1,
                                         const std::vector<Matrix>& G2,
                                         double omega,
                                         double dt,
                                         SpectralOp /*op2*/)
{
    FCRSeries out;
    out.F = compute_F_series_direct(G1, G2, omega, dt);
    std::vector<Matrix> G2_conj(G2.size());
    for (std::size_t k = 0; k < G2.size(); ++k) G2_conj[k] = G2[k].conjugate();
    out.C = compute_C_series_direct(G1, G2_conj, omega, dt);
    out.R = compute_R_series_direct(G1, G2, omega, dt);
    return out;
}

FCRSeries compute_FCR_time_series_convolution(const std::vector<Matrix>& G1,
                                              const std::vector<Matrix>& G2,
                                              double omega,
                                              double dt,
                                              SpectralOp op2)
{
    (void)op2;
    // TODO: full matrix convolution path (FFT + pagewise GEMM)
    return compute_FCR_time_series_direct(G1, G2, omega, dt, op2);
}

FCRSeries compute_FCR_time_series(const std::vector<Matrix>& G1,
                                  const std::vector<Matrix>& G2,
                                  double omega,
                                  double dt,
                                  SpectralOp op2,
                                  FCRMethod method)
{
    switch (method) {
        case FCRMethod::Direct:
            return compute_FCR_time_series_direct(G1, G2, omega, dt, op2);
        case FCRMethod::Convolution:
        default:
            return compute_FCR_time_series_convolution(G1, G2, omega, dt, op2);
    }
}

std::vector<Matrix> compute_F_series(const std::vector<Matrix>& G1,
                                     const std::vector<Matrix>& G2,
                                     double omega,
                                     double dt,
                                     FCRMethod method)
{
    switch (method) {
        case FCRMethod::Direct:      return compute_F_series_direct(G1, G2, omega, dt);
        case FCRMethod::Convolution: return compute_F_series_direct(G1, G2, omega, dt); // TODO
        default:                     return compute_F_series_direct(G1, G2, omega, dt);
    }
}

std::vector<Matrix> compute_C_series(const std::vector<Matrix>& G1,
                                     const std::vector<Matrix>& G2_conj,
                                     double omega,
                                     double dt,
                                     FCRMethod method)
{
    switch (method) {
        case FCRMethod::Direct:      return compute_C_series_direct(G1, G2_conj, omega, dt);
        case FCRMethod::Convolution: return compute_C_series_direct(G1, G2_conj, omega, dt); // TODO
        default:                     return compute_C_series_direct(G1, G2_conj, omega, dt);
    }
}

std::vector<Matrix> compute_R_series(const std::vector<Matrix>& G1,
                                     const std::vector<Matrix>& G2,
                                     double omega,
                                     double dt,
                                     FCRMethod method)
{
    switch (method) {
        case FCRMethod::Direct:      return compute_R_series_direct(G1, G2, omega, dt);
        case FCRMethod::Convolution: return compute_R_series_direct(G1, G2, omega, dt); // TODO
        default:                     return compute_R_series_direct(G1, G2, omega, dt);
    }
}

} // namespace taco::tcl4
