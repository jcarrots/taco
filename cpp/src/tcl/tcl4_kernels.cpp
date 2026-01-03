#include "taco/tcl4_kernels.hpp"

#include <stdexcept>
#include <vector>
#include <complex>

#include "taco/correlation_fft.hpp"

namespace taco::tcl4 {
namespace {

using Matrix = Eigen::MatrixXcd;
using cd     = std::complex<double>;

std::size_t g_fcr_fft_pad_factor = 0;

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
    std::size_t Nfft = 0;
    if (g_fcr_fft_pad_factor > 0) {
        std::size_t target = g_fcr_fft_pad_factor * N;
        if (target < L) target = L;
        Nfft = bcf::next_pow2(target);
    } else {
        Nfft = bcf::next_pow2(L);
    }
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

inline void fft_time_axis_inplace(std::vector<cd>& data,
                                  std::size_t Nfft,
                                  std::size_t block_size,
                                  bool forward,
                                  std::vector<cd>& scratch,
                                  bcf::FFTPlan& plan_1d,
                                  std::vector<cd>& tmp_1d)
{
#if defined(TACO_POCKETFFT_HEADER)
    (void)plan_1d;
    (void)tmp_1d;
    scratch.resize(Nfft * block_size);
    pocketfft::shape_t shape{ static_cast<std::ptrdiff_t>(Nfft),
                              static_cast<std::ptrdiff_t>(block_size) };
    pocketfft::stride_t stride{ static_cast<std::ptrdiff_t>(block_size * sizeof(cd)),
                                static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::axes_t axes{ 0 };
    pocketfft::c2c(shape, stride, stride, axes,
                   /*forward=*/forward,
                   reinterpret_cast<const cd*>(data.data()),
                   scratch.data(),
                   forward ? 1.0 : 1.0 / static_cast<double>(Nfft));
    data.swap(scratch);
#else
    (void)scratch;
    tmp_1d.resize(Nfft);
    for (std::size_t e = 0; e < block_size; ++e) {
        for (std::size_t n = 0; n < Nfft; ++n) tmp_1d[n] = data[n * block_size + e];
        if (forward) plan_1d.exec_forward(tmp_1d);
        else plan_1d.exec_inverse(tmp_1d);
        for (std::size_t n = 0; n < Nfft; ++n) data[n * block_size + e] = tmp_1d[n];
    }
#endif
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

void set_fcr_fft_pad_factor(std::size_t factor) {
    g_fcr_fft_pad_factor = factor;
}

std::size_t get_fcr_fft_pad_factor() {
    return g_fcr_fft_pad_factor;
}

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

    const std::size_t Nt = G1.size();
    if (Nt == 0 || G2.size() != Nt) {
        throw std::invalid_argument("compute_FCR_time_series_convolution: mismatched time-series lengths");
    }
    const auto rows = G1.front().rows();
    const auto cols = G1.front().cols();
    for (std::size_t k = 0; k < Nt; ++k) {
        if (G1[k].rows() != rows || G1[k].cols() != cols ||
            G2[k].rows() != rows || G2[k].cols() != cols) {
            throw std::invalid_argument("compute_FCR_time_series_convolution: inconsistent matrix dimensions");
        }
    }

    const cd I{0.0, 1.0};
    std::vector<cd> phase_plus(Nt), phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus[k]  = std::exp(I * omega * tk);
        phase_minus[k] = std::exp(-I * omega * tk);
    }

    const std::size_t L = 2 * Nt - 1;
    std::size_t Nfft = 0;
    if (g_fcr_fft_pad_factor > 0) {
        std::size_t target = g_fcr_fft_pad_factor * Nt;
        if (target < L) target = L;
        Nfft = bcf::next_pow2(target);
    } else {
        Nfft = bcf::next_pow2(L);
    }
    if (Nfft < 2) Nfft = 2;

    const std::size_t block_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    std::vector<cd> A(Nfft * block_size, cd{0.0, 0.0});
    std::vector<cd> B(Nfft * block_size, cd{0.0, 0.0});
    std::vector<cd> B_conj(Nfft * block_size, cd{0.0, 0.0});
    for (std::size_t k = 0; k < Nt; ++k) {
        cd* arow = A.data() + k * block_size;
        cd* brow = B.data() + k * block_size;
        cd* crow = B_conj.data() + k * block_size;
        const cd* g1 = G1[k].data();
        const cd* g2 = G2[k].data();
        const cd sA = phase_minus[k];
        for (std::size_t e = 0; e < block_size; ++e) {
            arow[e] = g1[e] * sA;
            brow[e] = g2[e];
            crow[e] = std::conj(g2[e]);
        }
    }

    bcf::FFTPlan plan(Nfft);
    std::vector<cd> fft_scratch;
    std::vector<cd> fft_tmp_1d;
    fft_time_axis_inplace(A, Nfft, block_size, /*forward=*/true, fft_scratch, plan, fft_tmp_1d);
    fft_time_axis_inplace(B, Nfft, block_size, /*forward=*/true, fft_scratch, plan, fft_tmp_1d);
    fft_time_axis_inplace(B_conj, Nfft, block_size, /*forward=*/true, fft_scratch, plan, fft_tmp_1d);

    std::vector<cd> P(Nfft * block_size, cd{0.0, 0.0});

    // -------- F(t): convolution term via FFT + pagewise GEMM --------
    for (std::size_t p = 0; p < Nfft; ++p) {
        const cd* Ap = A.data() + p * block_size;
        const cd* Bp = B.data() + p * block_size;
        cd* Pp = P.data() + p * block_size;
        Eigen::Map<const Matrix> Ahat(Ap, rows, cols);
        Eigen::Map<const Matrix> Bhat(Bp, rows, cols);
        Eigen::Map<Matrix> Phat(Pp, rows, cols);
        Phat.noalias() = Ahat * Bhat;
    }
    fft_time_axis_inplace(P, Nfft, block_size, /*forward=*/false, fft_scratch, plan, fft_tmp_1d);

    FCRSeries out;
    out.F.assign(Nt, Matrix::Zero(rows, cols));
    Matrix prefix_G2_op_phase = Matrix::Zero(rows, cols);
    for (std::size_t k = 0; k < Nt; ++k) {
        prefix_G2_op_phase += G2[k] * phase_plus[k] * dt;
        out.F[k].noalias() = G1[k] * prefix_G2_op_phase;
        out.F[k] *= phase_minus[k];
        Eigen::Map<const Matrix> conv_k(P.data() + k * block_size, rows, cols);
        out.F[k] -= conv_k * dt;
    }

    // -------- C(t): convolution term via FFT + pagewise GEMM --------
    for (std::size_t p = 0; p < Nfft; ++p) {
        const cd* Ap = A.data() + p * block_size;
        const cd* Bp = B_conj.data() + p * block_size;
        cd* Pp = P.data() + p * block_size;
        Eigen::Map<const Matrix> Ahat(Ap, rows, cols);
        Eigen::Map<const Matrix> Bhat(Bp, rows, cols);
        Eigen::Map<Matrix> Phat(Pp, rows, cols);
        Phat.noalias() = Ahat * Bhat;
    }
    fft_time_axis_inplace(P, Nfft, block_size, /*forward=*/false, fft_scratch, plan, fft_tmp_1d);

    out.C.assign(Nt, Matrix::Zero(rows, cols));
    Matrix prefix_G2_conj_phase = Matrix::Zero(rows, cols);
    for (std::size_t k = 0; k < Nt; ++k) {
        prefix_G2_conj_phase += G2[k].conjugate() * phase_plus[k] * dt;
        out.C[k].noalias() = G1[k] * prefix_G2_conj_phase;
        out.C[k] *= phase_minus[k];
        Eigen::Map<const Matrix> conv_k(P.data() + k * block_size, rows, cols);
        out.C[k] -= conv_k * dt;
    }

    // -------- R(t): prefix-integral form (already O(Nt)) --------
    out.R.assign(Nt, Matrix::Zero(rows, cols));
    Matrix prefix_G2_phase_minus = Matrix::Zero(rows, cols);
    Matrix prefix_P_phase_minus  = Matrix::Zero(rows, cols);
    for (std::size_t k = 0; k < Nt; ++k) {
        const cd w = phase_minus[k] * dt;
        prefix_G2_phase_minus += G2[k] * w;
        out.R[k].noalias() = G1[k] * prefix_G2_phase_minus;
        prefix_P_phase_minus.noalias() += (G1[k] * G2[k]) * w;
        out.R[k] -= prefix_P_phase_minus;
    }

    return out;
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
