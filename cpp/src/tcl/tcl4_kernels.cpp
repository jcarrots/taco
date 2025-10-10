#include "taco/tcl4_kernels.hpp"

#include <stdexcept>

namespace taco::tcl4 {
namespace {

using Matrix = Eigen::MatrixXcd;

Matrix apply_op(const Matrix& M, SpectralOp op)
{
    switch (op) {
        case SpectralOp::Identity:
            return M;
        case SpectralOp::Transpose:
            return M.transpose();
        case SpectralOp::Conjugate:
            return M.conjugate();
        case SpectralOp::Hermitian:
            return M.adjoint();
    }
    throw std::invalid_argument("Unknown SpectralOp");
}

} // namespace

FCRSeries compute_FCR_time_series(const std::vector<Matrix>& G1,
                                  const std::vector<Matrix>& G2,
                                  double omega,
                                  double dt,
                                  SpectralOp op2)
{
    const std::size_t Nt = G1.size();
    if (Nt == 0 || G2.size() != Nt) {
        throw std::invalid_argument("compute_FCR_time_series: mismatched time-series lengths");
    }

    const auto rows = G1.front().rows();
    const auto cols = G1.front().cols();
    for (std::size_t k = 0; k < Nt; ++k) {
        if (G1[k].rows() != rows || G1[k].cols() != cols ||
            G2[k].rows() != rows || G2[k].cols() != cols) {
            throw std::invalid_argument("compute_FCR_time_series: inconsistent matrix dimensions");
        }
    }

    const std::complex<double> I(0.0, 1.0);
    std::vector<std::complex<double>> phase_plus(Nt);
    std::vector<std::complex<double>> phase_minus(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        const double tk = static_cast<double>(k) * dt;
        phase_plus[k]  = std::exp(I * omega * tk);
        phase_minus[k] = std::exp(-I * omega * tk);
    }

    std::vector<Matrix> G2_op(Nt);
    std::vector<Matrix> G2_conj(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        G2_op[k]   = apply_op(G2[k], op2);
        G2_conj[k] = G2[k].conjugate();
    }

    FCRSeries series;
    series.F.resize(Nt, Matrix::Zero(rows, cols));
    series.C.resize(Nt, Matrix::Zero(rows, cols));
    series.R.resize(Nt, Matrix::Zero(rows, cols));

    Matrix prefix_G2_op_phase = Matrix::Zero(rows, cols);
    Matrix prefix_G2_conj_phase = Matrix::Zero(rows, cols);
    Matrix prefix_G2_phase_minus = Matrix::Zero(rows, cols);
    Matrix prefix_P_phase_minus = Matrix::Zero(rows, cols);

    for (std::size_t k = 0; k < Nt; ++k) {
        const Matrix& G1k = G1[k];

        prefix_G2_op_phase += G2_op[k] * phase_plus[k] * dt;
        Matrix firstTermF = G1k * prefix_G2_op_phase * phase_minus[k];
        Matrix secondTermF = Matrix::Zero(rows, cols);
        for (std::size_t m = 0; m <= k; ++m) {
            const Matrix& G1km = G1[k - m];
            secondTermF += G1km * G2_op[m] * phase_minus[k - m];
        }
        series.F[k] = firstTermF - secondTermF * dt;

        prefix_G2_conj_phase += G2_conj[k] * phase_plus[k] * dt;
        Matrix firstTermC = G1k * prefix_G2_conj_phase * phase_minus[k];
        Matrix secondTermC = Matrix::Zero(rows, cols);
        for (std::size_t m = 0; m <= k; ++m) {
            const Matrix& G1km = G1[k - m];
            secondTermC += G1km * G2_conj[m] * phase_minus[k - m];
        }
        series.C[k] = firstTermC - secondTermC * dt;

        prefix_G2_phase_minus += G2[k] * phase_minus[k] * dt;
        Matrix term1R = G1k * prefix_G2_phase_minus;
        Matrix Pk = G1k * G2[k];
        prefix_P_phase_minus += Pk * phase_minus[k] * dt;
        Matrix term2R = prefix_P_phase_minus;
        series.R[k] = term1R - term2R;
    }

    return series;
}

} // namespace taco::tcl4

