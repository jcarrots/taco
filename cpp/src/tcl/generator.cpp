// generator.cpp — Implementation of TCL2 superoperator builders

#include "taco/generator.hpp"

#include <stdexcept>
#include <vector>

#include "taco/ops.hpp"

namespace taco::tcl2 {

using Matrix = Eigen::MatrixXcd;

namespace {

inline void validate_inputs(const sys::System& system,
                            const SpectralKernels& kernels)
{
    const std::size_t bucket_count = system.fidx.buckets.size();
    const std::size_t channel_count = system.A_eig_parts.size();
    if (channel_count == 0) {
        throw std::invalid_argument("System must contain at least one jump operator");
    }
    for (const auto& parts : system.A_eig_parts) {
        if (parts.size() != bucket_count) {
            throw std::invalid_argument("System spectral decomposition does not match bucket count");
        }
    }
    if (kernels.buckets.size() != bucket_count) {
        throw std::invalid_argument("SpectralKernels bucket count does not match system frequency buckets");
    }
    for (const auto& bucket : kernels.buckets) {
        if (bucket.Gamma.rows() != static_cast<Eigen::Index>(channel_count) ||
            bucket.Gamma.cols() != static_cast<Eigen::Index>(channel_count)) {
            throw std::invalid_argument("SpectralKernels Gamma matrix has incorrect shape");
        }
    }
}

} // namespace

Matrix build_lamb_shift(const sys::System& system,
                        const SpectralKernels& kernels,
                        double imag_cutoff)
{
    validate_inputs(system, kernels);

    const std::size_t dim = system.eig.dim;
    const std::size_t channels = system.A_eig_parts.size();
    Matrix H_ls = Matrix::Zero(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));

    for (std::size_t b = 0; b < kernels.buckets.size(); ++b) {
        const auto& bucket_kernel = kernels.buckets[b];

        std::vector<const Matrix*> slices(channels, nullptr);
        std::vector<Matrix> daggers(channels);
        for (std::size_t alpha = 0; alpha < channels; ++alpha) {
            const Matrix& A_alpha_w = system.A_eig_parts[alpha][b];
            if (A_alpha_w.size() == 0 || A_alpha_w.squaredNorm() == 0.0) continue;
            slices[alpha] = &A_alpha_w;
            daggers[alpha] = A_alpha_w.adjoint();
        }

        for (std::size_t alpha = 0; alpha < channels; ++alpha) {
            if (!slices[alpha]) continue;
            const Matrix& A_alpha_dag = daggers[alpha];
            for (std::size_t beta = 0; beta < channels; ++beta) {
                if (!slices[beta]) continue;
                const double S = bucket_kernel.Gamma(alpha, beta).imag();
                if (std::abs(S) <= imag_cutoff) continue;
                const Matrix AdagA = A_alpha_dag * (*slices[beta]);
                if (AdagA.squaredNorm() == 0.0) continue;
                H_ls.noalias() += S * AdagA;
            }
        }
    }

    return ops::hermitize(H_ls);
}

Matrix build_unitary_superop(const sys::System& system,
                             const Matrix& H_eff)
{
    const std::size_t dim = system.eig.dim;
    const Eigen::Index N = static_cast<Eigen::Index>(dim);
    Matrix I = Matrix::Identity(N, N);
    Matrix left = ops::kron(I, H_eff);
    Matrix right = ops::kron(H_eff.transpose(), I);
    return std::complex<double>(0.0, -1.0) * (left - right);
}

Matrix build_dissipator_superop(const sys::System& system,
                                const SpectralKernels& kernels,
                                double gamma_cutoff,
                                Matrix* lamb_shift_out)
{
    validate_inputs(system, kernels);

    const std::size_t dim = system.eig.dim;
    const std::size_t channels = system.A_eig_parts.size();
    const Eigen::Index N = static_cast<Eigen::Index>(dim);
    const Matrix I = Matrix::Identity(N, N);
    Matrix L = Matrix::Zero(N * N, N * N);

    if (lamb_shift_out) {
        *lamb_shift_out = Matrix::Zero(N, N);
    }

    for (std::size_t b = 0; b < kernels.buckets.size(); ++b) {
        const auto& bucket_kernel = kernels.buckets[b];

        std::vector<const Matrix*> slices(channels, nullptr);
        std::vector<Matrix> daggers(channels);
        std::vector<Matrix> conjugates(channels);
        for (std::size_t alpha = 0; alpha < channels; ++alpha) {
            const Matrix& A_alpha_w = system.A_eig_parts[alpha][b];
            if (A_alpha_w.size() == 0 || A_alpha_w.squaredNorm() == 0.0) continue;
            slices[alpha] = &A_alpha_w;
            daggers[alpha] = A_alpha_w.adjoint();
            conjugates[alpha] = A_alpha_w.conjugate();
        }

        for (std::size_t alpha = 0; alpha < channels; ++alpha) {
            if (!slices[alpha]) continue;
            const Matrix& A_alpha_dag = daggers[alpha];
            const Matrix& A_alpha_conj = conjugates[alpha];

            for (std::size_t beta = 0; beta < channels; ++beta) {
                if (!slices[beta]) continue;
                const std::complex<double> G = bucket_kernel.Gamma(alpha, beta);
                const double gamma = 2.0 * G.real();
                const double S = G.imag();
                const bool need_jump = std::abs(gamma) > gamma_cutoff;
                const bool need_ls = lamb_shift_out && std::abs(S) > gamma_cutoff;
                if (!need_jump && !need_ls) continue;

                const Matrix& A_beta_w = *slices[beta];
                const Matrix AdagA = A_alpha_dag * A_beta_w;
                if (AdagA.squaredNorm() == 0.0) {
                    continue;
                }

                if (need_ls) {
                    lamb_shift_out->noalias() += S * AdagA;
                }

                if (!need_jump) continue;

                const Matrix term_jump = ops::kron(A_alpha_conj, A_beta_w);
                const Matrix term_left = ops::kron(I, AdagA);
                const Matrix term_right = ops::kron(AdagA.transpose(), I);
                L.noalias() += gamma * (term_jump - 0.5 * term_left - 0.5 * term_right);
            }
        }
    }

    if (lamb_shift_out) {
        *lamb_shift_out = ops::hermitize(*lamb_shift_out);
    }

    return L;
}

TCL2Components build_tcl2_components(const sys::System& system,
                                     const SpectralKernels& kernels,
                                     double cutoff)
{
    validate_inputs(system, kernels);

    Matrix diag_eps = system.eig.eps.asDiagonal().toDenseMatrix().cast<std::complex<double>>();
    Matrix H_ls;
    Matrix L_diss = build_dissipator_superop(system, kernels, cutoff, &H_ls);
    Matrix H_eff = diag_eps + H_ls;
    Matrix L_unitary = build_unitary_superop(system, H_eff);

    return TCL2Components{std::move(L_unitary), std::move(L_diss), std::move(H_ls)};
}

} // namespace taco::tcl2
