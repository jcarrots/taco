// bath_models.hpp — Common bath models and utilities to build C(t) and Γ(ω)

#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "taco/correlation_fft.hpp"
#include "taco/bath_tabulated.hpp"
#include "taco/gamma.hpp"
#include "taco/system.hpp"
#include "taco/generator.hpp"

namespace taco::bath {

// --------------------------- Spectral densities -----------------------------

struct OhmicDrude {
    double alpha{0.05};     // coupling strength
    double omega_c{5.0};    // cutoff frequency
    // J(ω) = 2 α ω exp(-ω/ωc) for ω>0; 0 otherwise
    inline double J(double w) const noexcept {
        return (w > 0.0) ? (2.0 * alpha * w * std::exp(-w / omega_c)) : 0.0;
    }
};

// --------------------------- Builders: C(t) and Γ(ω) ------------------------

template<class JModel>
inline TabulatedCorrelation build_correlation_from_J(std::size_t rank,
                                                     std::size_t N,
                                                     double dt,
                                                     const JModel& model,
                                                     double beta)
{
    auto Jcall = [&](double w){ return model.J(w); };
    return TabulatedCorrelation::from_spectral(rank, N, dt, Jcall, beta);
}

// Compute asymptotic Γ(ω) per system frequency bucket by integrating C(t)
// via prefix trapezoid (returns Γ(ω, T) at T=N·dt). Assumes diagonal (α=β) channels.
template<class CorrelationLike>
inline tcl2::SpectralKernels build_spectral_kernels_from_correlation(const sys::System& system,
                                                                     const CorrelationLike& corr,
                                                                     const std::vector<double>& t)
{
    const std::size_t buckets = system.fidx.buckets.size();
    const std::size_t channels = system.A_eig_parts.size();
    if (channels == 0) throw std::invalid_argument("build_spectral_kernels_from_correlation: no channels");
    if (t.size() < 2) throw std::invalid_argument("build_spectral_kernels_from_correlation: time grid too small");
    const double dt = t[1] - t[0];

    // Build scalar C(t) assuming diagonal identical channels; read from corr(τ,0,0)
    std::vector<std::complex<double>> C(t.size());
    for (std::size_t k = 0; k < t.size(); ++k) C[k] = corr(t[k], 0, 0);

    tcl2::SpectralKernels K;
    K.buckets.resize(buckets);
    for (std::size_t b = 0; b < buckets; ++b) {
        const double w = system.fidx.buckets[b].omega;
        auto G = gamma::compute_trapz(C, dt, w);
        const std::complex<double> Ginf = (G.empty() ? std::complex<double>(0.0,0.0) : G.back());
        K.buckets[b].omega = w;
        K.buckets[b].Gamma = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(channels), static_cast<Eigen::Index>(channels));
        for (std::size_t a = 0; a < channels; ++a) K.buckets[b].Gamma(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(a)) = Ginf;
    }
    return K;
}

// Convenience: from spectral density model J and β
template<class JModel>
inline tcl2::SpectralKernels build_spectral_kernels_from_J(const sys::System& system,
                                                           std::size_t N,
                                                           double dt,
                                                           const JModel& model,
                                                           double beta)
{
    auto tab = build_correlation_from_J(system.A_eig_parts.size(), N, dt, model, beta);
    return build_spectral_kernels_from_correlation(system, tab, tab.times());
}

// Time series of SpectralKernels: Γ(ω, t_k) for every bucket ω and time index k.
// Efficiently computed via a single streaming pass using GammaTrapzAccumulator.
template<class CorrelationLike>
inline std::vector<tcl2::SpectralKernels>
build_spectral_kernels_prefix_series(const sys::System& system,
                                     const CorrelationLike& corr,
                                     const std::vector<double>& t)
{
    const std::size_t buckets = system.fidx.buckets.size();
    const std::size_t channels = system.A_eig_parts.size();
    if (channels == 0) throw std::invalid_argument("build_spectral_kernels_prefix_series: no channels");
    if (t.size() < 2) throw std::invalid_argument("build_spectral_kernels_prefix_series: time grid too small");
    const double dt = t[1] - t[0];

    // Pack bucket omegas once
    std::vector<double> omegas(buckets);
    for (std::size_t b = 0; b < buckets; ++b) omegas[b] = system.fidx.buckets[b].omega;

    // Tabulate scalar C(t) from the correlation (assumed diagonal identical channels)
    std::vector<std::complex<double>> C(t.size());
    for (std::size_t k = 0; k < t.size(); ++k) C[k] = corr(t[k], 0, 0);

    // Compute Γ prefix for all omegas in one pass
    auto G = gamma::compute_trapz_prefix_multi(C, dt, omegas); // size: buckets x Nt
    const std::size_t Nt = t.size();

    std::vector<tcl2::SpectralKernels> series(Nt);
    for (std::size_t k = 0; k < Nt; ++k) {
        auto& Kk = series[k];
        Kk.buckets.resize(buckets);
        for (std::size_t b = 0; b < buckets; ++b) {
            Kk.buckets[b].omega = omegas[b];
            Kk.buckets[b].Gamma = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(channels), static_cast<Eigen::Index>(channels));
            for (std::size_t a = 0; a < channels; ++a) Kk.buckets[b].Gamma(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(a)) = G[b][k];
        }
    }
    return series;
}

} // namespace taco::bath
