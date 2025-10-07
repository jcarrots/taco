// gamma.hpp — Spectral kernel accumulator G(ω,t) = ∫_0^t e^{i ω τ} C(τ) dτ
// Header-only helpers for streaming and batch quadrature from discrete C(t).

#pragma once

#include <vector>
#include <complex>
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace taco::gamma {

using cplx = std::complex<double>;

// Compute prefix integrals G_k for k=0..N from samples C_k = C(k·dt)
// via composite trapezoid rule. G_0 = 0.
inline std::vector<cplx>
compute_trapz(const std::vector<cplx>& C, double dt, double omega)
{
    const std::size_t N = C.size();
    std::vector<cplx> G(N, cplx{0.0, 0.0});
    if (N < 2 || !(dt > 0.0)) return G;

    const cplx iomega_dt{0.0, omega * dt};
    const cplx phi_step = std::exp(iomega_dt);   // e^{i ω dt}

    cplx phi = cplx{1.0, 0.0};                  // e^{i ω · 0}
    for (std::size_t k = 1; k < N; ++k) {
        const cplx phi_next = phi * phi_step;    // e^{i ω k dt}
        G[k] = G[k-1] + dt * cplx{0.5, 0.0} * (phi * C[k-1] + phi_next * C[k]);
        phi = phi_next;
    }
    return G;
}

// Compute prefix integrals with composite Simpson where possible.
// If hold_odd=true, G at odd indices equals previous even; otherwise apply
// a trapezoid update for the last interval (mixed rule).
inline std::vector<cplx>
compute_simpson(const std::vector<cplx>& C, double dt, double omega, bool hold_odd = false)
{
    const std::size_t N = C.size();
    std::vector<cplx> G(N, cplx{0.0, 0.0});
    if (N < 2 || !(dt > 0.0)) return G;

    const cplx phi_step = std::exp(cplx{0.0, omega * dt});
    // Track phases at k-2, k-1, k
    cplx phi0 = cplx{1.0, 0.0};
    cplx phi1 = phi0 * phi_step;
    // For k=1 (odd), optionally do a trapezoid update
    if (!hold_odd) {
        G[1] = dt * cplx{0.5, 0.0} * (phi0 * C[0] + phi1 * C[1]);
    } else {
        G[1] = G[0];
    }
    for (std::size_t k = 2; k < N; ++k) {
        cplx phi2 = phi1 * phi_step;
        if ((k % 2) == 0) {
            // Simpson block over [k-2, k-1, k]
            G[k] = G[k-2] + (dt / 3.0) * (phi0 * C[k-2] + 4.0 * (phi1 * C[k-1]) + phi2 * C[k]);
        } else {
            // Odd index
            if (hold_odd) {
                G[k] = G[k-1];
            } else {
                G[k] = G[k-1] + dt * cplx{0.5, 0.0} * (phi1 * C[k-1] + phi2 * C[k]);
            }
        }
        phi0 = phi1; phi1 = phi2;
    }
    return G;
}

// Streaming trapezoid accumulator for multiple ω values.
// Usage:
//   GammaTrapzAccumulator acc(dt, omegas);
//   acc.push_sample(C0); acc.push_sample(C1); ...;
//   const auto& G = acc.values();   // size = omegas.size()
struct GammaTrapzAccumulator {
    double dt{0.0};
    std::vector<double> omegas;          // ω_j
    std::vector<cplx> phi;               // e^{i ω_j k dt}
    std::vector<cplx> G;                 // current integral for each ω_j
    cplx C_prev{0.0, 0.0};
    std::size_t k{0};

    GammaTrapzAccumulator() = default;
    GammaTrapzAccumulator(double dt_, const std::vector<double>& ws)
        : dt(dt_), omegas(ws), phi(ws.size(), cplx{1.0, 0.0}), G(ws.size(), cplx{0.0, 0.0})
    {
        if (!(dt > 0.0)) throw std::invalid_argument("GammaTrapzAccumulator: dt must be > 0");
    }

    // Reset to k=0 state
    void reset(double dt_, const std::vector<double>& ws) {
        dt = dt_; omegas = ws; k = 0; C_prev = cplx{0.0, 0.0};
        phi.assign(omegas.size(), cplx{1.0, 0.0});
        G.assign(omegas.size(), cplx{0.0, 0.0});
        if (!(dt > 0.0)) throw std::invalid_argument("GammaTrapzAccumulator: dt must be > 0");
    }

    // Push next sample C_k; k increments each call.
    // On first call (k==0→1), performs a single trapezoid update over [0,dt].
    void push_sample(cplx C_k) {
        if (k == 0) {
            // First sample: just store and advance index
            C_prev = C_k; k = 1; return;
        }
        const std::size_t m = omegas.size();
        for (std::size_t j = 0; j < m; ++j) {
            const cplx phi_next = phi[j] * std::exp(cplx{0.0, omegas[j] * dt});
            G[j] += dt * cplx{0.5, 0.0} * (phi[j] * C_prev + phi_next * C_k);
            phi[j] = phi_next;
        }
        C_prev = C_k; ++k;
    }

    const std::vector<cplx>& values() const noexcept { return G; }
};

} // namespace taco::gamma

