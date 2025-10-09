// gamma.hpp — Spectral kernel accumulator G(ω,t) = ∫_0^t e^{i ω τ} C(τ) dτ
// Header-only helpers for streaming and batch quadrature from discrete C(t).

#pragma once

#include <vector>
#include <complex>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <algorithm>

namespace taco::gamma {

using cplx = std::complex<double>;

// Streaming trapezoid accumulator for multiple ω values.
// Usage:
//   GammaTrapzAccumulator acc(dt, omegas);
//   acc.push_sample(C0); acc.push_sample(C1); ...;
//   const auto& G = acc.values();   // size = omegas.size()
struct GammaTrapzAccumulator {
    double dt{0.0};
    std::vector<double> omegas;          // ω_j
    std::vector<cplx> phi_step;          // e^{i ω_j dt} (cached)
    std::vector<cplx> phi;               // e^{i ω_j k dt}
    std::vector<cplx> G;                 // current integral for each ω_j
    cplx C_prev{0.0, 0.0};
    std::size_t k{0};
    cplx half_dt{0.0, 0.0};              // (dt/2)

    GammaTrapzAccumulator() = default;
    GammaTrapzAccumulator(double dt_, const std::vector<double>& ws)
        : dt(dt_), omegas(ws),
          phi_step(ws.size(), cplx{1.0,0.0}),
          phi(ws.size(), cplx{1.0, 0.0}),
          G(ws.size(), cplx{0.0, 0.0}),
          half_dt(dt_/2.0, 0.0)
    {
        if (!(dt > 0.0)) throw std::invalid_argument("GammaTrapzAccumulator: dt must be > 0");
        for (std::size_t j = 0; j < omegas.size(); ++j) {
            phi_step[j] = std::exp(cplx{0.0, omegas[j] * dt});
        }
    }

    // Reset to k=0 state
    void reset(double dt_, const std::vector<double>& ws) {
        dt = dt_; omegas = ws; k = 0; C_prev = cplx{0.0, 0.0};
        half_dt = cplx{dt_/2.0, 0.0};
        phi.assign(omegas.size(), cplx{1.0, 0.0});
        G.assign(omegas.size(), cplx{0.0, 0.0});
        phi_step.resize(omegas.size());
        for (std::size_t j = 0; j < omegas.size(); ++j) {
            phi_step[j] = std::exp(cplx{0.0, omegas[j] * dt});
        }
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
            const cplx phi_next = phi[j] * phi_step[j];
            G[j] += half_dt * (phi[j] * C_prev + phi_next * C_k);
            phi[j] = phi_next;
        }
        C_prev = C_k; ++k;
    }

    const std::vector<cplx>& values() const noexcept { return G; }
};

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

// Compute prefix integrals G_k(ω_j) for all k and a set of ω_j, using a single
// streaming pass (composite trapezoid). Returns M vectors (one per ω), each of
// length N = C.size(), with G[0]=0 and G[k] = ∫_0^{k·dt} e^{i ω t} C(t) dt.
inline std::vector<std::vector<cplx>>
compute_trapz_prefix_multi(const std::vector<cplx>& C,
                           double dt,
                           const std::vector<double>& omegas)
{
    const std::size_t N = C.size();
    const std::size_t M = omegas.size();
    std::vector<std::vector<cplx>> G(M);
    for (std::size_t j = 0; j < M; ++j) G[j].assign(N, cplx{0.0, 0.0});
    if (N < 2 || !(dt > 0.0) || M == 0) return G;

    GammaTrapzAccumulator acc(dt, omegas);
    acc.push_sample(C[0]);
    for (std::size_t k = 1; k < N; ++k) {
        acc.push_sample(C[k]);
        const auto& cur = acc.values();
        for (std::size_t j = 0; j < M; ++j) G[j][k] = cur[j];
    }
    return G;
}

// Omega-major batch path (cache-friendly when M << N):
// Returns an N x M column-major matrix (columns = omegas, rows = time index k)
// containing G(k,j) = ∫_0^{k·dt} e^{i ω_j t} C(t) dt with trapezoid updates.
inline Eigen::MatrixXcd
compute_trapz_prefix_multi_matrix(const std::vector<cplx>& C,
                                  double dt,
                                  const std::vector<double>& omegas)
{
    const std::size_t N = C.size();
    const std::size_t M = omegas.size();
    if (N == 0 || M == 0 || !(dt > 0.0)) return Eigen::MatrixXcd();
    const std::size_t NN = (N >= 2) ? N : 2; // ensure at least 2 rows for safe loops
    Eigen::MatrixXcd G(static_cast<Eigen::Index>(NN), static_cast<Eigen::Index>(M));
    // Initialize first row (k=0) to zero; if N==1 we truncate at return
    G.row(0).setZero();
    const cplx half_dt(dt / 2.0, 0.0);
    for (std::size_t j = 0; j < M; ++j) {
        const cplx step = std::exp(cplx{0.0, omegas[j] * dt});
        cplx phi = cplx{1.0, 0.0};
        cplx acc = cplx{0.0, 0.0};
        cplx prev = C[0];
        for (std::size_t k = 1; k < N; ++k) {
            const cplx phi_next = phi * step;
            acc += half_dt * (phi * prev + phi_next * C[k]);
            G(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(j)) = acc;
            phi = phi_next; prev = C[k];
        }
        // If N == 1, we only set G(0,j)=0 above; for N>=2, rows [1..N-1] were set.
        for (std::size_t k = N; k < NN; ++k) {
            // pad (no extra info), keeps shape consistent if N==1
            G(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(j)) = acc;
        }
    }
    // If N==1 return N x M (1 row). Otherwise return N x M.
    if (NN != N) {
        return G.topRows(static_cast<Eigen::Index>(N));
    }
    return G;
}

// Final-only multi-ω accumulator (no history): returns Γ(ω_j, T) for each ω_j.
inline std::vector<cplx>
compute_trapz_final_multi(const std::vector<cplx>& C,
                          double dt,
                          const std::vector<double>& omegas)
{
    const std::size_t N = C.size();
    const std::size_t M = omegas.size();
    std::vector<cplx> out(M, cplx{0.0, 0.0});
    if (N < 2 || M == 0 || !(dt > 0.0)) return out;
    const cplx half_dt(dt / 2.0, 0.0);
    for (std::size_t j = 0; j < M; ++j) {
        const cplx step = std::exp(cplx{0.0, omegas[j] * dt});
        cplx phi = cplx{1.0, 0.0};
        cplx acc = cplx{0.0, 0.0};
        cplx prev = C[0];
        for (std::size_t k = 1; k < N; ++k) {
            const cplx phi_next = phi * step;
            acc += half_dt * (phi * prev + phi_next * C[k]);
            phi = phi_next; prev = C[k];
        }
        out[j] = acc;
    }
    return out;
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

} // namespace taco::gamma
// --------------------------- Omega deduplication ----------------------------
struct OmegaDedup {
    std::vector<double> values;                     // unique omegas
    std::vector<std::size_t> map;                   // original index -> unique index
    std::vector<std::vector<std::size_t>> groups;   // unique index -> list of original indices
};

inline bool nearly_equal(double a, double b, double tol) noexcept {
    return std::abs(a - b) <= tol;
}

inline OmegaDedup deduplicate_omegas(const std::vector<double>& omegas, double tol = 1e-12) {
    const std::size_t M = omegas.size();
    std::vector<std::size_t> idx(M);
    for (std::size_t i = 0; i < M; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b){ return omegas[a] < omegas[b]; });
    OmegaDedup out;
    out.map.assign(M, 0);
    for (std::size_t k = 0; k < M; ++k) {
        const std::size_t i = idx[k];
        if (out.values.empty() || !nearly_equal(omegas[i], out.values.back(), tol)) {
            out.values.push_back(omegas[i]);
            out.groups.emplace_back();
        }
        const std::size_t u = out.values.size() - 1;
        out.map[i] = u;
        out.groups[u].push_back(i);
    }
    return out;
}
