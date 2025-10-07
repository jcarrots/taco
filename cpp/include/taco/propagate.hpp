#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <stdexcept>
#include <type_traits>

// Generic propagation tools for any "generator" type that exposes:
//   - std::size_t dimension() const
//   - double      current_time() const
//   - void        advance(double t1)
//   - void        apply(const Eigen::MatrixXcd& rho, Eigen::MatrixXcd& drho) const
// These utilities do not depend on a specific generator class (e.g., TCL2).

namespace taco::tcl {

inline void hermitize_and_normalize(Eigen::MatrixXcd& rho) {
    rho = 0.5 * (rho + rho.adjoint());
    const double tr = rho.trace().real();
    if (std::abs(tr) > 0.0) rho /= tr;
}

template<class Generator>
inline void ensure_time(Generator& gen, double t) {
    if (gen.current_time() < t) gen.advance(t);
}

// One RK4 step (monotone in time)
template<class Generator>
inline void step_rk4(Generator& gen, Eigen::MatrixXcd& rho, double t, double dt) {
    using Matrix = Eigen::MatrixXcd;
    const Eigen::Index dim = static_cast<Eigen::Index>(gen.dimension());
    Matrix k1(dim, dim), k2(dim, dim), k3(dim, dim), k4(dim, dim);

    ensure_time(gen, t);
    gen.apply(rho, k1);

    const double t12 = t + 0.5 * dt;
    Matrix tmp = rho + 0.5 * dt * k1;
    ensure_time(gen, t12);
    gen.apply(tmp, k2);

    tmp = rho + 0.5 * dt * k2;
    ensure_time(gen, t12);
    gen.apply(tmp, k3);

    const double t1 = t + dt;
    tmp = rho + dt * k3;
    ensure_time(gen, t1);
    gen.apply(tmp, k4);

    rho.noalias() += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    hermitize_and_normalize(rho);
}

// Build dense Liouvillian L (N^2 x N^2) at current generator time (small N only)
template<class Generator>
inline Eigen::MatrixXcd build_liouvillian(Generator& gen) {
    const Eigen::Index N  = static_cast<Eigen::Index>(gen.dimension());
    const Eigen::Index NN = N * N;
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(NN, NN);
    Eigen::MatrixXcd E = Eigen::MatrixXcd::Zero(N, N);
    Eigen::MatrixXcd dE(N, N);
    for (Eigen::Index j = 0; j < NN; ++j) {
        const Eigen::Index row = j % N;
        const Eigen::Index col = j / N;
        E.setZero();
        E(row, col) = 1.0;
        gen.apply(E, dE);
        Eigen::Map<const Eigen::VectorXcd> v(dE.data(), NN);
        L.col(j) = v;
    }
    return L;
}

// Convenience: build L after advancing generator to time t (monotone only)
template<class Generator>
inline Eigen::MatrixXcd build_liouvillian_at(Generator& gen, double t) {
    ensure_time(gen, t);
    return build_liouvillian(gen);
}

// One frozen-L exponential step: rho <- exp(dt * L(t)) * rho  (small N only)
template<class Generator>
inline void step_expm_frozen(Generator& gen, Eigen::MatrixXcd& rho, double t, double dt) {
    ensure_time(gen, t);
    const Eigen::Index N  = static_cast<Eigen::Index>(gen.dimension());
    const Eigen::Index NN = N * N;
    Eigen::MatrixXcd L = build_liouvillian(gen);
    Eigen::Map<const Eigen::VectorXcd> v_in(rho.data(), NN);
    Eigen::MatrixXcd M = (dt * L).exp();
    Eigen::VectorXcd v_out = M * v_in;
    Eigen::Map<Eigen::MatrixXcd>(rho.data(), N, N) = Eigen::Map<const Eigen::MatrixXcd>(v_out.data(), N, N);
    hermitize_and_normalize(rho);
    ensure_time(gen, t + dt);
}

// -------------------- Precompute-and-apply exponential helpers ---------------

// Precompute the matrix exponential M = exp(dt * L) (N^2 x N^2). Small N only.
inline Eigen::MatrixXcd precompute_expm(const Eigen::MatrixXcd& L, double dt) {
    return (dt * L).exp();
}

// Apply a precomputed exponential M to rho: rho <- M * vec(rho) reshaped.
inline void apply_precomputed_expm(const Eigen::MatrixXcd& M, Eigen::MatrixXcd& rho) {
    const Eigen::Index N  = rho.rows();
    const Eigen::Index NN = N * N;
    Eigen::Map<const Eigen::VectorXcd> v_in(rho.data(), NN);
    Eigen::VectorXcd v_out = M * v_in;
    Eigen::Map<Eigen::MatrixXcd>(rho.data(), N, N) = Eigen::Map<const Eigen::MatrixXcd>(v_out.data(), N, N);
}

// Recommended usage for frozen-L scheme:
//   1) L = build_liouvillian_at(gen, t)
//   2) M = precompute_expm(L, dt)
//   3) for steps: apply_precomputed_expm(M, rho); hermitize_and_normalize(rho);
//      (if L changes with time, rebuild L and M at new times as needed.)

// Propagate using RK4 (fixed step). Optional on_sample(t,rho) callback.
template<class Generator>
inline void propagate_rk4(Generator& gen,
                          Eigen::MatrixXcd& rho,
                          double t0,
                          double tf,
                          double dt,
                          const std::function<void(double,const Eigen::MatrixXcd&)>& on_sample = {},
                          std::size_t sample_every = 1) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_rk4: dt must be > 0");
    double t = t0;
    std::size_t step = 0;
    while (t + 0.5 * dt <= tf + 1e-15) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, rho);
        step_rk4(gen, rho, t, dt);
        t += dt;
        ++step;
    }
    ensure_time(gen, tf);
    hermitize_and_normalize(rho);
    if (on_sample) on_sample(t, rho);
}

// Propagate using frozen-L exponential per step (small N only)
template<class Generator>
inline void propagate_expm(Generator& gen,
                           Eigen::MatrixXcd& rho,
                           double t0,
                           double tf,
                           double dt,
                           const std::function<void(double,const Eigen::MatrixXcd&)>& on_sample = {},
                           std::size_t sample_every = 1) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_expm: dt must be > 0");
    double t = t0;
    std::size_t step = 0;
    while (t + 0.5 * dt <= tf + 1e-15) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, rho);
        step_expm_frozen(gen, rho, t, dt);
        t += dt;
        ++step;
    }
    ensure_time(gen, tf);
    hermitize_and_normalize(rho);
    if (on_sample) on_sample(t, rho);
}

} // namespace taco::tcl
