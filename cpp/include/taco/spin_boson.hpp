// spin_boson.hpp — Helpers for constructing spin-boson Hamiltonians and baths

#pragma once

#include <cstddef>
#include <vector>

#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/bath_tabulated.hpp"
#include "taco/bath_models.hpp"
#include "taco/tcl2.hpp"

namespace taco::spin_boson {

struct Params {
    double Delta{1.0};     // tunneling strength
    double epsilon{0.0};   // bias
    double alpha{0.05};    // coupling strength
    double omega_c{5.0};   // cutoff frequency
    double beta{1.0};      // inverse temperature (1/kT)
    std::size_t rank{1};   // number of bath channels
    Eigen::MatrixXcd coupling_operator = ops::sigma_z(); // system-bath coupling in lab basis
};

inline Eigen::MatrixXcd build_hamiltonian(const Params& p) {
    return -0.5 * p.Delta * ops::sigma_x() - 0.5 * p.epsilon * ops::sigma_z();
}

inline std::vector<tcl::JumpOperator> build_jump_operators(const Params& p) {
    tcl::JumpOperator op;
    op.label = "coupling";
    op.matrix = p.coupling_operator;
    return {op};
}

inline sys::System build_system(const Params& p, double freq_tol = 1e-9) {
    sys::System system;
    system.build(build_hamiltonian(p), {p.coupling_operator}, freq_tol);
    return system;
}

inline bath::TabulatedCorrelation build_bath(const Params& p,
                                             std::size_t N,
                                             double dt,
                                             const bath::PowerLawSpectrum& spectrum) {
    auto J = [&](double w) { return spectrum.J(w); };
    return bath::TabulatedCorrelation::from_spectral(p.rank, N, dt, J, p.beta);
}

struct Model {
    Params params;
    bath::TabulatedCorrelation bath;
    tcl::TCL2Generator generator;

    Model(const Params& p,
          bath::TabulatedCorrelation corr,
          const tcl::GeneratorOptions& opts = {})
        : params(p)
        , bath(std::move(corr))
        , generator(build_hamiltonian(params), build_jump_operators(params), bath, opts) {}

    Model(const Params& p,
          std::size_t N,
          double dt,
          const bath::PowerLawSpectrum& spectrum,
          const tcl::GeneratorOptions& opts = {})
        : Model(p, build_bath(p, N, dt, spectrum), opts) {}

    Model(const Params& p,
          std::size_t N,
          double dt,
          const tcl::GeneratorOptions& opts = {})
        : Model(p, build_bath(p, N, dt, bath::PowerLawSpectrum::ohmic(p.alpha, p.omega_c)), opts) {}
};

inline Model build_model(const Params& p,
                         std::size_t N,
                         double dt,
                         const tcl::GeneratorOptions& opts = {}) {
    return Model(p, N, dt, opts);
}


} // namespace taco::spin_boson
