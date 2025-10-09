// generator.hpp — Helpers to assemble TCL2 unitary and dissipative superoperators

#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

#include "taco/system.hpp"

namespace taco::tcl2 {

struct SpectralKernels {
    struct Bucket {
        double omega{0.0};
        Eigen::MatrixXcd Gamma; // (#channels x #channels) complex kernel values
    };
    std::vector<Bucket> buckets;
};

struct TCL2Components {
    Eigen::MatrixXcd L_unitary;    // -i[H_eff,·] superoperator (N^2 x N^2)
    Eigen::MatrixXcd L_dissipator; // Redfield/TCL2 dissipator superoperator
    Eigen::MatrixXcd H_lamb_shift; // Hermitian Lamb-shift correction (N x N)

    Eigen::MatrixXcd total() const { return L_unitary + L_dissipator; }
};

// Build the Lamb-shift Hamiltonian from spectral kernels
Eigen::MatrixXcd build_lamb_shift(const sys::System& system,
                                  const SpectralKernels& kernels,
                                  double imag_cutoff = 0.0);

// Build the unitary superoperator from an effective Hamiltonian H_eff
Eigen::MatrixXcd build_unitary_superop(const sys::System& system,
                                       const Eigen::MatrixXcd& H_eff);

// Build the second-order dissipative (TCL2) superoperator. Optionally accumulate
// the Lamb shift into the provided pointer.
Eigen::MatrixXcd build_dissipator_superop(const sys::System& system,
                                          const SpectralKernels& kernels,
                                          double gamma_cutoff = 0.0,
                                          Eigen::MatrixXcd* lamb_shift_out = nullptr);

// Assemble full TCL2 components (unitary + dissipator + Lamb shift)
TCL2Components build_tcl2_components(const sys::System& system,
                                     const SpectralKernels& kernels,
                                     double cutoff = 0.0);

} // namespace taco::tcl2

