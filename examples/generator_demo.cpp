#include <iostream>
#include <iomanip>
#include <limits>

#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/generator.hpp"

int main() {
    using namespace taco;
    using Matrix = Eigen::MatrixXcd;

    // Simple two-level system: H = (w0/2) * sigma_z
    const double w0 = 1.0;
    Matrix H = 0.5 * w0 * ops::sigma_z();

    // Single jump operator: sigma_minus in lab basis
    std::vector<Matrix> jump_ops{ ops::sigma_minus() };

    // Build eigensystem and spectral decomposition
    sys::System system;
    system.build(H, jump_ops, 1e-9);

    const std::size_t channels = system.A_eig_parts.size();
    const std::size_t buckets  = system.fidx.buckets.size();

    std::cout << "Channels: " << channels << ", Buckets: " << buckets << "\n";
    for (std::size_t i = 0; i < buckets; ++i) {
        std::cout << "  bucket " << i << ": omega = " << system.fidx.buckets[i].omega << "\n";
    }

    // Populate spectral kernels: only the downward transition (omega = -w0)
    tcl2::SpectralKernels kernels;
    kernels.buckets.resize(buckets);
    for (std::size_t b = 0; b < buckets; ++b) {
        kernels.buckets[b].omega = system.fidx.buckets[b].omega;
        kernels.buckets[b].Gamma = Matrix::Zero(channels, channels);
    }

    // Find the bucket closest to -w0 and assign a sample kernel value
    const double target = -w0;
    std::size_t idx = buckets;
    double best_delta = std::numeric_limits<double>::max();
    for (std::size_t b = 0; b < buckets; ++b) {
        double delta = std::abs(system.fidx.buckets[b].omega - target);
        if (delta < best_delta) {
            best_delta = delta;
            idx = b;
        }
    }
    if (idx == buckets) {
        std::cerr << "Failed to locate bucket for omega = -w0\n";
        return 1;
    }

    // Choose Re(Gamma) > 0 for a damping channel; imaginary part encodes Lamb shift
    kernels.buckets[idx].Gamma(0, 0) = std::complex<double>(0.05, -0.02);

    // Build TCL2 components (unitary + dissipator + Lamb shift)
    const double cutoff = 1e-12;
    const auto components = tcl2::build_tcl2_components(system, kernels, cutoff);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nLamb shift Hamiltonian (eigen basis):\n" << components.H_lamb_shift << "\n";

    std::cout << "Frobenius norms:\n";
    std::cout << "  ||L_unitary||_F  = " << ops::fro_norm(components.L_unitary) << "\n";
    std::cout << "  ||L_dissipator||_F = " << ops::fro_norm(components.L_dissipator) << "\n";
    std::cout << "  ||L_total||_F      = " << ops::fro_norm(components.total()) << "\n";

    // Apply generator to an excited-state density matrix
    Matrix rho_excited = ops::rho_qubit_1();
    Eigen::VectorXcd rho_vec = ops::vec(rho_excited);
    Eigen::VectorXcd drho_vec = components.total() * rho_vec;
    Matrix drho = ops::unvec(drho_vec, static_cast<std::size_t>(rho_excited.rows()));

    std::cout << "\nGenerator applied to |1><1|:\n" << drho << "\n";

    return 0;
}
