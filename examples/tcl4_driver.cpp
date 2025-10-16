#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

#include <Eigen/Dense>

#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"
#include "taco/tcl4_assemble.hpp"

int main(int argc, char** argv) {
    using namespace taco;
    using Matrix = Eigen::MatrixXcd;

    // --- Simple settings (can extend to CLI) ---
    struct Settings {
        double dt{0.000625};           // time step
        int nmax{2};                    // as in MATLAB driver
        int ns{2048};                   // sampling index scale
        double beta{0.5};               // inverse temperature
        double omega_c{10.0};           // cutoff
        double alpha{0.05};             // coupling strength for Ohmic J
        tcl4::FCRMethod method{tcl4::FCRMethod::Convolution};
    } S;

    // Basic arg parsing: --key=value
    auto parse_double = [](const std::string& s) { return std::stod(s); };
    auto parse_int    = [](const std::string& s) { return std::stoi(s); };
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--", 0) != 0) continue;
        const auto eq = arg.find('=');
        std::string key = arg.substr(2, eq == std::string::npos ? std::string::npos : eq - 2);
        std::string val = (eq == std::string::npos) ? std::string{} : arg.substr(eq + 1);
        try {
            if (key == "dt") S.dt = parse_double(val);
            else if (key == "nmax") S.nmax = parse_int(val);
            else if (key == "ns") S.ns = parse_int(val);
            else if (key == "beta") S.beta = parse_double(val);
            else if (key == "omega_c") S.omega_c = parse_double(val);
            else if (key == "alpha") S.alpha = parse_double(val);
            else if (key == "method") {
                if (val == "direct") S.method = tcl4::FCRMethod::Direct;
                else S.method = tcl4::FCRMethod::Convolution;
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error parsing " << arg << ": " << ex.what() << "\n";
            return 1;
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "tcl4_driver: dt=" << S.dt << ", nmax=" << S.nmax
              << ", ns=" << S.ns << ", beta=" << S.beta
              << ", omega_c=" << S.omega_c << ", alpha=" << S.alpha
              << ", method=" << (S.method == tcl4::FCRMethod::Convolution ? "convolution" : "direct")
              << "\n";

    // --- System: 2-level, H = sigma_x/2, coupling A = sigma_z/2 ---
    Matrix H = 0.5 * ops::sigma_x();
    Matrix A_lab = 0.5 * ops::sigma_z();
    sys::System system;
    system.build(H, {A_lab}, 1e-9);

    // Coupling operator in eigen basis for assembly
    Matrix A_eig = system.eig.to_eigen(A_lab);

    // --- Build correlation C(t) from Ohmic J(w) via FFT ---
    // Target horizon ~ 2*nmax*ns*dt as in MATLAB driver
    const double T_target = 2.0 * static_cast<double>(S.nmax) * static_cast<double>(S.ns) * S.dt;
    // Choose N so that (N*dt) >= T_target with a modest safety factor
    std::size_t N = static_cast<std::size_t>(std::ceil(T_target / S.dt)) + 1024;
    std::vector<double> tgrid; tgrid.reserve(N+1);
    std::vector<std::complex<double>> Ccorr; Ccorr.reserve(N+1);
    auto J = [&](double w) -> double {
        if (!(w > 0.0)) return 0.0;
        return 2.0 * S.alpha * w * std::exp(-w / S.omega_c);
    };
    bcf::bcf_fft_fun(N, S.dt, J, S.beta, tgrid, Ccorr);

    // --- Build Γ(ω,t) for unique buckets via prefix trapezoid ---
    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;
    // gamma_series: rows = time (Ccorr.size()), cols = nf
    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(Ccorr, S.dt, omegas);

    // --- Build TCL4 map and triple-kernel series ---
    tcl4::Tcl4Map map = tcl4::build_map(system, tgrid);
    auto kernels = tcl4::compute_triple_kernels(system, gamma_series, S.dt, S.nmax, S.method);

    // --- Select time index: tidx ≈ 2*nmax*ns (0-based) ---
    std::size_t tidx = static_cast<std::size_t>(2 * S.nmax * S.ns);
    if (tidx >= static_cast<std::size_t>(gamma_series.rows())) tidx = static_cast<std::size_t>(gamma_series.rows()) - 1;
    const double t_sel = static_cast<double>(tidx) * S.dt;
    std::cout << "Selected time index tidx=" << tidx << " (t=" << t_sel << ")\n";

    // --- Build M/I/K/X tensors and assemble Liouvillian correction ---
    auto mikx = tcl4::build_mikx(map, kernels, tidx);
    Eigen::MatrixXcd GW = tcl4::assemble_liouvillian(mikx, std::vector<Matrix>{A_eig});

    // --- Report diagnostics ---
    const double fro = ops::fro_norm(GW);
    const double herm_err = ops::fro_norm(GW - GW.adjoint());
    std::cout << "GW size: " << GW.rows() << "x" << GW.cols()
              << ", ||GW||_F=" << fro
              << ", ||GW-GW^H||_F=" << herm_err << "\n";

    // Optionally, rebuild and show Γ(NxN) at this time
    auto Gmat = tcl4::build_gamma_matrix_at(map, gamma_series, tidx);
    std::cout << "Gamma(tidx) in eigen basis:\n" << Gmat << "\n";

    return 0;
}

