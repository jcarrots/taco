#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"
#include "taco/tcl4_assemble.hpp"

// Minimal spin-boson TCL4 propagation demo: L_total = L2 + alpha^2 * DW
// Uses frozen-L explicit Euler per step (small dt recommended)

int main(int argc, char** argv) {
    using namespace taco;
    using Matrix = Eigen::MatrixXcd;

    struct Settings {
        double Delta{1.0};
        double epsilon{0.2};
        double alpha{0.05};
        double omega_c{5.0};
        double beta{5.0};
        // Correlation grid
        std::size_t Ncorr{4096};
        double dt_corr{0.001};
        // Propagation
        double t0{0.0};
        double tf{2.0};
        double dt{0.001};
        tcl4::FCRMethod method{tcl4::FCRMethod::Convolution};
    } S;

    auto parse_double = [](const std::string& s) { return std::stod(s); };
    auto parse_size   = [](const std::string& s) { return static_cast<std::size_t>(std::stoull(s)); };
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--", 0) != 0) continue;
        const auto eq = arg.find('=');
        std::string key = arg.substr(2, eq == std::string::npos ? std::string::npos : eq - 2);
        std::string val = (eq == std::string::npos) ? std::string{} : arg.substr(eq + 1);
        try {
            if (key == "delta") S.Delta = parse_double(val);
            else if (key == "epsilon") S.epsilon = parse_double(val);
            else if (key == "alpha") S.alpha = parse_double(val);
            else if (key == "omega_c") S.omega_c = parse_double(val);
            else if (key == "beta") S.beta = parse_double(val);
            else if (key == "ncorr") S.Ncorr = parse_size(val);
            else if (key == "dt_corr") S.dt_corr = parse_double(val);
            else if (key == "t0") S.t0 = parse_double(val);
            else if (key == "tf") S.tf = parse_double(val);
            else if (key == "dt") S.dt = parse_double(val);
            else if (key == "method") {
                if (val == "direct") S.method = tcl4::FCRMethod::Direct; else S.method = tcl4::FCRMethod::Convolution;
            } else if (key == "threads") {
#ifdef _OPENMP
                int th = static_cast<int>(parse_size(val));
                if (th > 0) omp_set_num_threads(th);
#else
                (void)val;
#endif
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error parsing " << arg << ": " << ex.what() << "\n";
            return 1;
        }
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Spin-boson TCL4 demo: delta=" << S.Delta
              << ", epsilon=" << S.epsilon
              << ", alpha=" << S.alpha
              << ", omega_c=" << S.omega_c
              << ", beta=" << S.beta
              << ", ncorr=" << S.Ncorr
              << ", dt_corr=" << S.dt_corr
              << ", t0=" << S.t0 << ", tf=" << S.tf << ", dt=" << S.dt
              << ", method=" << (S.method == tcl4::FCRMethod::Convolution ? "convolution" : "direct") << "\n";
#ifdef _OPENMP
    std::cout << "OpenMP: max_threads=" << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP: disabled at build time\n";
#endif

    // System in lab basis
    Matrix H_lab = -0.5 * S.Delta * ops::sigma_x() - 0.5 * S.epsilon * ops::sigma_z();
    Matrix A_lab = ops::sigma_z();

    // Build system and eigen quantities
    sys::System system; system.build(H_lab, {A_lab}, 1e-9);
    Matrix A_eig = system.A_eig.front();

    // Correlation C(t) via FFT from Ohmic J(w)
    std::vector<double> t; std::vector<std::complex<double>> C;
    auto J = [&](double w){ return (w > 0.0) ? (2.0 * S.alpha * w * std::exp(-w / S.omega_c)) : 0.0; };
    bcf::bcf_fft_fun(S.Ncorr, S.dt_corr, J, S.beta, t, C);

    // Γ series for buckets
    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;
    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(C, S.dt_corr, omegas);

    // Triple kernels (F/C/R) for all triples (ω1,ω2,ω3)
    auto kernels = tcl4::compute_triple_kernels(system, gamma_series, S.dt_corr, /*nmax*/2, S.method);

    // Prepare mapping and MIKX container
    tcl4::Tcl4Map map = tcl4::build_map(system, t);

    // Propagate with frozen-L Euler: rho_{k+1} = rho_k + dt * (L_total(t_k) * vec(rho_k))
    const std::size_t N = static_cast<std::size_t>(system.eig.dim);
    Matrix rho = ops::rho_qubit_1();
    const Matrix sigma_z = ops::sigma_z();

    const std::size_t steps = static_cast<std::size_t>(std::ceil((S.tf - S.t0) / S.dt));
    double tcur = S.t0;
    std::cout << "t,sz\n";
    for (std::size_t step = 0; step <= steps; ++step) {
        // Sample observable
        double sz = (rho * sigma_z).trace().real();
        std::cout << tcur << "," << sz << "\n";
        if (step == steps) break;

        // Choose nearest correlation-grid index
        std::size_t tidx = static_cast<std::size_t>(std::llround(tcur / S.dt_corr));
        if (tidx >= static_cast<std::size_t>(gamma_series.rows())) tidx = static_cast<std::size_t>(gamma_series.rows()) - 1;

        // TCL2 at time index: diagonal channels from gamma_series row
        const std::size_t buckets = nf;
        const std::size_t channels = system.A_eig_parts.size();
        tcl2::SpectralKernels K2; K2.buckets.resize(buckets);
        for (std::size_t b = 0; b < buckets; ++b) {
            K2.buckets[b].omega = system.fidx.buckets[b].omega;
            K2.buckets[b].Gamma = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(channels), static_cast<Eigen::Index>(channels));
            const auto val = gamma_series(static_cast<Eigen::Index>(tidx), static_cast<Eigen::Index>(b));
            for (std::size_t a = 0; a < channels; ++a) K2.buckets[b].Gamma(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(a)) = val;
        }
        auto comps2 = tcl2::build_tcl2_components(system, K2, /*cutoff*/0.0);
        Eigen::MatrixXcd L2 = comps2.total();

        // TCL4 correction at time index
        auto mikx = tcl4::build_mikx(map, kernels, tidx);
        Eigen::MatrixXcd GW = tcl4::assemble_liouvillian(mikx, std::vector<Matrix>{A_eig});
        Eigen::MatrixXcd Ltotal = L2 + (S.alpha * S.alpha) * GW;

        // Frozen-L Euler step on vec(rho)
        Eigen::VectorXcd v = ops::vec(rho);
        Eigen::VectorXcd dv = Ltotal * v;
        v += S.dt * dv;
        rho = ops::unvec(v, N);
        rho = 0.5 * (rho + rho.adjoint());
        const double tr = rho.trace().real();
        if (std::abs(tr) > 0.0) rho /= tr;

        tcur = std::min(S.tf, tcur + S.dt);
    }

    return 0;
}
