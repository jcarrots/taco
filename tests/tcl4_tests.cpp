#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>

#include <Eigen/Dense>

#include "taco/system.hpp"
#include "taco/ops.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/tcl4.hpp"

static inline double rel_err(std::complex<double> a, std::complex<double> b) {
    const double den = std::max(1.0, std::abs(b));
    return std::abs(a - b) / den;
}

int main() {
    using namespace taco;
    std::cout.setf(std::ios::fixed); std::cout.precision(12);
    std::cout << "Starting TCL4 test:\n" << std::endl;
    std::cout << "--- Build a simple 2-level system (H = sx/2) with one coupling (sz/2) ---" << std::endl;
    // --- Build a simple 2-level system (H = sx/2) with one coupling (sz/2) ---
    Eigen::MatrixXcd H = 0.5 * ops::sigma_x();
    Eigen::MatrixXcd A = 0.5 * ops::sigma_z();
    sys::System system; system.build(H, {A}, 1e-9);
    const std::size_t nf = system.fidx.buckets.size();

    auto run_case = [&](std::size_t N, double dt)->std::tuple<double,double,double> {
        const double beta = 0.5;
        const double omega_c = 10.0;
        std::vector<double> t; std::vector<std::complex<double>> C;
        auto J = [&](double w){ return (w > 0.0) ? (w * std::exp(-w / omega_c)) : 0.0; };
        bcf::bcf_fft_fun(N, dt, J, beta, t, C);

        std::vector<double> omegas(nf);
        for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;
        Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(C, dt, omegas);

        auto kernels_fft = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Convolution);
        auto kernels_dir = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Direct);

        const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
        std::vector<std::size_t> tids = {0, Nt/4, Nt/2, std::min(Nt-1, (std::size_t)(3*Nt/4)), Nt-1};

        double max_rel_F = 0.0, max_rel_C = 0.0, max_rel_R = 0.0;
        for (std::size_t i = 0; i < nf; ++i) {
            for (std::size_t j = 0; j < nf; ++j) {
                for (std::size_t k = 0; k < nf; ++k) {
                    for (auto tidx : tids) {
                        auto Ff = kernels_fft.F[i][j][k](static_cast<Eigen::Index>(tidx));
                        auto Cf = kernels_fft.C[i][j][k](static_cast<Eigen::Index>(tidx));
                        auto Rf = kernels_fft.R[i][j][k](static_cast<Eigen::Index>(tidx));
                        auto Fd = kernels_dir.F[i][j][k](static_cast<Eigen::Index>(tidx));
                        auto Cd = kernels_dir.C[i][j][k](static_cast<Eigen::Index>(tidx));
                        auto Rd = kernels_dir.R[i][j][k](static_cast<Eigen::Index>(tidx));
                        max_rel_F = std::max(max_rel_F, rel_err(Ff, Fd));
                        max_rel_C = std::max(max_rel_C, rel_err(Cf, Cd));
                        max_rel_R = std::max(max_rel_R, rel_err(Rf, Rd));
                    }
                }
            }
        }
        return {max_rel_F, max_rel_C, max_rel_R};
    };

    // Sweep 1: different N at fixed dt
    const double dt_fixed = 0.000625;
    std::vector<std::size_t> Ns = {2048, 8192, 16384};
    for (auto N : Ns) {
        auto [eF,eC,eR] = run_case(N, dt_fixed);
        std::cout << "Case N=" << N << ", dt=" << dt_fixed
                  << ": errF=" << eF << ", errC=" << eC << ", errR=" << eR << "\n";
    }

    // Sweep 2: different (dt, T); derive N = floor(T/dt)
    std::vector<double> dts = {0.0001, 0.0005, 0.0020};
    std::vector<double> Ts  = {2.0, 4.0, 8.0};
    for (double dtv : dts) {
        for (double Tv : Ts) {
            std::size_t N = static_cast<std::size_t>(std::floor(Tv / dtv));
            auto [eF,eC,eR] = run_case(N, dtv);
            std::cout << "Case N=" << N << ", dt=" << dtv << ", T~=" << Tv
                      << ": errF=" << eF << ", errC=" << eC << ", errR=" << eR << "\n";
        }
    }

    std::cout << "TCL4 test complete." << std::endl;
    return 0;
}
