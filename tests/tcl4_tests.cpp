#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "taco/system.hpp"
#include "taco/ops.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/tcl4.hpp"
#ifdef TACO_HAS_CUDA
#include "taco/exec.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_mikx.hpp"
#include "taco/backend/cuda/tcl4_assemble_cuda.hpp"
#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"
#endif

static inline double rel_err(std::complex<double> a, std::complex<double> b) {
    const double den = std::max(1.0, std::abs(b));
    return std::abs(a - b) / den;
}

#ifdef TACO_HAS_CUDA
static double max_abs_diff(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return std::numeric_limits<double>::infinity();
    return (a - b).cwiseAbs().maxCoeff();
}

static bool run_cuda_assemble_smoke() {
    using taco::tcl4::Tcl4Map;
    using taco::tcl4::TripleKernelSeries;

    const int N = 2;
    const std::size_t nf = 4;
    const std::size_t Nt = 4;

    Tcl4Map map;
    map.N = N;
    map.nf = static_cast<int>(nf);
    map.pair_to_freq = Eigen::MatrixXi::Constant(N, N, -1);
    map.pair_to_freq(0, 0) = 0;
    map.pair_to_freq(0, 1) = 1;
    map.pair_to_freq(1, 0) = 2;
    map.pair_to_freq(1, 1) = 3;

    TripleKernelSeries kernels;
    kernels.F.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    kernels.C.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    kernels.R.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));

    auto make_val = [](double base, std::size_t a, std::size_t b, std::size_t c, std::size_t t) {
        const double re = base + 100.0 * a + 10.0 * b + static_cast<double>(c) + 0.001 * t;
        const double im = 0.1 * base + 0.01 * a - 0.001 * b + 0.0001 * c + 0.00001 * t;
        return std::complex<double>(re, im);
    };

    for (std::size_t a = 0; a < nf; ++a) {
        for (std::size_t b = 0; b < nf; ++b) {
            for (std::size_t c = 0; c < nf; ++c) {
                kernels.F[a][b][c] = Eigen::VectorXcd(static_cast<Eigen::Index>(Nt));
                kernels.C[a][b][c] = Eigen::VectorXcd(static_cast<Eigen::Index>(Nt));
                kernels.R[a][b][c] = Eigen::VectorXcd(static_cast<Eigen::Index>(Nt));
                for (std::size_t t = 0; t < Nt; ++t) {
                    kernels.F[a][b][c](static_cast<Eigen::Index>(t)) = make_val(1.0, a, b, c, t);
                    kernels.C[a][b][c](static_cast<Eigen::Index>(t)) = make_val(2.0, a, b, c, t);
                    kernels.R[a][b][c](static_cast<Eigen::Index>(t)) = make_val(3.0, a, b, c, t);
                }
            }
        }
    }

    std::vector<Eigen::MatrixXcd> ops;
    Eigen::MatrixXcd A0 = Eigen::MatrixXcd::Identity(N, N);
    Eigen::MatrixXcd A1 = Eigen::MatrixXcd::Zero(N, N);
    A1(0, 1) = 1.0;
    A1(1, 0) = 1.0;
    ops.push_back(A0);
    ops.push_back(A1);

    taco::Exec exec;
    exec.backend = taco::Backend::Cuda;
    exec.gpu_id = 0;
    const std::vector<std::size_t> tidx_list = {0, Nt / 2, Nt - 1};

    double max_err = 0.0;
    double max_rel_err = 0.0;
    double cpu_gw_total_ms = 0.0;
    double cuda_gw_total_ms = 0.0;
    double cpu_total_ms = 0.0;
    double cuda_total_ms = 0.0;

    for (std::size_t tidx : tidx_list) {
        const auto t_cpu_total_start = std::chrono::high_resolution_clock::now();
        const auto mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
        const auto t_cpu_gw_start = std::chrono::high_resolution_clock::now();
        const Eigen::MatrixXcd gw_cpu = taco::tcl4::assemble_liouvillian(mikx, ops);
        const auto t_cpu_gw_end = std::chrono::high_resolution_clock::now();
        const auto l4_cpu = taco::tcl4::gw_to_liouvillian(gw_cpu, static_cast<std::size_t>(N));
        const auto t_cpu_total_end = std::chrono::high_resolution_clock::now();
        (void)l4_cpu;

        const auto t_gw_start = std::chrono::high_resolution_clock::now();
        const Eigen::MatrixXcd gw_gpu = taco::tcl4::assemble_liouvillian_cuda(mikx, ops, exec);
        const auto t_gw_end = std::chrono::high_resolution_clock::now();

        const auto t_cuda_total_start = std::chrono::high_resolution_clock::now();
        const auto mikx_cuda = taco::tcl4::build_mikx_cuda(map, kernels, tidx, exec);
        const auto gw_cuda = taco::tcl4::assemble_liouvillian_cuda(mikx_cuda, ops, exec);
        const auto l4_cuda = taco::tcl4::gw_to_liouvillian(gw_cuda, static_cast<std::size_t>(N));
        const auto t_cuda_total_end = std::chrono::high_resolution_clock::now();
        (void)l4_cuda;

        const double err = max_abs_diff(gw_cpu, gw_gpu);
        const double ref = std::max(1.0, gw_cpu.cwiseAbs().maxCoeff());
        const double rel_err = err / ref;
        max_err = std::max(max_err, err);
        max_rel_err = std::max(max_rel_err, rel_err);

        const double cpu_gw_ms =
            std::chrono::duration<double, std::milli>(t_cpu_gw_end - t_cpu_gw_start).count();
        const double cuda_gw_ms =
            std::chrono::duration<double, std::milli>(t_gw_end - t_gw_start).count();
        const double cpu_total =
            std::chrono::duration<double, std::milli>(t_cpu_total_end - t_cpu_total_start).count();
        const double cuda_total =
            std::chrono::duration<double, std::milli>(t_cuda_total_end - t_cuda_total_start).count();

        cpu_gw_total_ms += cpu_gw_ms;
        cuda_gw_total_ms += cuda_gw_ms;
        cpu_total_ms += cpu_total;
        cuda_total_ms += cuda_total;

        std::cout << "tidx=" << tidx
                  << " cpu_gw_ms=" << cpu_gw_ms
                  << " cuda_gw_ms=" << cuda_gw_ms
                  << " cpu_total_ms=" << cpu_total
                  << " cuda_total_ms=" << cuda_total
                  << "\n";
    }

    const double count = static_cast<double>(tidx_list.size());
    std::cout << "GW assemble CUDA compare: max_abs=" << max_err
              << ", rel=" << max_rel_err << "\n";
    std::cout << "GW assemble CPU avg_ms=" << (cpu_gw_total_ms / count)
              << ", CUDA avg_ms=" << (cuda_gw_total_ms / count) << "\n";
    std::cout << "TCL4 end-to-end CPU avg_ms=" << (cpu_total_ms / count)
              << ", CUDA avg_ms=" << (cuda_total_ms / count) << "\n";

    const double tol = 1e-9;
    if (max_err > tol && max_rel_err > tol) {
        std::cerr << "FAIL: GW mismatch above tolerance\n";
        return false;
    }
    std::cout << "CUDA assemble test PASS\n";
    return true;
}
#endif

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

    std::cout << "TCL4 CPU test complete." << std::endl;

    bool ok = true;
#ifdef TACO_HAS_CUDA
    std::cout << "\n--- CUDA assemble check ---\n";
    ok = run_cuda_assemble_smoke() && ok;
#else
    std::cout << "CUDA not enabled; skipping CUDA assemble check." << std::endl;
#endif

    std::cout << "TCL4 tests complete." << std::endl;
    return ok ? 0 : 1;
}
