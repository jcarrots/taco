#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/tcl4.hpp"

static inline double rel_err(std::complex<double> a, std::complex<double> b) {
    const double den = std::max(1.0, std::abs(b));
    return std::abs(a - b) / den;
}

struct ErrTriple {
    double F{0.0};
    double C{0.0};
    double R{0.0};
};

static ErrTriple max_rel_err(const taco::tcl4::TripleKernelSeries& a,
                             const taco::tcl4::TripleKernelSeries& b,
                             std::size_t tidx)
{
    ErrTriple out;
    const std::size_t nf = a.F.size();
    if (nf == 0) return out;
    const Eigen::Index t = static_cast<Eigen::Index>(tidx);
    for (std::size_t i = 0; i < nf; ++i) {
        for (std::size_t j = 0; j < nf; ++j) {
            for (std::size_t k = 0; k < nf; ++k) {
                out.F = std::max(out.F, rel_err(a.F[i][j][k](t), b.F[i][j][k](t)));
                out.C = std::max(out.C, rel_err(a.C[i][j][k](t), b.C[i][j][k](t)));
                out.R = std::max(out.R, rel_err(a.R[i][j][k](t), b.R[i][j][k](t)));
            }
        }
    }
    return out;
}

int main(int argc, char** argv) {
    using namespace taco;
    std::cout.setf(std::ios::fixed); std::cout.precision(9);

    // Simple 2-level model
    Eigen::MatrixXcd H = 0.5 * ops::sigma_x();
    Eigen::MatrixXcd A = 0.5 * ops::sigma_z();
    sys::System system; system.build(H, {A}, 1e-9);

    // Build C(t) via FFT
    double beta = 0.5, omega_c = 10.0, dt = 0.000625; std::size_t N = 4096;
    bool use_cuda = false;
    bool force_serial = false;
    bool force_omp = false;
    bool run_direct = true;
    int threads = 0;
    int gpu_id = 0;
    std::size_t tidx = std::numeric_limits<std::size_t>::max();
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--dt=",0)==0) dt = std::stod(arg.substr(5));
        else if (arg.rfind("--N=",0)==0) N = static_cast<std::size_t>(std::stoull(arg.substr(4)));
        else if (arg == "--cuda") use_cuda = true;
        else if (arg == "--serial") force_serial = true;
        else if (arg == "--omp") force_omp = true;
        else if (arg == "--no-direct") run_direct = false;
        else if (arg.rfind("--backend=",0)==0) {
            const std::string v = arg.substr(10);
            if (v == "cuda") use_cuda = true;
            else if (v == "serial") force_serial = true;
            else if (v == "omp") force_omp = true;
        } else if (arg.rfind("--threads=",0)==0) {
            threads = std::stoi(arg.substr(10));
        } else if (arg.rfind("--gpu_id=",0)==0) {
            gpu_id = std::stoi(arg.substr(9));
        } else if (arg.rfind("--tidx=",0)==0) {
            tidx = static_cast<std::size_t>(std::stoull(arg.substr(7)));
        }
    }
    std::vector<double> t; std::vector<std::complex<double>> C;
    auto J = [&](double w){ return (w>0.0) ? (w*std::exp(-w/omega_c)) : 0.0; };
    bcf::bcf_fft_fun(N, dt, J, beta, t, C);

    // Gamma series per bucket
    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;
    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(C, dt, omegas);

    std::vector<int> thread_cases;
#ifdef _OPENMP
    if (threads > 0) {
        thread_cases = {threads};
    } else {
        thread_cases = {1, std::max(1, omp_get_max_threads()/2), omp_get_max_threads()};
    }
#else
    thread_cases = {1};
#endif

    tcl4::TripleKernelSeries k_cpu_conv_last;
    bool have_cpu_conv = false;
    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    if (tidx == std::numeric_limits<std::size_t>::max()) tidx = (Nt > 0 ? Nt - 1 : 0);

    for (int th : thread_cases) {
        Exec exec;
#ifdef _OPENMP
        if (force_serial) exec.backend = Backend::Serial;
        else exec.backend = Backend::Omp;
#else
        exec.backend = Backend::Serial;
#endif
        if (force_omp) exec.backend = Backend::Omp;
        exec.threads = th;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto k_conv = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Convolution, exec);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_fft = std::chrono::duration<double>(t1 - t0).count();

        ErrTriple err_direct;
        double t_dir = 0.0;
        if (run_direct) {
            auto t2 = std::chrono::high_resolution_clock::now();
            auto k_dir = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Direct, exec);
            auto t3 = std::chrono::high_resolution_clock::now();
            t_dir = std::chrono::duration<double>(t3 - t2).count();
            err_direct = max_rel_err(k_conv, k_dir, tidx);
        }

        std::cout << "cpu threads=" << th
                  << ", t_fft=" << t_fft << "s";
        if (run_direct) {
            std::cout << ", t_dir=" << t_dir
                      << "s, errF=" << err_direct.F
                      << ", errC=" << err_direct.C
                      << ", errR=" << err_direct.R;
        }
        std::cout << "\n";

        k_cpu_conv_last = std::move(k_conv);
        have_cpu_conv = true;
    }

    if (use_cuda) {
#ifdef TACO_HAS_CUDA
        Exec exec;
        exec.backend = Backend::Cuda;
        exec.gpu_id = gpu_id;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto k_gpu = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Convolution, exec);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_gpu = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "gpu id=" << gpu_id << ", t_fft=" << t_gpu << "s";
        if (have_cpu_conv) {
            ErrTriple err_gpu = max_rel_err(k_cpu_conv_last, k_gpu, tidx);
            std::cout << ", errF=" << err_gpu.F
                      << ", errC=" << err_gpu.C
                      << ", errR=" << err_gpu.R;
        }
        std::cout << "\n";
#else
        std::cout << "gpu requested but TACO_HAS_CUDA is not enabled in this build\n";
#endif
    }

    return 0;
}
