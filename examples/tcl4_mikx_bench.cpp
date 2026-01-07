#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"
#ifdef TACO_HAS_CUDA
#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"
#endif

int main(int argc, char** argv) {
    using namespace taco;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);

    double dt = 0.000625;
    double beta = 0.5;
    double omega_c = 10.0;
    std::size_t Nt = 4096;
    int dim = 2;
    bool run_serial = false;
    bool run_omp = false;
    bool run_cuda = false;
    int threads = 0;
    int gpu_id = 0;
    std::size_t tidx = std::numeric_limits<std::size_t>::max();

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--dt=", 0) == 0) dt = std::stod(arg.substr(5));
        else if (arg.rfind("--beta=", 0) == 0) beta = std::stod(arg.substr(7));
        else if (arg.rfind("--omega_c=", 0) == 0) omega_c = std::stod(arg.substr(10));
        else if (arg.rfind("--N=", 0) == 0) Nt = static_cast<std::size_t>(std::stoull(arg.substr(4)));
        else if (arg.rfind("--dim=", 0) == 0) dim = std::stoi(arg.substr(6));
        else if (arg == "--serial") run_serial = true;
        else if (arg == "--omp") run_omp = true;
        else if (arg == "--cuda") run_cuda = true;
        else if (arg.rfind("--threads=", 0) == 0) threads = std::stoi(arg.substr(10));
        else if (arg.rfind("--gpu_id=", 0) == 0) gpu_id = std::stoi(arg.substr(9));
        else if (arg.rfind("--tidx=", 0) == 0) tidx = static_cast<std::size_t>(std::stoull(arg.substr(7)));
    }

    if (!run_serial && !run_omp && !run_cuda) {
#ifdef _OPENMP
        run_omp = true;
#else
        run_serial = true;
#endif
    }

    if (dim <= 0) {
        std::cerr << "dim must be > 0\n";
        return 1;
    }

    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) H(i, i) = static_cast<double>(i);

    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(dim, dim);
    if (dim == 1) {
        A(0, 0) = 1.0;
    } else {
        for (int i = 0; i + 1 < dim; ++i) {
            A(i, i + 1) = 1.0;
            A(i + 1, i) = 1.0;
        }
    }

    sys::System system;
    system.build(H, {A}, 1e-9);

    std::vector<double> tgrid;
    std::vector<std::complex<double>> Ccorr;
    auto J = [&](double w) { return (w > 0.0) ? (w * std::exp(-w / omega_c)) : 0.0; };
    bcf::bcf_fft_fun(Nt, dt, J, beta, tgrid, Ccorr);

    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;

    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(Ccorr, dt, omegas);
    const std::size_t Nt_series = static_cast<std::size_t>(gamma_series.rows());
    if (tidx == std::numeric_limits<std::size_t>::max()) tidx = (Nt_series > 0 ? Nt_series - 1 : 0);

    Exec exec;
#ifdef _OPENMP
    exec.backend = Backend::Omp;
    exec.threads = threads;
#else
    exec.backend = Backend::Serial;
#endif

    auto t0 = std::chrono::high_resolution_clock::now();
    auto kernels = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2,
                                                tcl4::FCRMethod::Convolution, exec);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_kernels = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "kernels t=" << t_kernels << "s, nf=" << nf << ", Nt=" << Nt_series << "\n";

    tcl4::Tcl4Map map = tcl4::build_map(system, tgrid);
    std::cout << "mikx tidx=" << tidx << " (t=" << (tidx < tgrid.size() ? tgrid[tidx] : tidx * dt) << ")\n";

#ifdef _OPENMP
    std::vector<int> thread_cases;
    if (threads > 0) {
        thread_cases = {threads};
    } else {
        thread_cases = {1, std::max(1, omp_get_max_threads() / 2), omp_get_max_threads()};
    }
#else
    std::vector<int> thread_cases = {1};
#endif

    if (run_serial) {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto mikx = tcl4::build_mikx_serial(map, kernels, tidx);
        auto t3 = std::chrono::high_resolution_clock::now();
        double t_mikx = std::chrono::duration<double>(t3 - t2).count();
        std::cout << "mikx serial t=" << t_mikx << "s (N=" << mikx.N << ")\n";
    }

#ifdef _OPENMP
    if (run_omp) {
        for (int th : thread_cases) {
            if (th > 0 && !omp_in_parallel()) omp_set_num_threads(th);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto mikx = tcl4::build_mikx_omp(map, kernels, tidx);
            auto t3 = std::chrono::high_resolution_clock::now();
            double t_mikx = std::chrono::duration<double>(t3 - t2).count();
            std::cout << "mikx omp threads=" << th << ", t=" << t_mikx << "s (N=" << mikx.N << ")\n";
        }
    }
#endif

    if (run_cuda) {
#ifdef TACO_HAS_CUDA
        Exec exec_cuda;
        exec_cuda.backend = Backend::Cuda;
        exec_cuda.gpu_id = gpu_id;
        auto t2 = std::chrono::high_resolution_clock::now();
        auto mikx = tcl4::build_mikx_cuda(map, kernels, tidx, exec_cuda);
        auto t3 = std::chrono::high_resolution_clock::now();
        double t_mikx = std::chrono::duration<double>(t3 - t2).count();
        std::cout << "mikx cuda gpu_id=" << gpu_id << ", t=" << t_mikx << "s (N=" << mikx.N << ")\n";
#else
        std::cout << "mikx cuda requested but TACO_HAS_CUDA is not enabled in this build\n";
#endif
    }

    return 0;
}
