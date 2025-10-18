#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <complex>
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

static inline double rel_err(std::complex<double> a, std::complex<double> b) {
    const double den = std::max(1.0, std::abs(b));
    return std::abs(a - b) / den;
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
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--dt=",0)==0) dt = std::stod(arg.substr(5));
        else if (arg.rfind("--N=",0)==0) N = static_cast<std::size_t>(std::stoull(arg.substr(4)));
    }
    std::vector<double> t; std::vector<std::complex<double>> C;
    auto J = [&](double w){ return (w>0.0) ? (w*std::exp(-w/omega_c)) : 0.0; };
    bcf::bcf_fft_fun(N, dt, J, beta, t, C);

    // Gamma series per bucket
    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;
    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(C, dt, omegas);

    auto bench = [&](int threads){
#ifdef _OPENMP
        if (threads>0) omp_set_num_threads(threads);
#endif
        auto t0 = std::chrono::high_resolution_clock::now();
        auto k_fft = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Convolution);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto k_dir = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Direct);
        auto t2 = std::chrono::high_resolution_clock::now();
        double t_fft = std::chrono::duration<double>(t1 - t0).count();
        double t_dir = std::chrono::duration<double>(t2 - t1).count();
        double eF=0,eC=0,eR=0; std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
        for (std::size_t i=0;i<nf;++i) for (std::size_t j=0;j<nf;++j) for (std::size_t k=0;k<nf;++k) {
            auto tidx = Nt-1;
            eF = std::max(eF, rel_err(k_fft.F[i][j][k](static_cast<Eigen::Index>(tidx)), k_dir.F[i][j][k](static_cast<Eigen::Index>(tidx))));
            eC = std::max(eC, rel_err(k_fft.C[i][j][k](static_cast<Eigen::Index>(tidx)), k_dir.C[i][j][k](static_cast<Eigen::Index>(tidx))));
            eR = std::max(eR, rel_err(k_fft.R[i][j][k](static_cast<Eigen::Index>(tidx)), k_dir.R[i][j][k](static_cast<Eigen::Index>(tidx))));
        }
        std::cout << "threads=" << threads << ", t_fft=" << t_fft << "s, t_dir=" << t_dir
                  << "s, errF=" << eF << ", errC=" << eC << ", errR=" << eR << "\n";
    };

    std::vector<int> thread_cases;
#ifdef _OPENMP
    thread_cases = {1, std::max(1, omp_get_max_threads()/2), omp_get_max_threads()};
#else
    thread_cases = {1};
#endif
    for (int th : thread_cases) bench(th);

    return 0;
}

