#include "taco/backend/cuda/tcl4_kernels_cuda.hpp"

#include <stdexcept>
#include<cuda_runtime.h>
#include<cuComplex.h>
#include<vector>
namespace taco::tcl4 {

TripleKernelSeries compute_triple_kernels_cuda(const sys::System& system,
                                               const Eigen::MatrixXcd& gamma_series,
                                               double dt,
                                               int nmax,
                                               FCRMethod method,
                                               const Exec& exec)
{   
    static void cuda_check(cudaError_t st, const char* what) {
    if (st != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
    }

    cuda_check(cudaSetDevice(exec.gpu_id),"cudaSetDevice")
    //allocate gpu memory for system, gammas and bath
    std::vector<double> h_omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) h_omegas[b] = system.fidx.buckets[b].omega;
    std::vector<int> h_mirror(nf);
    //precompute mirror
    std::vector<int> h_mirror(nf, -1);
    const double tol = std::max(1e-12, system.fidx.tol);
    for (std::size_t j = 0; j < nf; ++j) {
        const double w = system.fidx.buckets[j].omega;
        if (std::abs(w) <= tol) { h_mirror[j] = static_cast<int>(j); continue; }
        const double target = -w;
        double best = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (std::size_t jp = 0; jp < nf; ++jp) {
            double dw = std::abs(system.fidx.buckets[jp].omega - target);
            if (dw < best) { best = dw; best_idx = static_cast<int>(jp); }
        }
        h_mirror[j] = (best_idx >= 0 ? best_idx : static_cast<int>(j));
    }
    cuDoubleComplex* d_gamma = nullptr;
    double* d_omegas = nullptr;
    int* d_mirror = nullptr;
    cuda_check(cudaMalloc(&d_gamma, Nt * nf * sizeof(cuDoubleComplex)), "cudaMalloc(d_gamma)");
    cuda_check(cudaMalloc(&d_omegas, nf * sizeof(double)), "cudaMalloc(d_omegas)");
    cuda_check(cudaMalloc(&d_mirror, nf * sizeof(int)), "cudaMalloc(d_mirror)");
    //host to device
    cuda_check(cudaMemcpy(d_gamma, gamma_series.data(),
                        Nt * nf * sizeof(std::complex<double>),
                        cudaMemcpyHostToDevice),
            "cudaMemcpy(gamma)");

    cuda_check(cudaMemcpy(d_omegas, h_omegas.data(), nf * sizeof(double), cudaMemcpyHostToDevice),"cudaMemcpy(omegas)");

    cuda_check(cudaMemcpy(d_mirror, h_mirror.data(), nf * sizeof(int), cudaMemcpyHostToDevice),"cudaMemcpy(mirror)");



    for (std::ptrdiff_t idx = 0; idx < total; ++idx) {
        const std::size_t i = static_cast<std::size_t>(idx / nf_i);
        const std::size_t j = static_cast<std::size_t>(idx % nf_i);

        const auto g1col = gamma_series.col(static_cast<Eigen::Index>(i));
        const auto g2col = gamma_series.col(static_cast<Eigen::Index>(j));
        const int j_mirror = mirror_idx[j];
        const auto g2mcol =
            gamma_series.col(static_cast<Eigen::Index>(j_mirror >= 0 ? j_mirror : static_cast<int>(j)));

        const double wi = system.fidx.buckets[i].omega;
        const double wj = system.fidx.buckets[j].omega;
        for (std::size_t k = 0; k < nf; ++k) {
            const double omega = wi + wj + system.fidx.buckets[k].omega;
            Eigen::VectorXcd Ft = compute_F_series(g1col, g2mcol, omega, dt, method);
            Eigen::VectorXcd Ct = compute_C_series(g1col, g2col, omega, dt, method);
            Eigen::VectorXcd Rt = compute_R_series(g1col, g2col, omega, dt, method);
            result.F[i][j][k] = std::move(Ft);
            result.C[i][j][k] = std::move(Ct);
            result.R[i][j][k] = std::move(Rt);
        }
    }


}

} // namespace taco::tcl4

