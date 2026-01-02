#include "taco/backend/cuda/tcl4_kernels_cuda.hpp"

#include "taco/backend/cuda/tcl4_fcr_kernels_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace taco::tcl4 {

namespace {

inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

inline std::size_t next_pow2(std::size_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if constexpr (sizeof(std::size_t) >= 8) n |= n >> 32;
    return ++n;
}

} // namespace

TripleKernelSeries compute_triple_kernels_cuda(const sys::System& system,
                                               const Eigen::MatrixXcd& gamma_series,
                                               double dt,
                                               int nmax,
                                               FCRMethod method,
                                               const Exec& exec)
{
    (void)nmax;
    if (method != FCRMethod::Convolution) {
        throw std::invalid_argument("compute_triple_kernels_cuda: only FCRMethod::Convolution is supported");
    }
    if (!(dt > 0.0)) {
        throw std::invalid_argument("compute_triple_kernels_cuda: dt must be > 0");
    }

    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    const std::size_t nf = static_cast<std::size_t>(gamma_series.cols());
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("compute_triple_kernels_cuda: gamma_series column count does not match frequency buckets");
    }

    TripleKernelSeries result;
    result.F.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.C.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.R.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));

    if (Nt == 0 || nf == 0) return result;

    // ---- host-side metadata (omegas + mirror indices) ----
    std::vector<double> h_omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) h_omegas[b] = system.fidx.buckets[b].omega;

    std::vector<int> h_mirror(nf, -1);
    const double tol = std::max(1e-12, system.fidx.tol);
    for (std::size_t j = 0; j < nf; ++j) {
        const double w = system.fidx.buckets[j].omega;
        if (std::abs(w) <= tol) { h_mirror[j] = static_cast<int>(j); continue; }
        const double target = -w;
        double best = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (std::size_t jp = 0; jp < nf; ++jp) {
            const double dw = std::abs(system.fidx.buckets[jp].omega - target);
            if (dw < best) { best = dw; best_idx = static_cast<int>(jp); }
        }
        h_mirror[j] = (best_idx >= 0 ? best_idx : static_cast<int>(j));
    }

    // Match CPU padding rule: Nfft = next_pow2(max(2*Nt-1, pad_factor*Nt))
    std::size_t L = 2 * Nt - 1;
    std::size_t target = L;
    const std::size_t pad_factor = get_fcr_fft_pad_factor();
    if (pad_factor > 0) {
        const std::size_t pf = pad_factor * Nt;
        if (pf > target) target = pf;
    }
    std::size_t Nfft = next_pow2(target);
    if (Nfft < 2) Nfft = 2;

    // Choose a fixed plan batch to avoid per-chunk plan rebuilds.
    constexpr std::size_t kDefaultBatch = 64;
    const std::size_t Bplan = std::min(nf, kDefaultBatch);

    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> must match cuDoubleComplex storage (2 doubles)");

    // ---- device allocations ----
    cuda_check(cudaSetDevice(exec.gpu_id), "cudaSetDevice");

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    cuDoubleComplex* d_gamma = nullptr;
    double* d_omegas = nullptr;
    int* d_mirror = nullptr;
    cuDoubleComplex* d_F = nullptr;
    cuDoubleComplex* d_C = nullptr;
    cuDoubleComplex* d_R = nullptr;

    std::complex<double>* h_F = nullptr;
    std::complex<double>* h_C = nullptr;
    std::complex<double>* h_R = nullptr;
    std::vector<std::complex<double>> h_F_vec, h_C_vec, h_R_vec;
    const std::size_t out_elems = Nt * Bplan;
    const std::size_t out_bytes = out_elems * sizeof(cuDoubleComplex);

    cuda_fcr::FcrWorkspace ws;

    try {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gamma), Nt * nf * sizeof(cuDoubleComplex)), "cudaMalloc(d_gamma)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_omegas), nf * sizeof(double)), "cudaMalloc(d_omegas)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mirror), nf * sizeof(int)), "cudaMalloc(d_mirror)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_F), out_bytes), "cudaMalloc(d_F)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_C), out_bytes), "cudaMalloc(d_C)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), out_bytes), "cudaMalloc(d_R)");

        cuda_check(cudaMemcpy(d_gamma,
                              gamma_series.data(),
                              Nt * nf * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(gamma)");
        cuda_check(cudaMemcpy(d_omegas, h_omegas.data(), nf * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(omegas)");
        cuda_check(cudaMemcpy(d_mirror, h_mirror.data(), nf * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(mirror)");

        if (exec.pinned) {
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_F), out_bytes), "cudaMallocHost(h_F)");
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_C), out_bytes), "cudaMallocHost(h_C)");
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_R), out_bytes), "cudaMallocHost(h_R)");
        } else {
            h_F_vec.resize(out_elems);
            h_C_vec.resize(out_elems);
            h_R_vec.resize(out_elems);
            h_F = h_F_vec.data();
            h_C = h_C_vec.data();
            h_R = h_R_vec.data();
        }

        cuda_fcr::FcrDeviceInputs inputs;
        inputs.gamma  = d_gamma;
        inputs.omegas = d_omegas;
        inputs.mirror = d_mirror;
        inputs.Nt = Nt;
        inputs.nf = nf;
        inputs.dt = dt;

        const Eigen::Index Nt_e = static_cast<Eigen::Index>(Nt);

        for (std::size_t i = 0; i < nf; ++i) {
            for (std::size_t j = 0; j < nf; ++j) {
                // Full chunks
                std::size_t k0 = 0;
                for (; k0 + Bplan <= nf; k0 += Bplan) {
                    cuda_fcr::FcrBatch b;
                    b.batch = Bplan;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0);
                    b.F = d_F;
                    b.C = d_C;
                    b.R = d_R;

                    cuda_fcr::compute_fcr_convolution_batched(inputs, b, ws, stream);

                    cuda_check(cudaMemcpyAsync(h_F, d_F, out_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(F)");
                    cuda_check(cudaMemcpyAsync(h_C, d_C, out_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(C)");
                    cuda_check(cudaMemcpyAsync(h_R, d_R, out_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(R)");
                    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

                    for (std::size_t lane = 0; lane < Bplan; ++lane) {
                        const std::size_t k = k0 + lane;
                        result.F[i][j][k] = Eigen::Map<const Eigen::VectorXcd>(h_F + lane * Nt, Nt_e);
                        result.C[i][j][k] = Eigen::Map<const Eigen::VectorXcd>(h_C + lane * Nt, Nt_e);
                        result.R[i][j][k] = Eigen::Map<const Eigen::VectorXcd>(h_R + lane * Nt, Nt_e);
                    }
                }

                // Tail: run one overlapped chunk at k0_last = nf - Bplan and only store the last rem lanes.
                if (k0 < nf) {
                    const std::size_t rem = nf - k0; // rem < Bplan
                    const std::size_t k0_last = nf - Bplan;
                    const std::size_t lane0 = Bplan - rem;

                    cuda_fcr::FcrBatch b;
                    b.batch = Bplan;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0_last);
                    b.F = d_F;
                    b.C = d_C;
                    b.R = d_R;

                    cuda_fcr::compute_fcr_convolution_batched(inputs, b, ws, stream);

                    cuda_check(cudaMemcpyAsync(h_F, d_F, out_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(F tail)");
                    cuda_check(cudaMemcpyAsync(h_C, d_C, out_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(C tail)");
                    cuda_check(cudaMemcpyAsync(h_R, d_R, out_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(R tail)");
                    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize tail");

                    for (std::size_t lane = lane0; lane < Bplan; ++lane) {
                        const std::size_t k = k0_last + lane;
                        result.F[i][j][k] = Eigen::Map<const Eigen::VectorXcd>(h_F + lane * Nt, Nt_e);
                        result.C[i][j][k] = Eigen::Map<const Eigen::VectorXcd>(h_C + lane * Nt, Nt_e);
                        result.R[i][j][k] = Eigen::Map<const Eigen::VectorXcd>(h_R + lane * Nt, Nt_e);
                    }
                }
            }
        }

        // Clean up GPU workspace created inside batched kernels.
        if (ws.plan) cufftDestroy(ws.plan);
        if (ws.A) cudaFree(ws.A);
        if (ws.B) cudaFree(ws.B);
        if (ws.B_conj) cudaFree(ws.B_conj);
        if (ws.scan_tmp) cudaFree(ws.scan_tmp);

        if (exec.pinned) {
            cudaFreeHost(h_F);
            cudaFreeHost(h_C);
            cudaFreeHost(h_R);
        }

        cudaFree(d_R);
        cudaFree(d_C);
        cudaFree(d_F);
        cudaFree(d_mirror);
        cudaFree(d_omegas);
        cudaFree(d_gamma);
        cudaStreamDestroy(stream);
    } catch (...) {
        if (ws.plan) cufftDestroy(ws.plan);
        if (ws.A) cudaFree(ws.A);
        if (ws.B) cudaFree(ws.B);
        if (ws.B_conj) cudaFree(ws.B_conj);
        if (ws.scan_tmp) cudaFree(ws.scan_tmp);

        if (exec.pinned) {
            if (h_F) cudaFreeHost(h_F);
            if (h_C) cudaFreeHost(h_C);
            if (h_R) cudaFreeHost(h_R);
        }
        if (d_R) cudaFree(d_R);
        if (d_C) cudaFree(d_C);
        if (d_F) cudaFree(d_F);
        if (d_mirror) cudaFree(d_mirror);
        if (d_omegas) cudaFree(d_omegas);
        if (d_gamma) cudaFree(d_gamma);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }

    return result;
}

} // namespace taco::tcl4
