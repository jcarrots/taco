#include "taco/backend/cuda/tcl4_fused_cuda.hpp"

#include "taco/backend/cuda/tcl4_assemble_cuda.hpp"
#include "taco/backend/cuda/tcl4_fcr_kernels_cuda.hpp"
#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_kernels.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
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

__global__ void kernel_extract_time_slice(const cuDoubleComplex* __restrict__ F_batch,
                                          const cuDoubleComplex* __restrict__ C_batch,
                                          const cuDoubleComplex* __restrict__ R_batch,
                                          cuDoubleComplex* __restrict__ F_out,
                                          cuDoubleComplex* __restrict__ C_out,
                                          cuDoubleComplex* __restrict__ R_out,
                                          std::size_t Nt,
                                          std::size_t nf,
                                          int i,
                                          int j,
                                          int k0,
                                          int lane_start,
                                          int lane_end,
                                          std::size_t time_index)
{
    const int lane = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane < lane_start || lane >= lane_end) return;
    const std::size_t lane_u = static_cast<std::size_t>(lane);
    const std::size_t idx =
        (static_cast<std::size_t>(i) * nf + static_cast<std::size_t>(j)) * nf +
        static_cast<std::size_t>(k0) + lane_u;
    const std::size_t src = time_index + Nt * lane_u;
    F_out[idx] = F_batch[src];
    C_out[idx] = C_batch[src];
    R_out[idx] = R_batch[src];
}

__global__ void kernel_gw_to_liouvillian(const cuDoubleComplex* __restrict__ GW,
                                         cuDoubleComplex* __restrict__ L4,
                                         int N)
{
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = N2 * N2;
    if (idx >= total) return;

    const std::size_t col_L = idx / N2;
    const std::size_t row_L = idx - col_L * N2;

    const std::size_t n = row_L % N_u;
    const std::size_t m = row_L / N_u;
    const std::size_t i = col_L % N_u;
    const std::size_t j = col_L / N_u;

    const std::size_t row_G = n + i * N_u;
    const std::size_t col_G = m + j * N_u;
    L4[idx] = GW[row_G + col_G * N2];
}

} // namespace

Eigen::MatrixXcd build_TCL4_generator_cuda_fused(const sys::System& system,
                                                 const Eigen::MatrixXcd& gamma_series,
                                                 double dt,
                                                 std::size_t time_index,
                                                 FCRMethod method,
                                                 const Exec& exec)
{
    if (method != FCRMethod::Convolution) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: only FCRMethod::Convolution is supported");
    }
    if (time_index >= static_cast<std::size_t>(gamma_series.rows())) {
        throw std::out_of_range("build_TCL4_generator_cuda_fused: time_index out of range");
    }

    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    const std::size_t nf = static_cast<std::size_t>(gamma_series.cols());
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: gamma_series column count does not match frequency buckets");
    }
    if (nf == 0 || Nt == 0) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: empty gamma_series");
    }
    if (nf > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: nf too large for CUDA kernels");
    }

    Tcl4Map map = build_map(system, /*time_grid*/{});
    if (map.N <= 0) throw std::invalid_argument("build_TCL4_generator_cuda_fused: map.N must be > 0");
    if (map.nf <= 0) throw std::invalid_argument("build_TCL4_generator_cuda_fused: map.nf must be > 0");
    if (map.pair_to_freq.rows() != static_cast<Eigen::Index>(map.N) ||
        map.pair_to_freq.cols() != static_cast<Eigen::Index>(map.N)) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: map.pair_to_freq has wrong shape");
    }
    if (map.pair_to_freq.minCoeff() < 0) {
        throw std::runtime_error("build_TCL4_generator_cuda_fused: map.pair_to_freq contains -1 (missing frequency buckets)");
    }
    if (system.A_eig.empty()) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: coupling_ops must be non-empty");
    }

    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> must match cuDoubleComplex storage (2 doubles)");

    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t N2 = N * N;
    const std::size_t N6 = N * N * N * N * N * N;
    const std::size_t nf3 = nf * nf * nf;

    std::vector<double> h_omegas = map.omegas;
    std::vector<int> h_mirror = map.mirror_index;
    if (h_omegas.size() != nf || h_mirror.size() != nf) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: map frequency metadata has wrong size");
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

    constexpr std::size_t kDefaultBatch = 64;
    const std::size_t Bplan = std::min(nf, kDefaultBatch);

    const std::size_t num_ops = system.A_eig.size();
    if (num_ops > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: coupling_ops too large for CUDA kernel");
    }

    std::vector<std::complex<double>> h_ops(num_ops * N2);
    for (std::size_t op = 0; op < num_ops; ++op) {
        const auto& A = system.A_eig[op];
        if (A.rows() != static_cast<Eigen::Index>(N) || A.cols() != static_cast<Eigen::Index>(N)) {
            throw std::invalid_argument("build_TCL4_generator_cuda_fused: coupling operator has wrong shape");
        }
        std::copy(A.data(), A.data() + N2, h_ops.data() + op * N2);
    }

    cuda_check(cudaSetDevice(exec.gpu_id), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    cuDoubleComplex* d_gamma = nullptr;
    double* d_omegas = nullptr;
    int* d_mirror = nullptr;
    int* d_pair_to_freq = nullptr;
    cuDoubleComplex* d_ops = nullptr;

    cuDoubleComplex* d_F = nullptr;
    cuDoubleComplex* d_C = nullptr;
    cuDoubleComplex* d_R = nullptr;
    cuDoubleComplex* d_Ftmp = nullptr;
    cuDoubleComplex* d_Ctmp = nullptr;
    cuDoubleComplex* d_Rtmp = nullptr;

    cuDoubleComplex* dM = nullptr;
    cuDoubleComplex* dI = nullptr;
    cuDoubleComplex* dK = nullptr;
    cuDoubleComplex* dX = nullptr;
    cuDoubleComplex* dGW = nullptr;
    cuDoubleComplex* dL4 = nullptr;

    cuda_fcr::FcrWorkspace ws;

    try {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gamma), Nt * nf * sizeof(cuDoubleComplex)), "cudaMalloc(d_gamma)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_omegas), nf * sizeof(double)), "cudaMalloc(d_omegas)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mirror), nf * sizeof(int)), "cudaMalloc(d_mirror)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pair_to_freq), N2 * sizeof(int)), "cudaMalloc(d_pair_to_freq)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ops), num_ops * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(d_ops)");

        cuda_check(cudaMemcpyAsync(d_gamma,
                                   gamma_series.data(),
                                   Nt * nf * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync(gamma)");
        cuda_check(cudaMemcpyAsync(d_omegas, h_omegas.data(), nf * sizeof(double),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(omegas)");
        cuda_check(cudaMemcpyAsync(d_mirror, h_mirror.data(), nf * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(mirror)");
        cuda_check(cudaMemcpyAsync(d_pair_to_freq, map.pair_to_freq.data(), N2 * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(pair_to_freq)");
        cuda_check(cudaMemcpyAsync(d_ops, h_ops.data(), num_ops * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(coupling_ops)");

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_F), nf3 * sizeof(cuDoubleComplex)), "cudaMalloc(d_F)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_C), nf3 * sizeof(cuDoubleComplex)), "cudaMalloc(d_C)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), nf3 * sizeof(cuDoubleComplex)), "cudaMalloc(d_R)");

        const std::size_t out_elems = Nt * Bplan;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Ftmp), out_elems * sizeof(cuDoubleComplex)), "cudaMalloc(d_Ftmp)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Ctmp), out_elems * sizeof(cuDoubleComplex)), "cudaMalloc(d_Ctmp)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Rtmp), out_elems * sizeof(cuDoubleComplex)), "cudaMalloc(d_Rtmp)");

        cuda_fcr::FcrDeviceInputs inputs;
        inputs.gamma = d_gamma;
        inputs.omegas = d_omegas;
        inputs.mirror = d_mirror;
        inputs.Nt = Nt;
        inputs.nf = nf;
        inputs.dt = dt;

        constexpr int block_extract = 256;
        const dim3 grid_extract(static_cast<unsigned>((Bplan + block_extract - 1) / block_extract));

        for (std::size_t i = 0; i < nf; ++i) {
            for (std::size_t j = 0; j < nf; ++j) {
                std::size_t k0 = 0;
                for (; k0 + Bplan <= nf; k0 += Bplan) {
                    cuda_fcr::FcrBatch b;
                    b.batch = Bplan;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0);
                    b.F = d_Ftmp;
                    b.C = d_Ctmp;
                    b.R = d_Rtmp;

                    cuda_fcr::compute_fcr_convolution_batched(inputs, b, ws, stream);

                    kernel_extract_time_slice<<<grid_extract, block_extract, 0, stream>>>(
                        d_Ftmp, d_Ctmp, d_Rtmp,
                        d_F, d_C, d_R,
                        Nt, nf,
                        static_cast<int>(i), static_cast<int>(j), static_cast<int>(k0),
                        0, static_cast<int>(Bplan), time_index);
                    cuda_check(cudaGetLastError(), "kernel_extract_time_slice launch");
                }

                if (k0 < nf) {
                    const std::size_t rem = nf - k0;
                    const std::size_t k0_last = nf - Bplan;
                    const std::size_t lane0 = Bplan - rem;

                    cuda_fcr::FcrBatch b;
                    b.batch = Bplan;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0_last);
                    b.F = d_Ftmp;
                    b.C = d_Ctmp;
                    b.R = d_Rtmp;

                    cuda_fcr::compute_fcr_convolution_batched(inputs, b, ws, stream);

                    kernel_extract_time_slice<<<grid_extract, block_extract, 0, stream>>>(
                        d_Ftmp, d_Ctmp, d_Rtmp,
                        d_F, d_C, d_R,
                        Nt, nf,
                        static_cast<int>(i), static_cast<int>(j), static_cast<int>(k0_last),
                        static_cast<int>(lane0), static_cast<int>(Bplan), time_index);
                    cuda_check(cudaGetLastError(), "kernel_extract_time_slice tail launch");
                }
            }
        }

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dM), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dM)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dI), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dI)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dK), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dK)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dX), N6 * sizeof(cuDoubleComplex)), "cudaMalloc(dX)");

        cuda_mikx::MikxDeviceInputs in;
        in.F = d_F;
        in.C = d_C;
        in.R = d_R;
        in.pair_to_freq = d_pair_to_freq;
        in.N = map.N;
        in.nf = static_cast<int>(nf);

        cuda_mikx::MikxDeviceOutputs out;
        out.M = dM;
        out.I = dI;
        out.K = dK;
        out.X = dX;

        cuda_mikx::build_mikx_device(in, out, stream);

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dGW), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dGW)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dL4), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dL4)");

        assemble_liouvillian_cuda_device(dM, dI, dK, dX, d_ops,
                                         map.N, static_cast<int>(num_ops), dGW, stream);

        constexpr int block = 256;
        const std::size_t total = N2 * N2;
        const dim3 grid(static_cast<unsigned>((total + block - 1) / block));
        kernel_gw_to_liouvillian<<<grid, block, 0, stream>>>(dGW, dL4, map.N);
        cuda_check(cudaGetLastError(), "kernel_gw_to_liouvillian launch");

        Eigen::MatrixXcd L4(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        cuda_check(cudaMemcpyAsync(L4.data(), dL4, N2 * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(L4)");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        if (ws.plan) cufftDestroy(ws.plan);
        if (ws.A) cudaFree(ws.A);
        if (ws.B) cudaFree(ws.B);
        if (ws.B_conj) cudaFree(ws.B_conj);
        if (ws.scan_tmp) cudaFree(ws.scan_tmp);

        cudaFree(dL4);
        cudaFree(dGW);
        cudaFree(dX);
        cudaFree(dK);
        cudaFree(dI);
        cudaFree(dM);
        cudaFree(d_Rtmp);
        cudaFree(d_Ctmp);
        cudaFree(d_Ftmp);
        cudaFree(d_R);
        cudaFree(d_C);
        cudaFree(d_F);
        cudaFree(d_ops);
        cudaFree(d_pair_to_freq);
        cudaFree(d_mirror);
        cudaFree(d_omegas);
        cudaFree(d_gamma);
        cudaStreamDestroy(stream);
        return L4;
    } catch (...) {
        if (ws.plan) cufftDestroy(ws.plan);
        if (ws.A) cudaFree(ws.A);
        if (ws.B) cudaFree(ws.B);
        if (ws.B_conj) cudaFree(ws.B_conj);
        if (ws.scan_tmp) cudaFree(ws.scan_tmp);

        if (dL4) cudaFree(dL4);
        if (dGW) cudaFree(dGW);
        if (dX) cudaFree(dX);
        if (dK) cudaFree(dK);
        if (dI) cudaFree(dI);
        if (dM) cudaFree(dM);
        if (d_Rtmp) cudaFree(d_Rtmp);
        if (d_Ctmp) cudaFree(d_Ctmp);
        if (d_Ftmp) cudaFree(d_Ftmp);
        if (d_R) cudaFree(d_R);
        if (d_C) cudaFree(d_C);
        if (d_F) cudaFree(d_F);
        if (d_ops) cudaFree(d_ops);
        if (d_pair_to_freq) cudaFree(d_pair_to_freq);
        if (d_mirror) cudaFree(d_mirror);
        if (d_omegas) cudaFree(d_omegas);
        if (d_gamma) cudaFree(d_gamma);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }
}

} // namespace taco::tcl4
