#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace taco::tcl4::cuda_mikx {

namespace {

inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

__device__ __forceinline__ cuDoubleComplex cd_add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cuDoubleComplex cd_sub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ std::size_t idx3(std::size_t nf, int i, int j, int k) {
    return (static_cast<std::size_t>(i) * nf + static_cast<std::size_t>(j)) * nf + static_cast<std::size_t>(k);
}

__device__ __forceinline__ int pair_to_freq_at(const int* pair_to_freq, int N, int a, int b) {
    return pair_to_freq[static_cast<std::size_t>(a) + static_cast<std::size_t>(b) * static_cast<std::size_t>(N)];
}

__global__ void kernel_build_mik(const cuDoubleComplex* F,
                                 const cuDoubleComplex* R,
                                 const int* pair_to_freq,
                                 int N,
                                 int nf,
                                 cuDoubleComplex* M,
                                 cuDoubleComplex* I,
                                 cuDoubleComplex* K)
{
    // Column-major linearization over (row=(j,k), col=(p,q)):
    //   idx = row + col*N2
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N2 * N2) return;

    const std::size_t row_u = idx % N2;
    const std::size_t col_u = idx / N2;

    const int j = static_cast<int>(row_u % N_u);
    const int k = static_cast<int>(row_u / N_u);
    const int p = static_cast<int>(col_u % N_u);
    const int q = static_cast<int>(col_u / N_u);

    const int f_jk = pair_to_freq_at(pair_to_freq, N, j, k);
    const int f_jq = pair_to_freq_at(pair_to_freq, N, j, q);
    const int f_pj = pair_to_freq_at(pair_to_freq, N, p, j);
    const int f_pq = pair_to_freq_at(pair_to_freq, N, p, q);
    const int f_qk = pair_to_freq_at(pair_to_freq, N, q, k);
    const int f_kq = pair_to_freq_at(pair_to_freq, N, k, q);
    const int f_qp = pair_to_freq_at(pair_to_freq, N, q, p);
    const int f_qj = pair_to_freq_at(pair_to_freq, N, q, j);

    const std::size_t nf_u = static_cast<std::size_t>(nf);

    // M = F[f(j,q)][f(j,k)][f(p,j)] - R[f(j,q)][f(p,q)][f(q,k)]
    const cuDoubleComplex M1 = F[idx3(nf_u, f_jq, f_jk, f_pj)];
    const cuDoubleComplex M2 = R[idx3(nf_u, f_jq, f_pq, f_qk)];
    M[idx] = cd_sub(M1, M2);

    // I = F[f(j,k)][f(q,p)][f(k,q)]
    I[idx] = F[idx3(nf_u, f_jk, f_qp, f_kq)];

    // K = R[f(j,k)][f(p,q)][f(q,j)]
    K[idx] = R[idx3(nf_u, f_jk, f_pq, f_qj)];
}

__global__ void kernel_build_x(const cuDoubleComplex* C,
                               const cuDoubleComplex* R,
                               const int* pair_to_freq,
                               int N,
                               int nf,
                               cuDoubleComplex* X)
{
    // flat6(N,j,k,p,q,r,s) = j + N*(k + N*(p + N*(q + N*(r + N*s))))
    // so j is the fast/contiguous index.
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t N6 = N_u * N_u * N_u * N_u * N_u * N_u;
    if (idx >= N6) return;

    std::size_t t = idx;
    const int j = static_cast<int>(t % N_u); t /= N_u;
    const int k = static_cast<int>(t % N_u); t /= N_u;
    const int p = static_cast<int>(t % N_u); t /= N_u;
    const int q = static_cast<int>(t % N_u); t /= N_u;
    const int r = static_cast<int>(t % N_u); t /= N_u;
    const int s = static_cast<int>(t);

    const int f_jk = pair_to_freq_at(pair_to_freq, N, j, k);
    const int f_pq = pair_to_freq_at(pair_to_freq, N, p, q);
    const int f_rs = pair_to_freq_at(pair_to_freq, N, r, s);

    const std::size_t nf_u = static_cast<std::size_t>(nf);
    const std::size_t fidx = idx3(nf_u, f_jk, f_pq, f_rs);
    X[idx] = cd_add(C[fidx], R[fidx]);
}

} // namespace

void build_mikx_device(const MikxDeviceInputs& inputs,
                       const MikxDeviceOutputs& outputs,
                       cudaStream_t stream)
{
    if (inputs.N <= 0) throw std::invalid_argument("build_mikx_device: N must be > 0");
    if (inputs.nf <= 0) throw std::invalid_argument("build_mikx_device: nf must be > 0");
    if (!inputs.F || !inputs.C || !inputs.R) throw std::invalid_argument("build_mikx_device: null F/C/R");
    if (!inputs.pair_to_freq) throw std::invalid_argument("build_mikx_device: null pair_to_freq");
    if (!outputs.M || !outputs.I || !outputs.K || !outputs.X) throw std::invalid_argument("build_mikx_device: null outputs");

    constexpr int block = 256;

    const std::size_t N_u = static_cast<std::size_t>(inputs.N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t N4 = N2 * N2;
    const std::size_t N6 = N_u * N_u * N_u * N_u * N_u * N_u;

    const dim3 grid_MIK(static_cast<unsigned>((N4 + block - 1) / block));
    const dim3 grid_X(static_cast<unsigned>((N6 + block - 1) / block));

    kernel_build_mik<<<grid_MIK, block, 0, stream>>>(
        inputs.F, inputs.R, inputs.pair_to_freq, inputs.N, inputs.nf, outputs.M, outputs.I, outputs.K);
    cuda_check(cudaGetLastError(), "kernel_build_mik launch");

    kernel_build_x<<<grid_X, block, 0, stream>>>(
        inputs.C, inputs.R, inputs.pair_to_freq, inputs.N, inputs.nf, outputs.X);
    cuda_check(cudaGetLastError(), "kernel_build_x launch");
}

} // namespace taco::tcl4::cuda_mikx

namespace taco::tcl4 {

namespace {

inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

inline std::size_t idx3(std::size_t nf, std::size_t i, std::size_t j, std::size_t k) {
    return (i * nf + j) * nf + k;
}

inline std::size_t pow6(std::size_t N) {
    return N * N * N * N * N * N;
}

} // namespace

MikxTensors build_mikx_cuda(const Tcl4Map& map,
                            const TripleKernelSeries& kernels,
                            std::size_t time_index,
                            const Exec& exec)
{
    if (map.N <= 0) throw std::invalid_argument("build_mikx_cuda: map.N must be > 0");
    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t N2 = N * N;

    const std::size_t nf = static_cast<std::size_t>(map.nf);
    if (nf == 0) throw std::invalid_argument("build_mikx_cuda: map.nf must be > 0");

    // Deduce Nt from any one entry (assume consistent).
    if (kernels.F.empty() || kernels.F.front().empty() || kernels.F.front().front().empty()) {
        throw std::invalid_argument("build_mikx_cuda: kernels.F is empty");
    }
    const std::size_t Nt = static_cast<std::size_t>(kernels.F.front().front().front().size());
    if (time_index >= Nt) throw std::out_of_range("build_mikx_cuda: time_index out of range");

    if (map.pair_to_freq.rows() != static_cast<Eigen::Index>(N) ||
        map.pair_to_freq.cols() != static_cast<Eigen::Index>(N)) {
        throw std::invalid_argument("build_mikx_cuda: map.pair_to_freq has wrong shape");
    }
    if (map.pair_to_freq.minCoeff() < 0) {
        throw std::runtime_error("build_mikx_cuda: map.pair_to_freq contains -1 (missing frequency buckets)");
    }

    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> must match cuDoubleComplex storage (2 doubles)");

    // Flatten F/C/R at `time_index` into [nf^3] host buffers.
    const std::size_t nf3 = nf * nf * nf;
    std::vector<std::complex<double>> hF(nf3), hC(nf3), hR(nf3);
    for (std::size_t i = 0; i < nf; ++i) {
        for (std::size_t j = 0; j < nf; ++j) {
            for (std::size_t k = 0; k < nf; ++k) {
                const std::size_t at = idx3(nf, i, j, k);
                hF[at] = kernels.F[i][j][k](static_cast<Eigen::Index>(time_index));
                hC[at] = kernels.C[i][j][k](static_cast<Eigen::Index>(time_index));
                hR[at] = kernels.R[i][j][k](static_cast<Eigen::Index>(time_index));
            }
        }
    }

    cuda_check(cudaSetDevice(exec.gpu_id), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    cuDoubleComplex* dF = nullptr;
    cuDoubleComplex* dC = nullptr;
    cuDoubleComplex* dR = nullptr;
    int* d_pair_to_freq = nullptr;
    cuDoubleComplex* dM = nullptr;
    cuDoubleComplex* dI = nullptr;
    cuDoubleComplex* dK = nullptr;
    cuDoubleComplex* dX = nullptr;

    try {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dF), nf3 * sizeof(cuDoubleComplex)), "cudaMalloc(dF)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dC), nf3 * sizeof(cuDoubleComplex)), "cudaMalloc(dC)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dR), nf3 * sizeof(cuDoubleComplex)), "cudaMalloc(dR)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pair_to_freq), N2 * sizeof(int)), "cudaMalloc(d_pair_to_freq)");

        cuda_check(cudaMemcpyAsync(dF, hF.data(), nf3 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(F)");
        cuda_check(cudaMemcpyAsync(dC, hC.data(), nf3 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(C)");
        cuda_check(cudaMemcpyAsync(dR, hR.data(), nf3 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(R)");
        cuda_check(cudaMemcpyAsync(d_pair_to_freq,
                                   map.pair_to_freq.data(),
                                   N2 * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync(pair_to_freq)");

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dM), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dM)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dI), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dI)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dK), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dK)");
        const std::size_t N6 = pow6(N);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dX), N6 * sizeof(cuDoubleComplex)), "cudaMalloc(dX)");

        cuda_mikx::MikxDeviceInputs in;
        in.F = dF;
        in.C = dC;
        in.R = dR;
        in.pair_to_freq = d_pair_to_freq;
        in.N = map.N;
        in.nf = static_cast<int>(nf);

        cuda_mikx::MikxDeviceOutputs out;
        out.M = dM;
        out.I = dI;
        out.K = dK;
        out.X = dX;

        cuda_mikx::build_mikx_device(in, out, stream);

        MikxTensors tensors;
        tensors.N = map.N;
        tensors.M = Eigen::MatrixXcd(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        tensors.I = Eigen::MatrixXcd(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        tensors.K = Eigen::MatrixXcd(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        tensors.X.resize(N6);

        cuda_check(cudaMemcpyAsync(tensors.M.data(), dM, N2 * N2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(M)");
        cuda_check(cudaMemcpyAsync(tensors.I.data(), dI, N2 * N2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(I)");
        cuda_check(cudaMemcpyAsync(tensors.K.data(), dK, N2 * N2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(K)");
        cuda_check(cudaMemcpyAsync(tensors.X.data(), dX, N6 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(X)");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        cudaFree(dX);
        cudaFree(dK);
        cudaFree(dI);
        cudaFree(dM);
        cudaFree(d_pair_to_freq);
        cudaFree(dR);
        cudaFree(dC);
        cudaFree(dF);
        cudaStreamDestroy(stream);
        return tensors;
    } catch (...) {
        if (dX) cudaFree(dX);
        if (dK) cudaFree(dK);
        if (dI) cudaFree(dI);
        if (dM) cudaFree(dM);
        if (d_pair_to_freq) cudaFree(d_pair_to_freq);
        if (dR) cudaFree(dR);
        if (dC) cudaFree(dC);
        if (dF) cudaFree(dF);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }
}

} // namespace taco::tcl4

