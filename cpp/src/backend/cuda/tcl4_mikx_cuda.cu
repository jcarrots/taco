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

__global__ void kernel_build_mik(const cuDoubleComplex* __restrict__ F,
                                 const cuDoubleComplex* __restrict__ R,
                                 const int* __restrict__ pair_to_freq,
                                 int N,
                                 int nf,
                                 cuDoubleComplex* __restrict__ M,
                                 cuDoubleComplex* __restrict__ I,
                                 cuDoubleComplex* __restrict__ K)
{
    // Column-major linearization over (row=(j,k), col=(p,q)):
    //   idx = row + col*N2
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N2 * N2) return;

    const std::size_t row_u = idx % N2;
    const std::size_t col_u = idx / N2;

    const std::size_t j_u = row_u % N_u;
    const std::size_t k_u = row_u / N_u;
    const std::size_t p_u = col_u % N_u;
    const std::size_t q_u = col_u / N_u;

    const std::size_t jk = row_u;
    const std::size_t jq = j_u + N_u * q_u;
    const std::size_t pj = p_u + N_u * j_u;
    const std::size_t pq = col_u;
    const std::size_t qk = q_u + N_u * k_u;
    const std::size_t kq = k_u + N_u * q_u;
    const std::size_t qp = q_u + N_u * p_u;
    const std::size_t qj = q_u + N_u * j_u;

    const int f_jk = pair_to_freq[jk];
    const int f_jq = pair_to_freq[jq];
    const int f_pj = pair_to_freq[pj];
    const int f_pq = pair_to_freq[pq];
    const int f_qk = pair_to_freq[qk];
    const int f_kq = pair_to_freq[kq];
    const int f_qp = pair_to_freq[qp];
    const int f_qj = pair_to_freq[qj];

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

__global__ void kernel_build_x(const cuDoubleComplex* __restrict__ C,
                               const cuDoubleComplex* __restrict__ R,
                               const int* __restrict__ pair_to_freq,
                               int N,
                               int nf,
                               cuDoubleComplex* __restrict__ X)
{
    // flat6(N,j,k,p,q,r,s) = j + N*(k + N*(p + N*(q + N*(r + N*s))))
    // so j is the fast/contiguous index.
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t N4 = N2 * N2;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t N6 = N4 * N2;
    if (idx >= N6) return;

    const std::size_t jk = idx % N2;
    const std::size_t pq = (idx / N2) % N2;
    const std::size_t rs = idx / N4;

    const int f_jk = pair_to_freq[jk];
    const int f_pq = pair_to_freq[pq];
    const int f_rs = pair_to_freq[rs];

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

namespace {
MikxTensors build_mikx_cuda_impl(const Tcl4Map& map,
                                 const TripleKernelSeries& kernels,
                                 std::size_t time_index,
                                 const Exec& exec,
                                 double* kernel_ms)
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

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    if (kernel_ms) *kernel_ms = 0.0;
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

        if (kernel_ms) {
            cuda_check(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
            cuda_check(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)");
            cuda_check(cudaEventRecord(ev_start, stream), "cudaEventRecord(start)");
        }

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

        if (kernel_ms) {
            cuda_check(cudaEventRecord(ev_stop, stream), "cudaEventRecord(stop)");
            cuda_check(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");
            float ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&ms, ev_start, ev_stop), "cudaEventElapsedTime");
            *kernel_ms += static_cast<double>(ms);
            cuda_check(cudaEventDestroy(ev_start), "cudaEventDestroy(start)");
            cuda_check(cudaEventDestroy(ev_stop), "cudaEventDestroy(stop)");
            ev_start = nullptr;
            ev_stop = nullptr;
        }

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
        if (kernel_ms) {
            if (ev_start) cudaEventDestroy(ev_start);
            if (ev_stop) cudaEventDestroy(ev_stop);
        }
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
} // namespace

MikxTensors build_mikx_cuda(const Tcl4Map& map,
                            const TripleKernelSeries& kernels,
                            std::size_t time_index,
                            const Exec& exec)
{
    return build_mikx_cuda_impl(map, kernels, time_index, exec, nullptr);
}

MikxTensors build_mikx_cuda(const Tcl4Map& map,
                            const TripleKernelSeries& kernels,
                            std::size_t time_index,
                            const Exec& exec,
                            double* kernel_ms)
{
    double ms = 0.0;
    MikxTensors out = build_mikx_cuda_impl(map, kernels, time_index, exec, kernel_ms ? &ms : nullptr);
    if (kernel_ms) *kernel_ms = ms;
    return out;
}

namespace {
std::vector<MikxTensors> build_mikx_cuda_batch_impl(const Tcl4Map& map,
                                                    const TripleKernelSeries& kernels,
                                                    const std::vector<std::size_t>& time_indices,
                                                    const Exec& exec,
                                                    std::size_t batch_size,
                                                    double* kernel_ms_total)
{
    std::vector<MikxTensors> out;
    if (time_indices.empty()) return out;

    if (map.N <= 0) throw std::invalid_argument("build_mikx_cuda_batch: map.N must be > 0");
    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t N2 = N * N;

    const std::size_t nf = static_cast<std::size_t>(map.nf);
    if (nf == 0) throw std::invalid_argument("build_mikx_cuda_batch: map.nf must be > 0");

    if (kernels.F.empty() || kernels.F.front().empty() || kernels.F.front().front().empty()) {
        throw std::invalid_argument("build_mikx_cuda_batch: kernels.F is empty");
    }
    const std::size_t Nt = static_cast<std::size_t>(kernels.F.front().front().front().size());
    for (std::size_t ti : time_indices) {
        if (ti >= Nt) throw std::out_of_range("build_mikx_cuda_batch: time_index out of range");
    }

    if (map.pair_to_freq.rows() != static_cast<Eigen::Index>(N) ||
        map.pair_to_freq.cols() != static_cast<Eigen::Index>(N)) {
        throw std::invalid_argument("build_mikx_cuda_batch: map.pair_to_freq has wrong shape");
    }
    if (map.pair_to_freq.minCoeff() < 0) {
        throw std::runtime_error("build_mikx_cuda_batch: map.pair_to_freq contains -1 (missing frequency buckets)");
    }

    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> must match cuDoubleComplex storage (2 doubles)");

    const std::size_t nf3 = nf * nf * nf;
    const std::size_t total = time_indices.size();
    if (batch_size == 0 || batch_size > total) batch_size = total;

    const std::size_t bytes_per_t = nf3 * sizeof(cuDoubleComplex) * 3;
    if (bytes_per_t > 0) {
        const std::size_t max_bytes = 2048ull * 1024ull * 1024ull;
        const std::size_t max_batch = std::max<std::size_t>(1, max_bytes / bytes_per_t);
        if (batch_size > max_batch) batch_size = max_batch;
    } else {
        batch_size = 1;
    }

    out.resize(total);

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

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    if (kernel_ms_total) *kernel_ms_total = 0.0;
    try {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pair_to_freq), N2 * sizeof(int)), "cudaMalloc(d_pair_to_freq)");
        cuda_check(cudaMemcpyAsync(d_pair_to_freq,
                                   map.pair_to_freq.data(),
                                   N2 * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync(pair_to_freq)");

        const std::size_t N6 = pow6(N);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dM), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dM)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dI), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dI)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dK), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dK)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dX), N6 * sizeof(cuDoubleComplex)), "cudaMalloc(dX)");

        const auto& F = kernels.F;
        const auto& C = kernels.C;
        const auto& R = kernels.R;

        if (kernel_ms_total) {
            cuda_check(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
            cuda_check(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)");
        }

        for (std::size_t base = 0; base < total; base += batch_size) {
            const std::size_t B = std::min(batch_size, total - base);
            const std::size_t chunk_elems = B * nf3;

            std::vector<std::complex<double>> hF(chunk_elems);
            std::vector<std::complex<double>> hC(chunk_elems);
            std::vector<std::complex<double>> hR(chunk_elems);

            for (std::size_t b = 0; b < B; ++b) {
                const std::size_t ti = time_indices[base + b];
                const std::size_t offset = b * nf3;
                for (std::size_t i = 0; i < nf; ++i) {
                    for (std::size_t j = 0; j < nf; ++j) {
                        for (std::size_t k = 0; k < nf; ++k) {
                            const std::size_t at = idx3(nf, i, j, k);
                            const Eigen::Index ti_e = static_cast<Eigen::Index>(ti);
                            hF[offset + at] = F[i][j][k](ti_e);
                            hC[offset + at] = C[i][j][k](ti_e);
                            hR[offset + at] = R[i][j][k](ti_e);
                        }
                    }
                }
            }

            cuda_check(cudaMalloc(reinterpret_cast<void**>(&dF), chunk_elems * sizeof(cuDoubleComplex)), "cudaMalloc(dF)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&dC), chunk_elems * sizeof(cuDoubleComplex)), "cudaMalloc(dC)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&dR), chunk_elems * sizeof(cuDoubleComplex)), "cudaMalloc(dR)");

            cuda_check(cudaMemcpyAsync(dF, hF.data(), chunk_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(F)");
            cuda_check(cudaMemcpyAsync(dC, hC.data(), chunk_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(C)");
            cuda_check(cudaMemcpyAsync(dR, hR.data(), chunk_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(R)");

            for (std::size_t b = 0; b < B; ++b) {
                const std::size_t out_idx = base + b;
                const std::size_t offset = b * nf3;

                cuda_mikx::MikxDeviceInputs in;
                in.F = dF + offset;
                in.C = dC + offset;
                in.R = dR + offset;
                in.pair_to_freq = d_pair_to_freq;
                in.N = map.N;
                in.nf = static_cast<int>(nf);

                cuda_mikx::MikxDeviceOutputs out_dev;
                out_dev.M = dM;
                out_dev.I = dI;
                out_dev.K = dK;
                out_dev.X = dX;

                if (kernel_ms_total) {
                    cuda_check(cudaEventRecord(ev_start, stream), "cudaEventRecord(start)");
                }
                cuda_mikx::build_mikx_device(in, out_dev, stream);
                if (kernel_ms_total) {
                    cuda_check(cudaEventRecord(ev_stop, stream), "cudaEventRecord(stop)");
                    cuda_check(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");
                    float ms = 0.0f;
                    cuda_check(cudaEventElapsedTime(&ms, ev_start, ev_stop), "cudaEventElapsedTime");
                    *kernel_ms_total += static_cast<double>(ms);
                }

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

                out[out_idx] = std::move(tensors);
            }

            cudaFree(dR);
            cudaFree(dC);
            cudaFree(dF);
            dR = nullptr;
            dC = nullptr;
            dF = nullptr;
        }

        if (kernel_ms_total) {
            cuda_check(cudaEventDestroy(ev_start), "cudaEventDestroy(start)");
            cuda_check(cudaEventDestroy(ev_stop), "cudaEventDestroy(stop)");
            ev_start = nullptr;
            ev_stop = nullptr;
        }

        cudaFree(dX);
        cudaFree(dK);
        cudaFree(dI);
        cudaFree(dM);
        cudaFree(d_pair_to_freq);
        cudaStreamDestroy(stream);
        return out;
    } catch (...) {
        if (kernel_ms_total) {
            if (ev_start) cudaEventDestroy(ev_start);
            if (ev_stop) cudaEventDestroy(ev_stop);
        }
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
} // namespace

std::vector<MikxTensors> build_mikx_cuda_batch(const Tcl4Map& map,
                                               const TripleKernelSeries& kernels,
                                               const std::vector<std::size_t>& time_indices,
                                               const Exec& exec,
                                               std::size_t batch_size)
{
    return build_mikx_cuda_batch_impl(map, kernels, time_indices, exec, batch_size, nullptr);
}

std::vector<MikxTensors> build_mikx_cuda_batch(const Tcl4Map& map,
                                               const TripleKernelSeries& kernels,
                                               const std::vector<std::size_t>& time_indices,
                                               const Exec& exec,
                                               std::size_t batch_size,
                                               double* kernel_ms_total)
{
    double ms = 0.0;
    auto out = build_mikx_cuda_batch_impl(map, kernels, time_indices, exec, batch_size, kernel_ms_total ? &ms : nullptr);
    if (kernel_ms_total) *kernel_ms_total = ms;
    return out;
}

} // namespace taco::tcl4

