#include "taco/backend/cuda/tcl4_assemble_cuda.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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

__device__ __forceinline__ cuDoubleComplex cd_add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cuDoubleComplex cd_sub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ cuDoubleComplex cd_mul(cuDoubleComplex a, cuDoubleComplex b) {
    const double real = fma(a.x, b.x, -a.y * b.y);
    const double imag = fma(a.x, b.y, a.y * b.x);
    return make_cuDoubleComplex(real, imag);
}

__device__ __forceinline__ cuDoubleComplex cd_conj(cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x, -a.y);
}

__device__ __forceinline__ cuDoubleComplex cd_neg(cuDoubleComplex a) {
    return make_cuDoubleComplex(-a.x, -a.y);
}

inline int read_env_int(const char* name, int fallback) {
#ifdef _MSC_VER
    char* buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || !buf) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(buf, &end, 10);
    std::free(buf);
#else
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) return fallback;
#endif
    if (parsed > std::numeric_limits<int>::max() || parsed < std::numeric_limits<int>::min()) return fallback;
    return static_cast<int>(parsed);
}

inline bool gw_profile_enabled() {
    static const bool enabled = (read_env_int("TACO_CUDA_GW_PROFILE", 0) != 0);
    return enabled;
}

template <bool Diag>
__device__ __forceinline__ cuDoubleComplex compute_gw_sum(const cuDoubleComplex* __restrict__ M,
                                                          const cuDoubleComplex* __restrict__ I,
                                                          const cuDoubleComplex* __restrict__ K,
                                                          const cuDoubleComplex* __restrict__ X,
                                                          const cuDoubleComplex* __restrict__ coupling,
                                                          int N,
                                                          int num_ops,
                                                          std::size_t N_u,
                                                          std::size_t N2,
                                                          std::size_t N3,
                                                          std::size_t N4,
                                                          std::size_t N5,
                                                          std::size_t n_u,
                                                          std::size_t i_u,
                                                          std::size_t j_u,
                                                          std::size_t m_u)
{
    const std::size_t nN = n_u * N_u;
    const std::size_t iN = i_u * N_u;
    const std::size_t jN2 = j_u * N2;
    const std::size_t mN3 = m_u * N3;
    const std::size_t nN4 = n_u * N4;
    const std::size_t iN5 = i_u * N5;
    const std::size_t base_anjbni = nN + jN2 + nN4 + iN5;
    const std::size_t base_iajbni = i_u + jN2 + nN4 + iN5;
    const std::size_t base_bajmai = jN2 + mN3 + iN5;
    const std::size_t base_ibjmai = i_u + jN2 + mN3 + iN5;
    const std::size_t N4_plus_N = N4 + N_u;

    cuDoubleComplex res = make_cuDoubleComplex(0.0, 0.0);

    for (int iA = 0; iA < num_ops; ++iA) {
        const cuDoubleComplex* __restrict__ A = coupling + static_cast<std::size_t>(iA) * N2;
        const cuDoubleComplex* __restrict__ A_col_m = A + m_u * N_u;
        const cuDoubleComplex* __restrict__ A_col_i = A + i_u * N_u;
        const cuDoubleComplex A_jm = A_col_m[j_u];
        for (int iB = 0; iB < num_ops; ++iB) {
            const cuDoubleComplex* __restrict__ B = coupling + static_cast<std::size_t>(iB) * N2;
            const cuDoubleComplex* __restrict__ B_col_m = B + m_u * N_u;
            const cuDoubleComplex* __restrict__ B_col_i = B + i_u * N_u;
            const cuDoubleComplex B_jm = B_col_m[j_u];
            for (int a = 0; a < N; ++a) {
                const std::size_t a_u = static_cast<std::size_t>(a);
                const std::size_t aN = a_u * N_u;
                const std::size_t col_na = n_u + aN;
                const std::size_t row_an = a_u + nN;
                const cuDoubleComplex* __restrict__ A_col_a = A + aN;
                const cuDoubleComplex* __restrict__ B_col_a = B + aN;
                const cuDoubleComplex A_na = A_col_a[n_u];
                const cuDoubleComplex B_na = B_col_a[n_u];
                const cuDoubleComplex A_ai = A_col_i[a_u];
                const cuDoubleComplex B_ai = B_col_i[a_u];
                for (int b = 0; b < N; ++b) {
                    const std::size_t b_u = static_cast<std::size_t>(b);
                    const std::size_t bN = b_u * N_u;
                    const std::size_t row_bi = b_u + iN;
                    const std::size_t row_ib = i_u + bN;
                    const std::size_t col_ib = i_u + bN;
                    const cuDoubleComplex* __restrict__ A_col_b = A + bN;
                    const cuDoubleComplex* __restrict__ B_col_b = B + bN;
                    const cuDoubleComplex A_ab = A_col_b[a_u];
                    const cuDoubleComplex B_ab = B_col_b[a_u];
                    const cuDoubleComplex A_bi = A_col_i[b_u];
                    const cuDoubleComplex B_bi = B_col_i[b_u];
                    const cuDoubleComplex A_bm = A_col_m[b_u];
                    const cuDoubleComplex B_jb = B_col_b[j_u];

                    const std::size_t idx_M = row_bi + col_na * N2;
                    const std::size_t idx_I = row_an + col_ib * N2;
                    const std::size_t idx_K = row_ib + col_na * N2;

                    const cuDoubleComplex M_bi_na = M[idx_M];
                    const cuDoubleComplex I_an_ib = I[idx_I];
                    const cuDoubleComplex K_ib_na = K[idx_K];

                    const std::size_t idx_anjbni = base_anjbni + a_u + N3 * b_u;
                    const std::size_t idx_iajbni = base_iajbni + a_u * N_u + N3 * b_u;
                    const std::size_t idx_bajmai = base_bajmai + b_u + a_u * N4_plus_N;
                    const std::size_t idx_ibjmai = base_ibjmai + b_u * N_u + a_u * N4;
                    const cuDoubleComplex X_anjbni = X[idx_anjbni];
                    const cuDoubleComplex X_iajbni = X[idx_iajbni];
                    const cuDoubleComplex X_bajmai = X[idx_bajmai];
                    const cuDoubleComplex X_ibjmai = X[idx_ibjmai];

                    const cuDoubleComplex t1 = cd_mul(B_na, cd_mul(A_ab, cd_mul(B_bi, cd_mul(A_jm, M_bi_na))));
                    const cuDoubleComplex t2 = cd_mul(A_na, cd_mul(B_ab, cd_mul(B_bi, cd_mul(A_jm, I_an_ib))));
                    const cuDoubleComplex t3 = cd_mul(B_na, cd_mul(B_ab, cd_mul(A_bi, cd_mul(A_jm, K_ib_na))));
                    const cuDoubleComplex t4 = cd_mul(A_na, cd_mul(B_ai, cd_mul(B_jb, cd_mul(A_bm, X_anjbni))));
                    const cuDoubleComplex t5 = cd_mul(B_na, cd_mul(A_ai, cd_mul(B_jb, cd_mul(A_bm, X_iajbni))));
                    const cuDoubleComplex t6 = cd_mul(A_na, cd_mul(A_ab, cd_mul(B_bi, cd_mul(B_jm, X_bajmai))));
                    const cuDoubleComplex t7 = cd_mul(A_na, cd_mul(B_ab, cd_mul(A_bi, cd_mul(B_jm, X_ibjmai))));

                    cuDoubleComplex tmp = cd_add(cd_sub(t1, t2), cd_add(t3, t4));
                    tmp = cd_sub(tmp, t5);
                    tmp = cd_sub(tmp, t6);
                    tmp = cd_add(tmp, t7);
                    res = cd_sub(res, tmp);
                }
            }

            if constexpr (Diag) {
                for (int a = 0; a < N; ++a) {
                    const std::size_t a_u = static_cast<std::size_t>(a);
                    const std::size_t aN = a_u * N_u;
                    const std::size_t row_ba_base = aN;
                    const cuDoubleComplex* __restrict__ A_col_a = A + aN;
                    const cuDoubleComplex A_na = A_col_a[n_u];
                    for (int b = 0; b < N; ++b) {
                        const std::size_t b_u = static_cast<std::size_t>(b);
                        const std::size_t bN = b_u * N_u;
                        const std::size_t col_ab = a_u + bN;
                        const std::size_t row_ba = b_u + row_ba_base;
                        const cuDoubleComplex* __restrict__ A_col_b = A + bN;
                        const cuDoubleComplex* __restrict__ B_col_b = B + bN;
                        const cuDoubleComplex A_ab = A_col_b[a_u];
                        const cuDoubleComplex B_ab = B_col_b[a_u];
                        for (int c = 0; c < N; ++c) {
                            const std::size_t c_u = static_cast<std::size_t>(c);
                            const std::size_t cN = c_u * N_u;
                            const std::size_t row_ci = c_u + iN;
                            const std::size_t row_ic = i_u + cN;
                            const std::size_t col_ic = i_u + cN;
                            const cuDoubleComplex* __restrict__ A_col_c = A + cN;
                            const cuDoubleComplex* __restrict__ B_col_c = B + cN;
                            const cuDoubleComplex A_bc = A_col_c[b_u];
                            const cuDoubleComplex B_bc = B_col_c[b_u];
                            const cuDoubleComplex A_ci = A_col_i[c_u];
                            const cuDoubleComplex B_ci = B_col_i[c_u];

                            const std::size_t idx_M = row_ci + col_ab * N2;
                            const std::size_t idx_K = row_ic + col_ab * N2;
                            const std::size_t idx_I = row_ba + col_ic * N2;

                            const cuDoubleComplex M_ci_ab = M[idx_M];
                            const cuDoubleComplex K_ic_ab = K[idx_K];
                            const cuDoubleComplex I_ba_ic = I[idx_I];

                            const cuDoubleComplex add1 =
                                cd_mul(A_na, cd_mul(B_ab, cd_mul(A_bc, cd_mul(B_ci, M_ci_ab))));
                            const cuDoubleComplex add2 =
                                cd_mul(A_na, cd_mul(B_ab, cd_mul(B_bc, cd_mul(A_ci, K_ic_ab))));
                            const cuDoubleComplex add3 =
                                cd_neg(cd_mul(A_na, cd_mul(A_ab, cd_mul(B_bc, cd_mul(B_ci, I_ba_ic)))));

                            res = cd_add(res, cd_add(add1, cd_add(add2, add3)));
                        }
                    }
                }
            }
        }
    }

    return res;
}

__global__ void kernel_assemble_gw_offdiag(const cuDoubleComplex* __restrict__ M,
                                           const cuDoubleComplex* __restrict__ I,
                                           const cuDoubleComplex* __restrict__ K,
                                           const cuDoubleComplex* __restrict__ X,
                                           const cuDoubleComplex* __restrict__ coupling,
                                           int N,
                                           int num_ops,
                                           cuDoubleComplex* __restrict__ GW)
{
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t N3 = N2 * N_u;
    const std::size_t N4 = N2 * N2;
    const std::size_t N5 = N4 * N_u;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = N2 * N2;
    if (idx >= total) return;

    const std::size_t col_u = idx / N2;
    const std::size_t row_u = idx - col_u * N2;

    const std::size_t i_u = row_u / N_u;
    const std::size_t n_u = row_u - i_u * N_u;
    const std::size_t j_u = col_u / N_u;
    const std::size_t m_u = col_u - j_u * N_u;

    if (j_u == m_u) return;

    const cuDoubleComplex res = compute_gw_sum<false>(
        M, I, K, X, coupling, N, num_ops, N_u, N2, N3, N4, N5, n_u, i_u, j_u, m_u);
    GW[idx] = res;
}

__global__ void kernel_assemble_gw_diag(const cuDoubleComplex* __restrict__ M,
                                        const cuDoubleComplex* __restrict__ I,
                                        const cuDoubleComplex* __restrict__ K,
                                        const cuDoubleComplex* __restrict__ X,
                                        const cuDoubleComplex* __restrict__ coupling,
                                        int N,
                                        int num_ops,
                                        cuDoubleComplex* __restrict__ GW)
{
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t N3 = N2 * N_u;
    const std::size_t N4 = N2 * N2;
    const std::size_t N5 = N4 * N_u;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = N2 * N_u;
    if (idx >= total) return;

    const std::size_t j_u = idx / N2;
    const std::size_t row_u = idx - j_u * N2;
    const std::size_t i_u = row_u / N_u;
    const std::size_t n_u = row_u - i_u * N_u;
    const std::size_t m_u = j_u;

    const cuDoubleComplex res = compute_gw_sum<true>(
        M, I, K, X, coupling, N, num_ops, N_u, N2, N3, N4, N5, n_u, i_u, j_u, m_u);

    const std::size_t col_u = j_u + j_u * N_u;
    const std::size_t out_idx = row_u + col_u * N2;
    GW[out_idx] = res;
}

__global__ void kernel_symmetrize_gw(cuDoubleComplex* __restrict__ GW, std::size_t N2) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = N2 * N2;
    if (idx >= total) return;
    const std::size_t col = idx / N2;
    const std::size_t row = idx - col * N2;
    if (row > col) return;

    const cuDoubleComplex t_rc = GW[row + col * N2];
    const cuDoubleComplex t_cr = GW[col + row * N2];
    const cuDoubleComplex g_rc = cd_add(t_rc, cd_conj(t_cr));
    GW[row + col * N2] = g_rc;
    if (row != col) {
        GW[col + row * N2] = cd_conj(g_rc);
    }
}

} // namespace

void assemble_liouvillian_cuda_device(const cuDoubleComplex* dM,
                                      const cuDoubleComplex* dI,
                                      const cuDoubleComplex* dK,
                                      const cuDoubleComplex* dX,
                                      const cuDoubleComplex* d_ops,
                                      int N,
                                      int num_ops,
                                      cuDoubleComplex* dGW,
                                      cudaStream_t stream)
{
    if (N <= 0) {
        throw std::invalid_argument("assemble_liouvillian_cuda_device: N must be > 0");
    }
    if (num_ops <= 0) {
        throw std::invalid_argument("assemble_liouvillian_cuda_device: num_ops must be > 0");
    }
    if (!dM || !dI || !dK || !dX || !d_ops || !dGW) {
        throw std::invalid_argument("assemble_liouvillian_cuda_device: null device pointer");
    }

    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;

    constexpr int block = 128;
    const std::size_t total_off = N2 * N2;
    const dim3 grid_off(static_cast<unsigned>((total_off + block - 1) / block));
    kernel_assemble_gw_offdiag<<<grid_off, block, 0, stream>>>(
        dM, dI, dK, dX, d_ops, N, num_ops, dGW);
    cuda_check(cudaGetLastError(), "kernel_assemble_gw_offdiag launch");

    const std::size_t total_diag = N2 * N_u;
    const dim3 grid_diag(static_cast<unsigned>((total_diag + block - 1) / block));
    kernel_assemble_gw_diag<<<grid_diag, block, 0, stream>>>(
        dM, dI, dK, dX, d_ops, N, num_ops, dGW);
    cuda_check(cudaGetLastError(), "kernel_assemble_gw_diag launch");

    kernel_symmetrize_gw<<<grid_off, block, 0, stream>>>(dGW, N2);
    cuda_check(cudaGetLastError(), "kernel_symmetrize_gw launch");
}

Eigen::MatrixXcd assemble_liouvillian_cuda(const MikxTensors& tensors,
                                           const std::vector<Eigen::MatrixXcd>& coupling_ops,
                                           const Exec& exec)
{
    if (tensors.N <= 0) {
        throw std::invalid_argument("assemble_liouvillian_cuda: tensors.N must be > 0");
    }
    const std::size_t N = static_cast<std::size_t>(tensors.N);
    const std::size_t N2 = N * N;
    const std::size_t N6 = N * N * N * N * N * N;

    if (tensors.M.rows() != static_cast<Eigen::Index>(N2) || tensors.M.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.I.rows() != static_cast<Eigen::Index>(N2) || tensors.I.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.K.rows() != static_cast<Eigen::Index>(N2) || tensors.K.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.X.size() != N6) {
        throw std::invalid_argument("assemble_liouvillian_cuda: tensor dimensions do not match N");
    }

    if (coupling_ops.empty()) {
        throw std::invalid_argument("assemble_liouvillian_cuda: coupling_ops must be non-empty");
    }
    for (const auto& C : coupling_ops) {
        if (C.rows() != static_cast<Eigen::Index>(N) || C.cols() != static_cast<Eigen::Index>(N)) {
            throw std::invalid_argument("assemble_liouvillian_cuda: coupling operator has wrong shape");
        }
    }
    if (coupling_ops.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("assemble_liouvillian_cuda: coupling_ops too large for CUDA kernel");
    }

    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> must match cuDoubleComplex storage (2 doubles)");

    const bool profile = gw_profile_enabled();
    const auto wall_start = profile ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};

    const std::size_t num_ops = coupling_ops.size();
    std::vector<std::complex<double>> h_ops(num_ops * N2);
    for (std::size_t op = 0; op < num_ops; ++op) {
        const auto& A = coupling_ops[op];
        std::copy(A.data(), A.data() + N2, h_ops.data() + op * N2);
    }

    cuda_check(cudaSetDevice(exec.gpu_id), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    cuDoubleComplex* dM = nullptr;
    cuDoubleComplex* dI = nullptr;
    cuDoubleComplex* dK = nullptr;
    cuDoubleComplex* dX = nullptr;
    cuDoubleComplex* d_ops = nullptr;
    cuDoubleComplex* dGW = nullptr;

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_after_off = nullptr;
    cudaEvent_t ev_after_diag = nullptr;
    cudaEvent_t ev_after_sym = nullptr;

    try {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dM), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dM)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dI), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dI)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dK), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dK)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dX), N6 * sizeof(cuDoubleComplex)), "cudaMalloc(dX)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ops), num_ops * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(d_ops)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dGW), N2 * N2 * sizeof(cuDoubleComplex)), "cudaMalloc(dGW)");

        cuda_check(cudaMemcpyAsync(dM, tensors.M.data(), N2 * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(M)");
        cuda_check(cudaMemcpyAsync(dI, tensors.I.data(), N2 * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(I)");
        cuda_check(cudaMemcpyAsync(dK, tensors.K.data(), N2 * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(K)");
        cuda_check(cudaMemcpyAsync(dX, tensors.X.data(), N6 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(X)");
        cuda_check(cudaMemcpyAsync(d_ops, h_ops.data(), num_ops * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(coupling)");

        if (profile) {
            cuda_check(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
            cuda_check(cudaEventCreate(&ev_after_off), "cudaEventCreate(after_off)");
            cuda_check(cudaEventCreate(&ev_after_diag), "cudaEventCreate(after_diag)");
            cuda_check(cudaEventCreate(&ev_after_sym), "cudaEventCreate(after_sym)");
            cuda_check(cudaEventRecord(ev_start, stream), "cudaEventRecord(start)");
        }

        constexpr int block = 128;
        const std::size_t total_off = N2 * N2;
        const dim3 grid_off(static_cast<unsigned>((total_off + block - 1) / block));
        kernel_assemble_gw_offdiag<<<grid_off, block, 0, stream>>>(
            dM, dI, dK, dX, d_ops, static_cast<int>(N), static_cast<int>(num_ops), dGW);
        cuda_check(cudaGetLastError(), "kernel_assemble_gw_offdiag launch");
        if (profile) {
            cuda_check(cudaEventRecord(ev_after_off, stream), "cudaEventRecord(after_off)");
        }

        const std::size_t total_diag = N2 * N;
        const dim3 grid_diag(static_cast<unsigned>((total_diag + block - 1) / block));
        kernel_assemble_gw_diag<<<grid_diag, block, 0, stream>>>(
            dM, dI, dK, dX, d_ops, static_cast<int>(N), static_cast<int>(num_ops), dGW);
        cuda_check(cudaGetLastError(), "kernel_assemble_gw_diag launch");
        if (profile) {
            cuda_check(cudaEventRecord(ev_after_diag, stream), "cudaEventRecord(after_diag)");
        }

        kernel_symmetrize_gw<<<grid_off, block, 0, stream>>>(dGW, N2);
        cuda_check(cudaGetLastError(), "kernel_symmetrize_gw launch");
        if (profile) {
            cuda_check(cudaEventRecord(ev_after_sym, stream), "cudaEventRecord(after_sym)");
        }

        Eigen::MatrixXcd GW(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        cuda_check(cudaMemcpyAsync(GW.data(), dGW, N2 * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(GW)");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        if (profile) {
            cuda_check(cudaEventSynchronize(ev_after_sym), "cudaEventSynchronize(after_sym)");
            float ms_off = 0.0f;
            float ms_diag = 0.0f;
            float ms_sym = 0.0f;
            float ms_total = 0.0f;
            cuda_check(cudaEventElapsedTime(&ms_off, ev_start, ev_after_off), "cudaEventElapsedTime(off)");
            cuda_check(cudaEventElapsedTime(&ms_diag, ev_after_off, ev_after_diag), "cudaEventElapsedTime(diag)");
            cuda_check(cudaEventElapsedTime(&ms_sym, ev_after_diag, ev_after_sym), "cudaEventElapsedTime(sym)");
            cuda_check(cudaEventElapsedTime(&ms_total, ev_start, ev_after_sym), "cudaEventElapsedTime(total)");

            const auto wall_end = std::chrono::high_resolution_clock::now();
            const double wall_ms =
                std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

            std::fprintf(stdout,
                         "cuda_gw: wall_ms=%.3f kernel_ms=%.3f offdiag_ms=%.3f diag_ms=%.3f sym_ms=%.3f\n",
                         wall_ms,
                         static_cast<double>(ms_total),
                         static_cast<double>(ms_off),
                         static_cast<double>(ms_diag),
                         static_cast<double>(ms_sym));
        }

        if (ev_after_sym) cudaEventDestroy(ev_after_sym);
        if (ev_after_diag) cudaEventDestroy(ev_after_diag);
        if (ev_after_off) cudaEventDestroy(ev_after_off);
        if (ev_start) cudaEventDestroy(ev_start);

        cudaFree(dGW);
        cudaFree(d_ops);
        cudaFree(dX);
        cudaFree(dK);
        cudaFree(dI);
        cudaFree(dM);
        cudaStreamDestroy(stream);
        return GW;
    } catch (...) {
        if (ev_after_sym) cudaEventDestroy(ev_after_sym);
        if (ev_after_diag) cudaEventDestroy(ev_after_diag);
        if (ev_after_off) cudaEventDestroy(ev_after_off);
        if (ev_start) cudaEventDestroy(ev_start);
        if (dGW) cudaFree(dGW);
        if (d_ops) cudaFree(d_ops);
        if (dX) cudaFree(dX);
        if (dK) cudaFree(dK);
        if (dI) cudaFree(dI);
        if (dM) cudaFree(dM);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }
}

} // namespace taco::tcl4
