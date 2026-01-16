#include "taco/backend/cuda/rk4_dense_cuda.hpp"

#ifdef TACO_HAS_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <stdexcept>
#include <string>

namespace taco::tcl {

namespace {

inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

inline dim3 grid_1d(std::size_t n, int block) {
    return dim3(static_cast<unsigned int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block)));
}

inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void cublas_check(cublasStatus_t status, const char* what) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": " + cublas_status_string(status));
}

template <typename WorkspaceT>
inline cublasHandle_t ensure_cublas_handle(WorkspaceT& ws, cudaStream_t stream) {
    cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(ws.cublas_handle);
    if (!handle) {
        cublas_check(cublasCreate(&handle), "cublasCreate");
        ws.cublas_handle = reinterpret_cast<void*>(handle);
    }
    cublas_check(cublasSetStream(handle, stream), "cublasSetStream");
    cublas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    return handle;
}

inline void matvec_dense_cuda_colmajor_cublas(const cuDoubleComplex* L,
                                              const cuDoubleComplex* x,
                                              cuDoubleComplex* y,
                                              int D,
                                              cublasHandle_t handle) {
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    cublas_check(cublasZgemv(handle,
                             CUBLAS_OP_N,
                             D,
                             D,
                             &alpha,
                             L,
                             D,
                             x,
                             1,
                             &beta,
                             y,
                             1),
                 "cublasZgemv");
}

inline void matvec_dense_cuda_colmajor_cublas_f32(const cuFloatComplex* L,
                                                  const cuFloatComplex* x,
                                                  cuFloatComplex* y,
                                                  int D,
                                                  cublasHandle_t handle) {
    const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
    cublas_check(cublasCgemv(handle,
                             CUBLAS_OP_N,
                             D,
                             D,
                             &alpha,
                             reinterpret_cast<const cuComplex*>(L),
                             D,
                             reinterpret_cast<const cuComplex*>(x),
                             1,
                             &beta,
                             reinterpret_cast<cuComplex*>(y),
                             1),
                 "cublasCgemv");
}

// Dense matrix-vector multiply y = L * x with L stored in column-major order.
//
// Implementation notes:
// - 1 warp (32 threads) computes 32 consecutive rows (coalesced column loads).
// - For each 32-column tile, lanes load one x element and broadcast via shuffles.
__global__ void matvec_dense_cuda_colmajor_warp(const cuDoubleComplex* __restrict__ L,
                                                const cuDoubleComplex* __restrict__ x,
                                                cuDoubleComplex* __restrict__ y,
                                                int D) {
    const int lane = static_cast<int>(threadIdx.x); // 0..31
    const int row = static_cast<int>(blockIdx.x) * 32 + lane;

    double sum_x = 0.0;
    double sum_y = 0.0;

    // Iterate over columns in 32-wide tiles.
    for (int j0 = 0; j0 < D; j0 += 32) {
        const int col = j0 + lane;
        const cuDoubleComplex b = (col < D) ? x[col] : make_cuDoubleComplex(0.0, 0.0);

        #pragma unroll
        for (int s = 0; s < 32; ++s) {
            const int j = j0 + s;
            const double bx = __shfl_sync(0xffffffff, b.x, s);
            const double by = __shfl_sync(0xffffffff, b.y, s);

            if (row < D && j < D) {
                const cuDoubleComplex a = L[static_cast<std::size_t>(row) + static_cast<std::size_t>(j) * static_cast<std::size_t>(D)];
                // sum += a * b (manual complex FMA to avoid cuCmul/cuCadd overhead)
                sum_x = fma(a.x, bx, sum_x);
                sum_x = fma(-a.y, by, sum_x);
                sum_y = fma(a.x, by, sum_y);
                sum_y = fma(a.y, bx, sum_y);
            }
        }
    }

    if (row < D) {
        y[row] = make_cuDoubleComplex(sum_x, sum_y);
    }
}

__global__ void matvec_dense_cuda_colmajor_warp_f32(const cuFloatComplex* __restrict__ L,
                                                    const cuFloatComplex* __restrict__ x,
                                                    cuFloatComplex* __restrict__ y,
                                                    int D) {
    const int lane = static_cast<int>(threadIdx.x); // 0..31
    const int row = static_cast<int>(blockIdx.x) * 32 + lane;

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (int j0 = 0; j0 < D; j0 += 32) {
        const int col = j0 + lane;
        const cuFloatComplex b = (col < D) ? x[col] : make_cuFloatComplex(0.0f, 0.0f);

        #pragma unroll
        for (int s = 0; s < 32; ++s) {
            const int j = j0 + s;
            const float bx = __shfl_sync(0xffffffff, b.x, s);
            const float by = __shfl_sync(0xffffffff, b.y, s);

            if (row < D && j < D) {
                const cuFloatComplex a = L[static_cast<std::size_t>(row) + static_cast<std::size_t>(j) * static_cast<std::size_t>(D)];
                sum_x = fmaf(a.x, bx, sum_x);
                sum_x = fmaf(-a.y, by, sum_x);
                sum_y = fmaf(a.x, by, sum_y);
                sum_y = fmaf(a.y, bx, sum_y);
            }
        }
    }

    if (row < D) {
        y[row] = make_cuFloatComplex(sum_x, sum_y);
    }
}

// out = a + alpha * b
__global__ void vec_lincomb_cuda(const cuDoubleComplex* __restrict__ a,
                                 const cuDoubleComplex* __restrict__ b,
                                 cuDoubleComplex* __restrict__ out,
                                 double alpha,
                                 int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const cuDoubleComplex ai = a[i];
    const cuDoubleComplex bi = b[i];
    out[i] = make_cuDoubleComplex(ai.x + alpha * bi.x, ai.y + alpha * bi.y);
}

__global__ void vec_lincomb_cuda_f32(const cuFloatComplex* __restrict__ a,
                                     const cuFloatComplex* __restrict__ b,
                                     cuFloatComplex* __restrict__ out,
                                     float alpha,
                                     int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const cuFloatComplex ai = a[i];
    const cuFloatComplex bi = b[i];
    out[i] = make_cuFloatComplex(ai.x + alpha * bi.x, ai.y + alpha * bi.y);
}

__global__ void rk4_update_cuda_kernel(cuDoubleComplex* r,
                                       const cuDoubleComplex* k1,
                                       const cuDoubleComplex* k2,
                                       const cuDoubleComplex* k3,
                                       const cuDoubleComplex* k4,
                                       double dt,
                                       int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    const double acc_x = k1[i].x + 2.0 * k2[i].x + 2.0 * k3[i].x + k4[i].x;
    const double acc_y = k1[i].y + 2.0 * k2[i].y + 2.0 * k3[i].y + k4[i].y;
    const double scale = dt / 6.0;
    r[i].x += scale * acc_x;
    r[i].y += scale * acc_y;
}

__global__ void rk4_update_cuda_kernel_f32(cuFloatComplex* r,
                                          const cuFloatComplex* k1,
                                          const cuFloatComplex* k2,
                                          const cuFloatComplex* k3,
                                          const cuFloatComplex* k4,
                                          float dt,
                                          int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    const float acc_x = k1[i].x + 2.0f * k2[i].x + 2.0f * k3[i].x + k4[i].x;
    const float acc_y = k1[i].y + 2.0f * k2[i].y + 2.0f * k3[i].y + k4[i].y;
    const float scale = dt / 6.0f;
    r[i].x += scale * acc_x;
    r[i].y += scale * acc_y;
}

__global__ void half_sum_cuda_kernel(const cuDoubleComplex* A,
                                     const cuDoubleComplex* B,
                                     cuDoubleComplex* out,
                                     std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
    if (i >= n) return;
    const cuDoubleComplex a = A[i];
    const cuDoubleComplex b = B[i];
    out[i] = make_cuDoubleComplex(0.5 * (a.x + b.x), 0.5 * (a.y + b.y));
}

__global__ void half_sum_cuda_kernel_f32(const cuFloatComplex* A,
                                         const cuFloatComplex* B,
                                         cuFloatComplex* out,
                                         std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
    if (i >= n) return;
    const cuFloatComplex a = A[i];
    const cuFloatComplex b = B[i];
    out[i] = make_cuFloatComplex(0.5f * (a.x + b.x), 0.5f * (a.y + b.y));
}

} // namespace

Rk4DenseCudaWorkspace::~Rk4DenseCudaWorkspace() { release(); }

void Rk4DenseCudaWorkspace::release() {
    if (k1) cudaFree(k1);
    if (k2) cudaFree(k2);
    if (k3) cudaFree(k3);
    if (k4) cudaFree(k4);
    if (tmp) cudaFree(tmp);
    if (cublas_handle) {
        (void)cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_handle));
    }
    k1 = k2 = k3 = k4 = tmp = nullptr;
    cublas_handle = nullptr;
    n = 0;
}

void Rk4DenseCudaWorkspace::resize(int n_) {
    if (n_ <= 0) throw std::invalid_argument("Rk4DenseCudaWorkspace::resize: n must be > 0");
    if (n == n_) return;
    release();
    const std::size_t bytes = static_cast<std::size_t>(n_) * sizeof(cuDoubleComplex);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k1), bytes), "cudaMalloc(k1)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k2), bytes), "cudaMalloc(k2)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k3), bytes), "cudaMalloc(k3)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k4), bytes), "cudaMalloc(k4)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&tmp), bytes), "cudaMalloc(tmp)");
    n = n_;
}

Rk4DenseCudaWorkspaceF32::~Rk4DenseCudaWorkspaceF32() { release(); }

void Rk4DenseCudaWorkspaceF32::release() {
    if (k1) cudaFree(k1);
    if (k2) cudaFree(k2);
    if (k3) cudaFree(k3);
    if (k4) cudaFree(k4);
    if (tmp) cudaFree(tmp);
    if (cublas_handle) {
        (void)cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_handle));
    }
    k1 = k2 = k3 = k4 = tmp = nullptr;
    cublas_handle = nullptr;
    n = 0;
}

void Rk4DenseCudaWorkspaceF32::resize(int n_) {
    if (n_ <= 0) throw std::invalid_argument("Rk4DenseCudaWorkspaceF32::resize: n must be > 0");
    if (n == n_) return;
    release();
    const std::size_t bytes = static_cast<std::size_t>(n_) * sizeof(cuFloatComplex);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k1), bytes), "cudaMalloc(k1)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k2), bytes), "cudaMalloc(k2)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k3), bytes), "cudaMalloc(k3)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&k4), bytes), "cudaMalloc(k4)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&tmp), bytes), "cudaMalloc(tmp)");
    n = n_;
}

void half_sum_cuda(const cuDoubleComplex* A,
                   const cuDoubleComplex* B,
                   cuDoubleComplex* out,
                   std::size_t n,
                   cudaStream_t stream) {
    constexpr int block = 256;
    const dim3 grid = grid_1d(n, block);
    half_sum_cuda_kernel<<<grid, block, 0, stream>>>(A, B, out, n);
    cuda_check(cudaGetLastError(), "half_sum_cuda: kernel launch");
}

void half_sum_cuda_f32(const cuFloatComplex* A,
                       const cuFloatComplex* B,
                       cuFloatComplex* out,
                       std::size_t n,
                       cudaStream_t stream) {
    constexpr int block = 256;
    const dim3 grid = grid_1d(n, block);
    half_sum_cuda_kernel_f32<<<grid, block, 0, stream>>>(A, B, out, n);
    cuda_check(cudaGetLastError(), "half_sum_cuda_f32: kernel launch");
}

void rk4_update_cuda(const cuDoubleComplex* L,
                     cuDoubleComplex* r,
                     int D,
                     Rk4DenseCudaWorkspace& ws,
                     double dt,
                     cudaStream_t stream,
                     Rk4DenseCudaMethod method) {
    rk4_update_cuda(L, L, L, r, D, ws, dt, stream, method);
}

void rk4_update_cuda(const cuDoubleComplex* L0,
                     const cuDoubleComplex* Lhalf,
                     const cuDoubleComplex* L1,
                     cuDoubleComplex* r,
                     int D,
                     Rk4DenseCudaWorkspace& ws,
                     double dt,
                     cudaStream_t stream,
                     Rk4DenseCudaMethod method) {
    if (!(dt > 0.0)) throw std::invalid_argument("rk4_update_cuda: dt must be > 0");
    if (D <= 0) throw std::invalid_argument("rk4_update_cuda: D must be > 0");
    ws.resize(D);

    constexpr int vec_block = 256;
    const dim3 grid_vec = grid_1d(static_cast<std::size_t>(D), vec_block);

    constexpr int matvec_block = 32;
    const dim3 grid_matvec = grid_1d(static_cast<std::size_t>(D), matvec_block);

    cublasHandle_t cublas = nullptr;
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        cublas = ensure_cublas_handle(ws, stream);
    } else if (method != Rk4DenseCudaMethod::WarpKernel) {
        throw std::invalid_argument("rk4_update_cuda: unsupported method");
    }

    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas(L0, r, ws.k1, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp<<<grid_matvec, matvec_block, 0, stream>>>(L0, r, ws.k1, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda: matvec(k1)");
    }

    vec_lincomb_cuda<<<grid_vec, vec_block, 0, stream>>>(r, ws.k1, ws.tmp, 0.5 * dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda: tmp=r+0.5*dt*k1");
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas(Lhalf, ws.tmp, ws.k2, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp<<<grid_matvec, matvec_block, 0, stream>>>(Lhalf, ws.tmp, ws.k2, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda: matvec(k2)");
    }

    vec_lincomb_cuda<<<grid_vec, vec_block, 0, stream>>>(r, ws.k2, ws.tmp, 0.5 * dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda: tmp=r+0.5*dt*k2");
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas(Lhalf, ws.tmp, ws.k3, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp<<<grid_matvec, matvec_block, 0, stream>>>(Lhalf, ws.tmp, ws.k3, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda: matvec(k3)");
    }

    vec_lincomb_cuda<<<grid_vec, vec_block, 0, stream>>>(r, ws.k3, ws.tmp, dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda: tmp=r+dt*k3");
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas(L1, ws.tmp, ws.k4, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp<<<grid_matvec, matvec_block, 0, stream>>>(L1, ws.tmp, ws.k4, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda: matvec(k4)");
    }

    rk4_update_cuda_kernel<<<grid_vec, vec_block, 0, stream>>>(r, ws.k1, ws.k2, ws.k3, ws.k4, dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda: update");
}

void rk4_update_cuda_f32(const cuFloatComplex* L,
                         cuFloatComplex* r,
                         int D,
                         Rk4DenseCudaWorkspaceF32& ws,
                         float dt,
                         cudaStream_t stream,
                         Rk4DenseCudaMethod method) {
    rk4_update_cuda_f32(L, L, L, r, D, ws, dt, stream, method);
}

void rk4_update_cuda_f32(const cuFloatComplex* L0,
                         const cuFloatComplex* Lhalf,
                         const cuFloatComplex* L1,
                         cuFloatComplex* r,
                         int D,
                         Rk4DenseCudaWorkspaceF32& ws,
                         float dt,
                         cudaStream_t stream,
                         Rk4DenseCudaMethod method) {
    if (!(dt > 0.0f)) throw std::invalid_argument("rk4_update_cuda_f32: dt must be > 0");
    if (D <= 0) throw std::invalid_argument("rk4_update_cuda_f32: D must be > 0");
    ws.resize(D);

    constexpr int vec_block = 256;
    const dim3 grid_vec = grid_1d(static_cast<std::size_t>(D), vec_block);

    constexpr int matvec_block = 32;
    const dim3 grid_matvec = grid_1d(static_cast<std::size_t>(D), matvec_block);

    cublasHandle_t cublas = nullptr;
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        cublas = ensure_cublas_handle(ws, stream);
    } else if (method != Rk4DenseCudaMethod::WarpKernel) {
        throw std::invalid_argument("rk4_update_cuda_f32: unsupported method");
    }

    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas_f32(L0, r, ws.k1, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp_f32<<<grid_matvec, matvec_block, 0, stream>>>(L0, r, ws.k1, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: matvec(k1)");
    }

    vec_lincomb_cuda_f32<<<grid_vec, vec_block, 0, stream>>>(r, ws.k1, ws.tmp, 0.5f * dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: tmp=r+0.5*dt*k1");
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas_f32(Lhalf, ws.tmp, ws.k2, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp_f32<<<grid_matvec, matvec_block, 0, stream>>>(Lhalf, ws.tmp, ws.k2, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: matvec(k2)");
    }

    vec_lincomb_cuda_f32<<<grid_vec, vec_block, 0, stream>>>(r, ws.k2, ws.tmp, 0.5f * dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: tmp=r+0.5*dt*k2");
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas_f32(Lhalf, ws.tmp, ws.k3, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp_f32<<<grid_matvec, matvec_block, 0, stream>>>(Lhalf, ws.tmp, ws.k3, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: matvec(k3)");
    }

    vec_lincomb_cuda_f32<<<grid_vec, vec_block, 0, stream>>>(r, ws.k3, ws.tmp, dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: tmp=r+dt*k3");
    if (method == Rk4DenseCudaMethod::CublasGemv) {
        matvec_dense_cuda_colmajor_cublas_f32(L1, ws.tmp, ws.k4, D, cublas);
    } else {
        matvec_dense_cuda_colmajor_warp_f32<<<grid_matvec, matvec_block, 0, stream>>>(L1, ws.tmp, ws.k4, D);
        cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: matvec(k4)");
    }

    rk4_update_cuda_kernel_f32<<<grid_vec, vec_block, 0, stream>>>(r, ws.k1, ws.k2, ws.k3, ws.k4, dt, D);
    cuda_check(cudaGetLastError(), "rk4_update_cuda_f32: update");
}

} // namespace taco::tcl

#endif // TACO_HAS_CUDA
