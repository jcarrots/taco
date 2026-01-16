#pragma once

#include <cstddef>

#include "taco/exec.hpp"

#ifdef TACO_HAS_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace taco::tcl {

// Dense RK4 integrator on CUDA for complex vectors:
//   r' = L(t) r
//
// Notes:
// - This is intended for *small* dense systems (L is D-by-D). Memory is O(D^2).
// - Matrices are expected in Eigen's default column-major layout (same as Eigen::MatrixXcd::data()).

struct Rk4DenseCudaWorkspace {
    cuDoubleComplex* k1{nullptr};
    cuDoubleComplex* k2{nullptr};
    cuDoubleComplex* k3{nullptr};
    cuDoubleComplex* k4{nullptr};
    cuDoubleComplex* tmp{nullptr};
    void* cublas_handle{nullptr}; // opaque (cublasHandle_t), created lazily when needed
    int n{0};

    void resize(int n_);
    void release();
    ~Rk4DenseCudaWorkspace();
};

// FP32 variant (cuFloatComplex).
struct Rk4DenseCudaWorkspaceF32 {
    cuFloatComplex* k1{nullptr};
    cuFloatComplex* k2{nullptr};
    cuFloatComplex* k3{nullptr};
    cuFloatComplex* k4{nullptr};
    cuFloatComplex* tmp{nullptr};
    void* cublas_handle{nullptr}; // opaque (cublasHandle_t), created lazily when needed
    int n{0};

    void resize(int n_);
    void release();
    ~Rk4DenseCudaWorkspaceF32();
};

enum class Rk4DenseCudaMethod {
    WarpKernel, // custom CUDA kernel matvec (good for small/medium dense systems)
    CublasGemv, // cuBLAS ZGEMV matvec (often better for larger dense systems)
};

// Compute out = 0.5 * (A + B) elementwise for n complex entries.
// Intended helper for building L(t+dt/2) from endpoint matrices.
void half_sum_cuda(const cuDoubleComplex* A,
                   const cuDoubleComplex* B,
                   cuDoubleComplex* out,
                   std::size_t n,
                   cudaStream_t stream);

// One RK4 update for r' = L r with a constant L (uses the same L for all stages).
void rk4_update_cuda(const cuDoubleComplex* L,
                     cuDoubleComplex* r,
                     int D,
                     Rk4DenseCudaWorkspace& ws,
                     double dt,
                     cudaStream_t stream,
                     Rk4DenseCudaMethod method = Rk4DenseCudaMethod::WarpKernel);

// One RK4 update for r' = L(t) r with stage matrices at t, t+dt/2, t+dt.
void rk4_update_cuda(const cuDoubleComplex* L0,
                     const cuDoubleComplex* Lhalf,
                     const cuDoubleComplex* L1,
                     cuDoubleComplex* r,
                     int D,
                     Rk4DenseCudaWorkspace& ws,
                     double dt,
                     cudaStream_t stream,
                     Rk4DenseCudaMethod method = Rk4DenseCudaMethod::WarpKernel);

// ------------------------------- FP32 API -------------------------------

void half_sum_cuda_f32(const cuFloatComplex* A,
                       const cuFloatComplex* B,
                       cuFloatComplex* out,
                       std::size_t n,
                       cudaStream_t stream);

void rk4_update_cuda_f32(const cuFloatComplex* L,
                         cuFloatComplex* r,
                         int D,
                         Rk4DenseCudaWorkspaceF32& ws,
                         float dt,
                         cudaStream_t stream,
                         Rk4DenseCudaMethod method = Rk4DenseCudaMethod::WarpKernel);

void rk4_update_cuda_f32(const cuFloatComplex* L0,
                         const cuFloatComplex* Lhalf,
                         const cuFloatComplex* L1,
                         cuFloatComplex* r,
                         int D,
                         Rk4DenseCudaWorkspaceF32& ws,
                         float dt,
                         cudaStream_t stream,
                         Rk4DenseCudaMethod method = Rk4DenseCudaMethod::WarpKernel);

} // namespace taco::tcl

#endif // TACO_HAS_CUDA
