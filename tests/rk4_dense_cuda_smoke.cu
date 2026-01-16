#include "taco/backend/cuda/rk4_dense_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace {

inline bool check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(status));
    return false;
}

inline cuDoubleComplex to_cu(std::complex<double> z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}

inline cuFloatComplex to_cu_f32(std::complex<double> z) {
    return make_cuFloatComplex(static_cast<float>(z.real()), static_cast<float>(z.imag()));
}

} // namespace

int main() {
#if !defined(TACO_HAS_CUDA)
    std::puts("rk4_dense_cuda_smoke: skipped (TACO_HAS_CUDA not set)");
    return 0;
#else
    // Simple diagonal system r' = L r where L is diagonal with negative real entries:
    // exact solution: r_i(t) = exp(lambda_i t) * r_i(0)
    constexpr int D = 1024;
    constexpr double dt = 1e-3;
    constexpr int nsteps = 1000;
    constexpr double T = dt * nsteps;

    std::vector<std::complex<double>> hL(static_cast<std::size_t>(D) * static_cast<std::size_t>(D), {0.0, 0.0});
    std::vector<std::complex<double>> hr(D, {1.0, 0.0});

    for (int i = 0; i < D; ++i) {
        const double lambda_i = -1.0 - 0.1 * static_cast<double>(i);
        hL[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(D)] = {lambda_i, 0.0};
    }

    const std::size_t vbytes = static_cast<std::size_t>(D) * sizeof(cuDoubleComplex);
    const std::size_t Lbytes = static_cast<std::size_t>(D) * static_cast<std::size_t>(D) * sizeof(cuDoubleComplex);

    // Pack host buffers into cuDoubleComplex for device copies.
    std::vector<cuDoubleComplex> hL_cu(hL.size());
    for (std::size_t i = 0; i < hL.size(); ++i) hL_cu[i] = to_cu(hL[i]);
    std::vector<cuDoubleComplex> hr_cu(static_cast<std::size_t>(D));
    for (int i = 0; i < D; ++i) hr_cu[static_cast<std::size_t>(i)] = to_cu(hr[static_cast<std::size_t>(i)]);

    cuDoubleComplex* dL = nullptr;
    cuDoubleComplex* dr = nullptr;
    if (!check(cudaMalloc(&dL, Lbytes), "cudaMalloc(L)")) return 1;
    if (!check(cudaMalloc(&dr, vbytes), "cudaMalloc(r)")) return 1;

    if (!check(cudaMemcpy(dL, hL_cu.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L)")) return 1;
    if (!check(cudaMemcpy(dr, hr_cu.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r)")) return 1;

    const std::vector<cuDoubleComplex> hr0_cu = hr_cu;

    auto run_one = [&](taco::tcl::Rk4DenseCudaMethod method, const char* name) -> double {
        if (!check(cudaMemcpy(dr, hr0_cu.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r reset)")) return -1.0;

        taco::tcl::Rk4DenseCudaWorkspace ws;
        for (int step = 0; step < nsteps; ++step) {
            taco::tcl::rk4_update_cuda(dL, dr, D, ws, dt, /*stream=*/0, method);
        }

        if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return -1.0;
        if (!check(cudaMemcpy(hr_cu.data(), dr, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r back)")) return -1.0;

        for (int i = 0; i < D; ++i) {
            hr[static_cast<std::size_t>(i)] = std::complex<double>(hr_cu[static_cast<std::size_t>(i)].x,
                                                                   hr_cu[static_cast<std::size_t>(i)].y);
        }

        double err2 = 0.0;
        for (int i = 0; i < D; ++i) {
            const double lambda_i = -1.0 - 0.1 * static_cast<double>(i);
            const double exact = std::exp(lambda_i * T);
            const double diff = hr[static_cast<std::size_t>(i)].real() - exact;
            err2 += diff * diff;
        }

        const double l2err = std::sqrt(err2);
        std::printf("rk4_dense_cuda_smoke (%s): D=%d steps=%d dt=%.3g L2_error=%e\n", name, D, nsteps, dt, l2err);
        return l2err;
    };

    const double err_warp = run_one(taco::tcl::Rk4DenseCudaMethod::WarpKernel, "warp");
    if (!std::isfinite(err_warp) || err_warp > 1e-6) {
        std::fprintf(stderr, "rk4_dense_cuda_smoke: warp error too large\n");
        return 2;
    }

    const double err_cublas = run_one(taco::tcl::Rk4DenseCudaMethod::CublasGemv, "cublas");
    if (!std::isfinite(err_cublas) || err_cublas > 1e-6) {
        std::fprintf(stderr, "rk4_dense_cuda_smoke: cublas error too large\n");
        return 3;
    }

    // ------------------------------- FP32 path -------------------------------
    std::vector<cuFloatComplex> hL_f32(hL.size());
    for (std::size_t i = 0; i < hL.size(); ++i) hL_f32[i] = to_cu_f32(hL[i]);
    std::vector<cuFloatComplex> hr_f32(static_cast<std::size_t>(D));
    for (int i = 0; i < D; ++i) hr_f32[static_cast<std::size_t>(i)] = make_cuFloatComplex(1.0f, 0.0f);
    const std::vector<cuFloatComplex> hr0_f32 = hr_f32;

    const std::size_t vbytes_f32 = static_cast<std::size_t>(D) * sizeof(cuFloatComplex);
    const std::size_t Lbytes_f32 = static_cast<std::size_t>(D) * static_cast<std::size_t>(D) * sizeof(cuFloatComplex);

    cuFloatComplex* dL_f32 = nullptr;
    cuFloatComplex* dr_f32 = nullptr;
    if (!check(cudaMalloc(&dL_f32, Lbytes_f32), "cudaMalloc(L_f32)")) return 4;
    if (!check(cudaMalloc(&dr_f32, vbytes_f32), "cudaMalloc(r_f32)")) return 4;

    if (!check(cudaMemcpy(dL_f32, hL_f32.data(), Lbytes_f32, cudaMemcpyHostToDevice), "cudaMemcpy(L_f32)")) return 4;

    auto run_one_f32 = [&](taco::tcl::Rk4DenseCudaMethod method, const char* name) -> double {
        if (!check(cudaMemcpy(dr_f32, hr0_f32.data(), vbytes_f32, cudaMemcpyHostToDevice), "cudaMemcpy(r_f32 reset)")) return -1.0;

        taco::tcl::Rk4DenseCudaWorkspaceF32 ws;
        for (int step = 0; step < nsteps; ++step) {
            taco::tcl::rk4_update_cuda_f32(dL_f32, dr_f32, D, ws, static_cast<float>(dt), /*stream=*/0, method);
        }

        if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(f32)")) return -1.0;
        if (!check(cudaMemcpy(hr_f32.data(), dr_f32, vbytes_f32, cudaMemcpyDeviceToHost), "cudaMemcpy(r_f32 back)")) return -1.0;

        double err2 = 0.0;
        for (int i = 0; i < D; ++i) {
            const double lambda_i = -1.0 - 0.1 * static_cast<double>(i);
            const double exact = std::exp(lambda_i * T);
            const double diff = static_cast<double>(hr_f32[static_cast<std::size_t>(i)].x) - exact;
            err2 += diff * diff;
        }
        const double l2err = std::sqrt(err2);
        std::printf("rk4_dense_cuda_smoke (%s,f32): D=%d steps=%d dt=%.3g L2_error=%e\n", name, D, nsteps, dt, l2err);
        return l2err;
    };

    const double err_warp_f32 = run_one_f32(taco::tcl::Rk4DenseCudaMethod::WarpKernel, "warp");
    if (!std::isfinite(err_warp_f32) || err_warp_f32 > 1e-4) {
        std::fprintf(stderr, "rk4_dense_cuda_smoke: warp f32 error too large\n");
        return 5;
    }

    const double err_cublas_f32 = run_one_f32(taco::tcl::Rk4DenseCudaMethod::CublasGemv, "cublas");
    if (!std::isfinite(err_cublas_f32) || err_cublas_f32 > 1e-4) {
        std::fprintf(stderr, "rk4_dense_cuda_smoke: cublas f32 error too large\n");
        return 6;
    }

    cudaFree(dL_f32);
    cudaFree(dr_f32);

    cudaFree(dL);
    cudaFree(dr);
    return 0;
#endif
}
