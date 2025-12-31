#pragma once

#include <cstddef>

#ifdef TACO_HAS_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace taco::tcl4::cuda_fcr {

struct FcrDeviceInputs {
    const cuDoubleComplex* gamma{nullptr}; // [Nt, nf] column-major: gamma[t + Nt*b]
    const double* omegas{nullptr};         // [nf]
    const int* mirror{nullptr};            // [nf]
    std::size_t Nt{0};
    std::size_t nf{0};
    double dt{0.0};
};

struct FcrBatch {
    std::size_t batch{0};
    std::size_t Nfft{0};

    cuDoubleComplex* F{nullptr}; // [batch, Nt]
    cuDoubleComplex* C{nullptr}; // [batch, Nt]
    cuDoubleComplex* R{nullptr}; // [batch, Nt]
};

void launch_scale(cuDoubleComplex* data, std::size_t n, double scale, cudaStream_t stream);
void launch_pointwise_mul(cuDoubleComplex* inout, const cuDoubleComplex* other, std::size_t n, cudaStream_t stream);
void launch_axpby(const cuDoubleComplex* x,
                  const cuDoubleComplex* y,
                  cuDoubleComplex* out,
                  std::size_t n,
                  cuDoubleComplex a,
                  cuDoubleComplex b,
                  cudaStream_t stream);

void compute_fcr_convolution_batched(const FcrDeviceInputs& inputs, const FcrBatch& batch, cudaStream_t stream);

} // namespace taco::tcl4::cuda_fcr
#endif

