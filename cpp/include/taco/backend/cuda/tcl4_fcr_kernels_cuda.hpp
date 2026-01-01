#pragma once

#include <cstddef>

#ifdef TACO_HAS_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
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

    // Batch metadata: this call computes (F,C,R)(i,j,k) for fixed (i,j) and
    // k = k0 + b for b in [0, batch).
    int i{0};
    int j{0};
    int k0{0};

    // Outputs are column-major by time within each batch item: out[t + Nt*b].
    cuDoubleComplex* F{nullptr}; // [Nt, batch]
    cuDoubleComplex* C{nullptr}; // [Nt, batch]
    cuDoubleComplex* R{nullptr}; // [Nt, batch]
};

// Persistent CUDA workspace for the convolution/FFT method.
// Allocate/initialize once (per GPU/stream configuration) and reuse across batches.
struct FcrWorkspace {
    // cuFFT plan for batched 1D transforms of length Nfft, batch=batch.
    cufftHandle plan{0};
    std::size_t plan_batch{0};
    std::size_t plan_Nfft{0};

    // Scratch buffers (padded signals), typically sized [plan_batch * plan_Nfft].
    cuDoubleComplex* A{nullptr};
    cuDoubleComplex* B{nullptr};
    cuDoubleComplex* B_conj{nullptr};

    // CUB temporary storage for scans (e.g., prefix sums); raw byte buffer on device.
    void* scan_tmp{nullptr};
    std::size_t scan_tmp_bytes{0};
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

void compute_fcr_convolution_batched(const FcrDeviceInputs& inputs,
                                     const FcrBatch& batch,
                                     FcrWorkspace& ws,
                                     cudaStream_t stream);

} // namespace taco::tcl4::cuda_fcr
#endif
