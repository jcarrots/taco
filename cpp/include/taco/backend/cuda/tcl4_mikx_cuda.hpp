#pragma once

#include <cstddef>

#include "taco/exec.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"

#ifdef TACO_HAS_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace taco::tcl4::cuda_mikx {

// Device-side view of triple kernels at a single time index.
//
// Layout:
// - F/C/R are flattened [nf, nf, nf] with linear index
//     idx3(i,j,k) = (i*nf + j)*nf + k
//   so k is the fast/contiguous index.
// - pair_to_freq is a [N, N] matrix in column-major order:
//     pair_to_freq[a + b*N] == map.pair_to_freq(a,b)
struct MikxDeviceInputs {
    const cuDoubleComplex* F{nullptr}; // [nf^3]
    const cuDoubleComplex* C{nullptr}; // [nf^3]
    const cuDoubleComplex* R{nullptr}; // [nf^3]
    const int* pair_to_freq{nullptr};  // [N*N] column-major (a + b*N)
    int N{0};
    int nf{0};
};

// Device-side outputs in the same indexing/layout conventions used by the CPU reference:
// - M/I/K are (N^2 x N^2) matrices in Eigen's default column-major layout.
// - X is a length N^6 vector in column-major flat6 order:
//     flat6(N,j,k,p,q,r,s) = j + N*(k + N*(p + N*(q + N*(r + N*s))))
struct MikxDeviceOutputs {
    cuDoubleComplex* M{nullptr}; // [N^4]
    cuDoubleComplex* I{nullptr}; // [N^4]
    cuDoubleComplex* K{nullptr}; // [N^4]
    cuDoubleComplex* X{nullptr}; // [N^6]
};

// Compute M/I/K/X on the GPU given flattened F/C/R at a single time index.
void build_mikx_device(const MikxDeviceInputs& inputs,
                       const MikxDeviceOutputs& outputs,
                       cudaStream_t stream);

} // namespace taco::tcl4::cuda_mikx

namespace taco::tcl4 {

// Host convenience wrapper:
// - Flattens `kernels.F/C/R` at `time_index` into contiguous buffers.
// - Runs `cuda_mikx::build_mikx_device(...)`.
// - Copies M/I/K/X back and returns a MikxTensors object.
MikxTensors build_mikx_cuda(const Tcl4Map& map,
                            const TripleKernelSeries& kernels,
                            std::size_t time_index,
                            const Exec& exec);

} // namespace taco::tcl4

#endif // TACO_HAS_CUDA

