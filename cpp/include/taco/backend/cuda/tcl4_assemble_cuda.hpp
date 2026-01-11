#pragma once

#include <vector>

#include <Eigen/Dense>

#include "taco/exec.hpp"
#include "taco/tcl4_mikx.hpp"

#ifdef TACO_HAS_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#endif

namespace taco::tcl4 {

// CUDA implementation of assemble_liouvillian (GW raw matrix).
Eigen::MatrixXcd assemble_liouvillian_cuda(const MikxTensors& tensors,
                                           const std::vector<Eigen::MatrixXcd>& coupling_ops,
                                           const Exec& exec);

#ifdef TACO_HAS_CUDA
// Device-side helper: assemble GW directly on the GPU without host transfers.
void assemble_liouvillian_cuda_device(const cuDoubleComplex* dM,
                                      const cuDoubleComplex* dI,
                                      const cuDoubleComplex* dK,
                                      const cuDoubleComplex* dX,
                                      const cuDoubleComplex* d_ops,
                                      int N,
                                      int num_ops,
                                      cuDoubleComplex* dGW,
                                      cudaStream_t stream);

// Device-side helper: assemble the raw (unsymmetrized) GW matrix directly on the GPU.
// The fused TCL4 CUDA path can then symmetrize + permute into L4 in a single kernel.
void assemble_liouvillian_cuda_device_raw(const cuDoubleComplex* dM,
                                          const cuDoubleComplex* dI,
                                          const cuDoubleComplex* dK,
                                          const cuDoubleComplex* dX,
                                          const cuDoubleComplex* d_ops,
                                          int N,
                                          int num_ops,
                                          cuDoubleComplex* dGW_raw,
                                          cudaStream_t stream);
#endif

} // namespace taco::tcl4
