#include "taco/backend/cuda/tcl4_fcr_kernels_cuda.hpp"

#include <stdexcept>
#include <string>

namespace taco::tcl4::cuda_fcr {

namespace {

inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

__device__ __forceinline__ cuDoubleComplex cd_add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cuDoubleComplex cd_mul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuDoubleComplex cd_scale(cuDoubleComplex a, double s) {
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__global__ void kernel_scale(cuDoubleComplex* data, std::size_t n, double scale) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) data[i] = cd_scale(data[i], scale);
}

__global__ void kernel_pointwise_mul(cuDoubleComplex* inout,
                                     const cuDoubleComplex* other,
                                     std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) inout[i] = cd_mul(inout[i], other[i]);
}

__global__ void kernel_axpby(const cuDoubleComplex* x,
                             const cuDoubleComplex* y,
                             cuDoubleComplex* out,
                             std::size_t n,
                             cuDoubleComplex a,
                             cuDoubleComplex b)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cd_add(cd_mul(a, x[i]), cd_mul(b, y[i]));
}

inline dim3 grid_1d(std::size_t n, int block) {
    return dim3(static_cast<unsigned>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block)));
}

} // namespace

void launch_scale(cuDoubleComplex* data,
                  std::size_t n,
                  double scale,
                  cudaStream_t stream)
{
    constexpr int block = 256;
    kernel_scale<<<grid_1d(n, block), block, 0, stream>>>(data, n, scale);
    cuda_check(cudaGetLastError(), "kernel_scale launch");
}

void launch_pointwise_mul(cuDoubleComplex* inout,
                          const cuDoubleComplex* other,
                          std::size_t n,
                          cudaStream_t stream)
{
    constexpr int block = 256;
    kernel_pointwise_mul<<<grid_1d(n, block), block, 0, stream>>>(inout, other, n);
    cuda_check(cudaGetLastError(), "kernel_pointwise_mul launch");
}

void launch_axpby(const cuDoubleComplex* x,
                  const cuDoubleComplex* y,
                  cuDoubleComplex* out,
                  std::size_t n,
                  cuDoubleComplex a,
                  cuDoubleComplex b,
                  cudaStream_t stream)
{
    constexpr int block = 256;
    kernel_axpby<<<grid_1d(n, block), block, 0, stream>>>(x, y, out, n, a, b);
    cuda_check(cudaGetLastError(), "kernel_axpby launch");
}

void compute_fcr_convolution_batched(const FcrDeviceInputs& /*inputs*/,
                                     const FcrBatch& /*batch*/,
                                     cudaStream_t /*stream*/)
{
    throw std::runtime_error("compute_fcr_convolution_batched: not implemented");
}

} // namespace taco::tcl4::cuda_fcr

