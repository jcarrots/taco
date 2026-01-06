#include "taco/backend/cuda/tcl4_fcr_kernels_cuda.hpp"
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace taco::tcl4::cuda_fcr {

// This file implements the CUDA backend for the scalar (1×1) F/C/R series formulas,
// batched over k for fixed (i,j). See `FcrBatch`/`FcrDeviceInputs` for the indexing conventions.
//
// Layout conventions:
// - Scratch FFT buffers `A`/`B` are stored as [Nfft, batch] with `A[b*Nfft + t]`.
// - Output buffers `F/C/R` are stored as [Nt, batch] with `out[t + Nt*b]`.
//
// Kernel launch mapping:
// - We use a 2D grid where blockIdx.y selects the batch lane b, and x-dimension tiles time t.
//   This matches the output layout (time is contiguous), giving coalesced accesses.

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

__device__ __forceinline__ cuDoubleComplex cd_conj(cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x, -a.y);
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

inline dim3 grid_2d_tiled(std::size_t n, std::size_t batch, int block_t, int block_b) {
    return dim3(static_cast<unsigned>((n + static_cast<std::size_t>(block_t) - 1) / static_cast<std::size_t>(block_t)),
                static_cast<unsigned>((batch + static_cast<std::size_t>(block_b) - 1) / static_cast<std::size_t>(block_b)));
}

constexpr int kDefaultBlockT = 256;
constexpr int kDefaultBlockB = 2;

inline int round_up_warp(int v) {
    return ((v + 31) / 32) * 32;
}

inline int read_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) return fallback;
    if (parsed > std::numeric_limits<int>::max() || parsed < std::numeric_limits<int>::min()) return fallback;
    return static_cast<int>(parsed);
}

inline int clamp_block_t(int v) {
    if (v <= 0) v = kDefaultBlockT;
    if (v > 1024) v = 1024;
    v = round_up_warp(v);
    if (v > 1024) v = 1024;
    if (v < 32) v = 32;
    return v;
}

inline int clamp_block_b(int v, int block_t) {
    if (v <= 0) v = kDefaultBlockB;
    if (v < 1) v = 1;
    const int max_b = std::max(1, 1024 / block_t);
    if (v > max_b) v = max_b;
    return v;
}

struct FcrBlockConfig {
    int block_t;
    int block_b;
    std::size_t smem1;
    std::size_t smem2;
};

const FcrBlockConfig& get_fcr_block_config() {
    static const FcrBlockConfig cfg = []() {
        int block_t = clamp_block_t(read_env_int("TACO_CUDA_FCR_BLOCK_T", kDefaultBlockT));
        int block_b = clamp_block_b(read_env_int("TACO_CUDA_FCR_BLOCK_B", kDefaultBlockB), block_t);
        FcrBlockConfig out;
        out.block_t = block_t;
        out.block_b = block_b;
        out.smem1 = static_cast<std::size_t>(block_t) * sizeof(cuDoubleComplex);
        out.smem2 = static_cast<std::size_t>(block_t) * 2 * sizeof(cuDoubleComplex);
        return out;
    }();
    return cfg;
}

inline bool fcr_profile_enabled() {
    static const bool enabled = (read_env_int("TACO_CUDA_FCR_PROFILE", 0) != 0);
    return enabled;
}

inline void cufft_check(cufftResult status, const char* what) {
    if (status == CUFFT_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": cuFFT error code " + std::to_string(static_cast<int>(status)));
}

// Pack A/B for the F convolution term:
//   A(t) = g1(t) * exp(-i ω t)    (with ω depending on lane b)
//   B(t) = g2_mirror(t)          (mirror applied to j)
// Both are zero-padded to length Nfft; A/B are stored as [Nfft, batch].
__global__ void kernel_pack_conv_F(cuDoubleComplex* A,
                                   cuDoubleComplex* B,
                                   const cuDoubleComplex* gamma,
                                   const double* omegas,
                                   const int* mirror,
                                   int i,
                                   int j,
                                   int k0,
                                   int batch,
                                   std::size_t Nt,
                                   std::size_t Nfft,
                                   double dt)
{
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t b = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const bool valid_t = (t < Nfft);
    const bool valid_g = (t < Nt);
    const bool valid_b = (b < static_cast<std::size_t>(batch));

    extern __shared__ cuDoubleComplex smem[];
    cuDoubleComplex* g1_sh = smem;
    cuDoubleComplex* g2_sh = smem + blockDim.x;
    if (threadIdx.y == 0) {
        if (valid_g) {
            const std::size_t t_idx = t;
            g1_sh[threadIdx.x] = gamma[t_idx + Nt * static_cast<std::size_t>(i)];
            int jm = mirror[j];
            if (jm < 0) jm = j;
            g2_sh[threadIdx.x] = gamma[t_idx + Nt * static_cast<std::size_t>(jm)];
        } else {
            g1_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
            g2_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();
    if (!valid_t || !valid_b) return;

    const std::size_t base = b * Nfft;
    if (!valid_g) {
        A[base + t] = make_cuDoubleComplex(0.0, 0.0);
        B[base + t] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    const int k = k0 + static_cast<int>(b);
    const double omega = omegas[i] + omegas[j] + omegas[k];
    const double theta = omega * dt * static_cast<double>(t);
    double s = 0.0, c = 1.0;
    sincos(theta, &s, &c);
    const cuDoubleComplex phase_minus = make_cuDoubleComplex(c, -s);

    const cuDoubleComplex g1  = g1_sh[threadIdx.x];
    const cuDoubleComplex g2m = g2_sh[threadIdx.x];

    A[base + t] = cd_mul(g1, phase_minus);
    B[base + t] = g2m;
}

// Pack A/B for the C convolution term:
//   A(t) = g1(t) * exp(-i ω t)
//   B(t) = conj(g2(t))
// Both are zero-padded to length Nfft.
__global__ void kernel_pack_conv_C(cuDoubleComplex* A,
                                   cuDoubleComplex* B,
                                   const cuDoubleComplex* gamma,
                                   const double* omegas,
                                   int i,
                                   int j,
                                   int k0,
                                   int batch,
                                   std::size_t Nt,
                                   std::size_t Nfft,
                                   double dt)
{
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t b = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const bool valid_t = (t < Nfft);
    const bool valid_g = (t < Nt);
    const bool valid_b = (b < static_cast<std::size_t>(batch));

    extern __shared__ cuDoubleComplex smem[];
    cuDoubleComplex* g1_sh = smem;
    cuDoubleComplex* g2c_sh = smem + blockDim.x;
    if (threadIdx.y == 0) {
        if (valid_g) {
            const std::size_t t_idx = t;
            const cuDoubleComplex g2 = gamma[t_idx + Nt * static_cast<std::size_t>(j)];
            g1_sh[threadIdx.x] = gamma[t_idx + Nt * static_cast<std::size_t>(i)];
            g2c_sh[threadIdx.x] = cd_conj(g2);
        } else {
            g1_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
            g2c_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();
    if (!valid_t || !valid_b) return;

    const std::size_t base = b * Nfft;
    if (!valid_g) {
        A[base + t] = make_cuDoubleComplex(0.0, 0.0);
        B[base + t] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    const int k = k0 + static_cast<int>(b);
    const double omega = omegas[i] + omegas[j] + omegas[k];
    const double theta = omega * dt * static_cast<double>(t);
    double s = 0.0, c = 1.0;
    sincos(theta, &s, &c);
    const cuDoubleComplex phase_minus = make_cuDoubleComplex(c, -s);

    const cuDoubleComplex g1 = g1_sh[threadIdx.x];
    const cuDoubleComplex g2c = g2c_sh[threadIdx.x];

    A[base + t] = cd_mul(g1, phase_minus);
    B[base + t] = g2c;
}

// Pack the integrand for the F prefix term:
//   integrand(t) = dt * g2_mirror(t) * exp(+i ω t)
// This gets scanned (cumulative sum) over t to form the left Riemann prefix integral.
__global__ void kernel_pack_prefix_F(cuDoubleComplex* out, // [Nt, batch]
                                     const cuDoubleComplex* gamma,
                                     const double* omegas,
                                     const int* mirror,
                                     int i,
                                     int j,
                                     int k0,
                                     int batch,
                                     std::size_t Nt,
                                     double dt)
{
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t b = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const bool valid_t = (t < Nt);
    const bool valid_b = (b < static_cast<std::size_t>(batch));

    extern __shared__ cuDoubleComplex smem[];
    cuDoubleComplex* g2m_sh = smem;
    if (threadIdx.y == 0) {
        if (valid_t) {
            int jm = mirror[j];
            if (jm < 0) jm = j;
            g2m_sh[threadIdx.x] = gamma[t + Nt * static_cast<std::size_t>(jm)];
        } else {
            g2m_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();
    if (!valid_t || !valid_b) return;

    const int k = k0 + static_cast<int>(b);
    const double omega = omegas[i] + omegas[j] + omegas[k];
    const double theta = omega * dt * static_cast<double>(t);
    double s = 0.0, c = 1.0;
    sincos(theta, &s, &c);
    const cuDoubleComplex phase_plus = make_cuDoubleComplex(c, s);

    const cuDoubleComplex g2m = g2m_sh[threadIdx.x];
    out[t + Nt * b] = cd_scale(cd_mul(g2m, phase_plus), dt);
}

// Pack the integrand for the C prefix term:
//   integrand(t) = dt * conj(g2(t)) * exp(+i ω t)
__global__ void kernel_pack_prefix_C(cuDoubleComplex* out, // [Nt, batch]
                                     const cuDoubleComplex* gamma,
                                     const double* omegas,
                                     int i,
                                     int j,
                                     int k0,
                                     int batch,
                                     std::size_t Nt,
                                     double dt)
{
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t b = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const bool valid_t = (t < Nt);
    const bool valid_b = (b < static_cast<std::size_t>(batch));

    extern __shared__ cuDoubleComplex smem[];
    cuDoubleComplex* g2c_sh = smem;
    if (threadIdx.y == 0) {
        if (valid_t) {
            const cuDoubleComplex g2 = gamma[t + Nt * static_cast<std::size_t>(j)];
            g2c_sh[threadIdx.x] = cd_conj(g2);
        } else {
            g2c_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();
    if (!valid_t || !valid_b) return;

    const int k = k0 + static_cast<int>(b);
    const double omega = omegas[i] + omegas[j] + omegas[k];
    const double theta = omega * dt * static_cast<double>(t);
    double s = 0.0, c = 1.0;
    sincos(theta, &s, &c);
    const cuDoubleComplex phase_plus = make_cuDoubleComplex(c, s);

    const cuDoubleComplex g2c = g2c_sh[threadIdx.x];
    out[t + Nt * b] = cd_scale(cd_mul(g2c, phase_plus), dt);
}

// Final assembly for F or C (same math, different prefix + conv input choices):
//   term1(t) = (g1(t) * exp(-i ω t)) * prefix(t)
//   out(t)   = term1(t) - conv(t)
// Here `out` initially contains prefix(t) (after the scan), and `conv` is the FFT-based
// causal convolution result scaled by dt (see `convolution_cufft`).
__global__ void kernel_assemble_FC(cuDoubleComplex* out, // [Nt, batch] prefix in, result out
                                   const cuDoubleComplex* conv, // [Nfft, batch]
                                   const cuDoubleComplex* gamma,
                                   const double* omegas,
                                   int i,
                                   int j,
                                   int k0,
                                   int batch,
                                   std::size_t Nt,
                                   std::size_t Nfft,
                                   double dt)
{
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t b = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const bool valid_t = (t < Nt);
    const bool valid_b = (b < static_cast<std::size_t>(batch));

    extern __shared__ cuDoubleComplex smem[];
    cuDoubleComplex* g1_sh = smem;
    if (threadIdx.y == 0) {
        if (valid_t) {
            g1_sh[threadIdx.x] = gamma[t + Nt * static_cast<std::size_t>(i)];
        } else {
            g1_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();
    if (!valid_t || !valid_b) return;

    const int k = k0 + static_cast<int>(b);
    const double omega = omegas[i] + omegas[j] + omegas[k];
    const double theta = omega * dt * static_cast<double>(t);
    double s = 0.0, c = 1.0;
    sincos(theta, &s, &c);
    const cuDoubleComplex phase_minus = make_cuDoubleComplex(c, -s);

    const cuDoubleComplex g1 = g1_sh[threadIdx.x];
    const cuDoubleComplex g1_phase = cd_mul(g1, phase_minus);
    const cuDoubleComplex prefix = out[t + Nt * b];
    const cuDoubleComplex term1 = cd_mul(g1_phase, prefix);
    const cuDoubleComplex term2 = conv[t + Nfft * b];
    out[t + Nt * b] = make_cuDoubleComplex(term1.x - term2.x, term1.y - term2.y);
}

// Pack integrands for the R series (no convolution in R):
//   prefix_g2(t) = dt * cumsum( g2(t) * exp(-i ω t) )
//   prefix_P(t)  = dt * cumsum( (g1(t)*g2(t)) * exp(-i ω t) )
__global__ void kernel_pack_R_integrands(cuDoubleComplex* out_g2, // [Nt, batch]
                                        cuDoubleComplex* out_P,  // [Nt, batch]
                                        const cuDoubleComplex* gamma,
                                        const double* omegas,
                                        int i,
                                        int j,
                                        int k0,
                                        int batch,
                                        std::size_t Nt,
                                        double dt)
{
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t b = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const bool valid_t = (t < Nt);
    const bool valid_b = (b < static_cast<std::size_t>(batch));

    extern __shared__ cuDoubleComplex smem[];
    cuDoubleComplex* g1_sh = smem;
    cuDoubleComplex* g2_sh = smem + blockDim.x;
    if (threadIdx.y == 0) {
        if (valid_t) {
            g1_sh[threadIdx.x] = gamma[t + Nt * static_cast<std::size_t>(i)];
            g2_sh[threadIdx.x] = gamma[t + Nt * static_cast<std::size_t>(j)];
        } else {
            g1_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
            g2_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();
    if (!valid_t || !valid_b) return;

    const int k = k0 + static_cast<int>(b);
    const double omega = omegas[i] + omegas[j] + omegas[k];
    const double theta = omega * dt * static_cast<double>(t);
    double s = 0.0, c = 1.0;
    sincos(theta, &s, &c);
    const cuDoubleComplex phase_minus = make_cuDoubleComplex(c, -s);

    const cuDoubleComplex g1 = g1_sh[threadIdx.x];
    const cuDoubleComplex g2 = g2_sh[threadIdx.x];

    const std::size_t idx = t + Nt * b;
    out_g2[idx] = cd_scale(cd_mul(g2, phase_minus), dt);
    out_P[idx]  = cd_scale(cd_mul(cd_mul(g1, g2), phase_minus), dt);
}

// Final assembly for R:
//   R(t) = g1(t) * prefix_g2(t) - prefix_P(t)
// Here `out_R` initially contains prefix_g2(t), and `prefix_P` is provided separately.
__global__ void kernel_assemble_R(cuDoubleComplex* out_R, // [Nt, batch] prefix_g2 in, R out
                                  const cuDoubleComplex* prefix_P, // [Nt, batch]
                                  const cuDoubleComplex* gamma,
                                  int i,
                                  int batch,
                                  std::size_t Nt)
{
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t b = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const bool valid_t = (t < Nt);
    const bool valid_b = (b < static_cast<std::size_t>(batch));

    extern __shared__ cuDoubleComplex smem[];
    cuDoubleComplex* g1_sh = smem;
    if (threadIdx.y == 0) {
        if (valid_t) {
            g1_sh[threadIdx.x] = gamma[t + Nt * static_cast<std::size_t>(i)];
        } else {
            g1_sh[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();
    if (!valid_t || !valid_b) return;

    const std::size_t idx = t + Nt * b;
    const cuDoubleComplex g1 = g1_sh[threadIdx.x];
    const cuDoubleComplex term1 = cd_mul(g1, out_R[idx]);
    const cuDoubleComplex term2 = prefix_P[idx];
    out_R[idx] = make_cuDoubleComplex(term1.x - term2.x, term1.y - term2.y);
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


//define the sum for cuDoubleComplex type for prefix sum
struct Cadd {
  __host__ __device__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) const {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
  }
};

void prefix_sum_cuda(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int n,
                     void*& d_tmp, size_t& tmp_bytes, cudaStream_t stream)
{
  size_t need = 0;
  cub::DeviceScan::InclusiveScan(nullptr, need, d_in, d_out, Cadd{}, n, stream);
  if (need > tmp_bytes) {
    if (d_tmp) cuda_check(cudaFree(d_tmp), "cudaFree(scan_tmp)");
    cuda_check(cudaMalloc(&d_tmp, need), "cudaMalloc(scan_tmp)");
    tmp_bytes = need;
  }
  cub::DeviceScan::InclusiveScan(d_tmp, tmp_bytes, d_in, d_out, Cadd{}, n, stream);
}




void convolution_cufft(cufftHandle plan,
                       cuDoubleComplex* A,  // [batch * Nfft] in/out      
                       cuDoubleComplex* B,  // [batch * Nfft] in/out
                       std::size_t batch,
                       std::size_t Nfft,
                       double dt,
                       cudaStream_t stream)
{   std::size_t n=batch*Nfft;
    cufftSetStream(plan, stream); 
    auto* Ac = reinterpret_cast<cufftDoubleComplex*>(A);
    auto* Bc = reinterpret_cast<cufftDoubleComplex*>(B);

    // Batched FFT convolution: for each lane b, do
    //   A_b = FFT(A_b), B_b = FFT(B_b), A_b *= B_b, A_b = IFFT(A_b)
    // cuFFT's inverse is unnormalized, so we scale by (dt / Nfft) to match CPU `causal_conv_fft`,
    // which returns dt * ifft(fft(A)*fft(B)) with an internally normalized inverse.
    cufftExecZ2Z(plan, Ac, Ac, CUFFT_FORWARD);
    cufftExecZ2Z(plan, Bc, Bc, CUFFT_FORWARD);

    launch_pointwise_mul(A,B,n,stream);
    cufftExecZ2Z(plan, Ac, Ac, CUFFT_INVERSE);
    launch_scale(A, n, dt / static_cast<double>(Nfft), stream);
}



// host function (no __global__)
void compute_fcr_convolution_batched(const FcrDeviceInputs& inputs,
                                     const FcrBatch& batch,
                                     FcrWorkspace& ws,
                                     cudaStream_t stream)
{
    const std::size_t Nt   = inputs.Nt;
    const std::size_t nf   = inputs.nf;
    const std::size_t B    = batch.batch;
    const std::size_t Nfft = batch.Nfft;

    if (B == 0 || Nt == 0) return;
    if (!inputs.gamma || !inputs.omegas || !inputs.mirror) throw std::runtime_error("compute_fcr_convolution_batched: null inputs");
    if (!batch.F || !batch.C || !batch.R) throw std::runtime_error("compute_fcr_convolution_batched: null outputs");
    if (Nfft < 2 * Nt - 1) throw std::runtime_error("compute_fcr_convolution_batched: Nfft must be >= 2*Nt-1");
    if (!(inputs.dt > 0.0)) throw std::runtime_error("compute_fcr_convolution_batched: dt must be > 0");

    if (batch.i < 0 || batch.j < 0 || batch.k0 < 0) throw std::runtime_error("compute_fcr_convolution_batched: negative index");
    if (static_cast<std::size_t>(batch.i) >= nf || static_cast<std::size_t>(batch.j) >= nf) {
        throw std::runtime_error("compute_fcr_convolution_batched: i/j out of range");
    }
    if (static_cast<std::size_t>(batch.k0) + B > nf) {
        throw std::runtime_error("compute_fcr_convolution_batched: k0+batch out of range");
    }
    if (Nt > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("compute_fcr_convolution_batched: Nt too large for CUB DeviceScan");
    }
    if (B > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        Nfft > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("compute_fcr_convolution_batched: batch/Nfft too large for cuFFT plan");
    }

    // ---- ensure plan + scratch ----
    if (ws.plan != 0 && (ws.plan_batch != B || ws.plan_Nfft != Nfft)) {
        cufft_check(cufftDestroy(ws.plan), "cufftDestroy(plan)");
        ws.plan = 0;
        ws.plan_batch = 0;
        ws.plan_Nfft = 0;
        if (ws.A) cuda_check(cudaFree(ws.A), "cudaFree(ws.A)");
        if (ws.B) cuda_check(cudaFree(ws.B), "cudaFree(ws.B)");
        if (ws.B_conj) cuda_check(cudaFree(ws.B_conj), "cudaFree(ws.B_conj)");
        ws.A = nullptr;
        ws.B = nullptr;
        ws.B_conj = nullptr;
    }
    if (ws.plan == 0) {
        cufft_check(cufftPlan1d(&ws.plan,
                                static_cast<int>(Nfft),
                                CUFFT_Z2Z,
                                static_cast<int>(B)),
                    "cufftPlan1d");
        ws.plan_batch = B;
        ws.plan_Nfft = Nfft;
    }
    const std::size_t scratch_elems = B * Nfft;
    const std::size_t scratch_bytes = scratch_elems * sizeof(cuDoubleComplex);
    if (!ws.A) cuda_check(cudaMalloc(reinterpret_cast<void**>(&ws.A), scratch_bytes), "cudaMalloc(ws.A)");
    if (!ws.B) cuda_check(cudaMalloc(reinterpret_cast<void**>(&ws.B), scratch_bytes), "cudaMalloc(ws.B)");

    // ---- ensure scan temp (for length Nt scans) ----
    {
        size_t need = 0;
        cub::DeviceScan::InclusiveScan(nullptr, need,
                                       ws.A, ws.A,
                                       Cadd{}, static_cast<int>(Nt), stream);
        if (need > ws.scan_tmp_bytes) {
            if (ws.scan_tmp) cuda_check(cudaFree(ws.scan_tmp), "cudaFree(ws.scan_tmp)");
            cuda_check(cudaMalloc(&ws.scan_tmp, need), "cudaMalloc(ws.scan_tmp)");
            ws.scan_tmp_bytes = need;
        }
    }

    const FcrBlockConfig cfg = get_fcr_block_config();
    const int block_t = cfg.block_t;
    int block_b = cfg.block_b;
    if (block_b > static_cast<int>(B)) block_b = static_cast<int>(B);
    if (block_b < 1) block_b = 1;
    const dim3 block(block_t, block_b);
    const dim3 grid_conv = grid_2d_tiled(Nfft, B, block_t, block_b);
    const dim3 grid_time = grid_2d_tiled(Nt, B, block_t, block_b);

    const bool want_profile = fcr_profile_enabled();
    static std::atomic<int> profile_once{0};
    const bool profile_this = want_profile && (profile_once.fetch_add(1) == 0);
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_f = nullptr;
    cudaEvent_t ev_c = nullptr;
    cudaEvent_t ev_r = nullptr;
    if (profile_this) {
        cuda_check(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&ev_f), "cudaEventCreate(F)");
        cuda_check(cudaEventCreate(&ev_c), "cudaEventCreate(C)");
        cuda_check(cudaEventCreate(&ev_r), "cudaEventCreate(R)");
        cuda_check(cudaEventRecord(ev_start, stream), "cudaEventRecord(start)");
    }

    // ---- F: g2m (mirror[j]) ----
    // Compute convF(t) = dt * causal_conv( g1(t)*e^{-iωt}, g2_mirror(t) )
    kernel_pack_conv_F<<<grid_conv, block, cfg.smem2, stream>>>(ws.A, ws.B,
                                                        inputs.gamma, inputs.omegas, inputs.mirror,
                                                        batch.i, batch.j, batch.k0, static_cast<int>(B),
                                                        Nt, Nfft, inputs.dt);
    cuda_check(cudaGetLastError(), "kernel_pack_conv_F launch");
    convolution_cufft(ws.plan, ws.A, ws.B, B, Nfft, inputs.dt, stream);

    kernel_pack_prefix_F<<<grid_time, block, cfg.smem1, stream>>>(ws.B,
                                                           inputs.gamma, inputs.omegas, inputs.mirror,
                                                           batch.i, batch.j, batch.k0, static_cast<int>(B),
                                                           Nt, inputs.dt);
    cuda_check(cudaGetLastError(), "kernel_pack_prefix_F launch");
    for (std::size_t b = 0; b < B; ++b) {
        const cuDoubleComplex* in = ws.B + b * Nt;
        cuDoubleComplex* out = batch.F + b * Nt;
        cub::DeviceScan::InclusiveScan(ws.scan_tmp, ws.scan_tmp_bytes,
                                       in, out,
                                       Cadd{}, static_cast<int>(Nt), stream);
    }
    kernel_assemble_FC<<<grid_time, block, cfg.smem1, stream>>>(batch.F, ws.A,
                                                        inputs.gamma, inputs.omegas,
                                                        batch.i, batch.j, batch.k0, static_cast<int>(B),
                                                        Nt, Nfft, inputs.dt);
    cuda_check(cudaGetLastError(), "kernel_assemble_FC(F) launch");
    if (profile_this) {
        cuda_check(cudaEventRecord(ev_f, stream), "cudaEventRecord(F)");
    }

    // ---- C: conj(g2) ----
    // Compute convC(t) = dt * causal_conv( g1(t)*e^{-iωt}, conj(g2(t)) )
    kernel_pack_conv_C<<<grid_conv, block, cfg.smem2, stream>>>(ws.A, ws.B,
                                                        inputs.gamma, inputs.omegas,
                                                        batch.i, batch.j, batch.k0, static_cast<int>(B),
                                                        Nt, Nfft, inputs.dt);
    cuda_check(cudaGetLastError(), "kernel_pack_conv_C launch");
    convolution_cufft(ws.plan, ws.A, ws.B, B, Nfft, inputs.dt, stream);

    kernel_pack_prefix_C<<<grid_time, block, cfg.smem1, stream>>>(ws.B,
                                                          inputs.gamma, inputs.omegas,
                                                          batch.i, batch.j, batch.k0, static_cast<int>(B),
                                                          Nt, inputs.dt);
    cuda_check(cudaGetLastError(), "kernel_pack_prefix_C launch");
    for (std::size_t b = 0; b < B; ++b) {
        const cuDoubleComplex* in = ws.B + b * Nt;
        cuDoubleComplex* out = batch.C + b * Nt;
        cub::DeviceScan::InclusiveScan(ws.scan_tmp, ws.scan_tmp_bytes,
                                       in, out,
                                       Cadd{}, static_cast<int>(Nt), stream);
    }
    kernel_assemble_FC<<<grid_time, block, cfg.smem1, stream>>>(batch.C, ws.A,
                                                        inputs.gamma, inputs.omegas,
                                                        batch.i, batch.j, batch.k0, static_cast<int>(B),
                                                        Nt, Nfft, inputs.dt);
    cuda_check(cudaGetLastError(), "kernel_assemble_FC(C) launch");
    if (profile_this) {
        cuda_check(cudaEventRecord(ev_c, stream), "cudaEventRecord(C)");
    }

    // ---- R: prefix(g2*e^{-iwt}) and prefix((g1*g2)*e^{-iwt}) ----
    kernel_pack_R_integrands<<<grid_time, block, cfg.smem2, stream>>>(ws.A, ws.B,
                                                              inputs.gamma, inputs.omegas,
                                                              batch.i, batch.j, batch.k0, static_cast<int>(B),
                                                              Nt, inputs.dt);
    cuda_check(cudaGetLastError(), "kernel_pack_R_integrands launch");
    for (std::size_t b = 0; b < B; ++b) {
        const cuDoubleComplex* in = ws.A + b * Nt;
        cuDoubleComplex* out = batch.R + b * Nt;
        cub::DeviceScan::InclusiveScan(ws.scan_tmp, ws.scan_tmp_bytes,
                                       in, out,
                                       Cadd{}, static_cast<int>(Nt), stream);
    }
    for (std::size_t b = 0; b < B; ++b) {
        const cuDoubleComplex* in = ws.B + b * Nt;
        cuDoubleComplex* out = ws.A + b * Nt; // reuse ws.A for prefix_P
        cub::DeviceScan::InclusiveScan(ws.scan_tmp, ws.scan_tmp_bytes,
                                       in, out,
                                       Cadd{}, static_cast<int>(Nt), stream);
    }
    kernel_assemble_R<<<grid_time, block, cfg.smem1, stream>>>(batch.R, ws.A,
                                                       inputs.gamma,
                                                       batch.i, static_cast<int>(B), Nt);
    cuda_check(cudaGetLastError(), "kernel_assemble_R launch");
    if (profile_this) {
        cuda_check(cudaEventRecord(ev_r, stream), "cudaEventRecord(R)");
        cuda_check(cudaEventSynchronize(ev_r), "cudaEventSynchronize(R)");
        float ms_f = 0.0f;
        float ms_c = 0.0f;
        float ms_r = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms_f, ev_start, ev_f), "cudaEventElapsedTime(F)");
        cuda_check(cudaEventElapsedTime(&ms_c, ev_f, ev_c), "cudaEventElapsedTime(C)");
        cuda_check(cudaEventElapsedTime(&ms_r, ev_c, ev_r), "cudaEventElapsedTime(R)");
        std::fprintf(stderr,
                     "cuda_fcr profile: block=%dx%d B=%zu Nt=%zu Nfft=%zu ms_F=%.3f ms_C=%.3f ms_R=%.3f\n",
                     block_t, block_b, B, Nt, Nfft, ms_f, ms_c, ms_r);
        cuda_check(cudaEventDestroy(ev_start), "cudaEventDestroy(start)");
        cuda_check(cudaEventDestroy(ev_f), "cudaEventDestroy(F)");
        cuda_check(cudaEventDestroy(ev_c), "cudaEventDestroy(C)");
        cuda_check(cudaEventDestroy(ev_r), "cudaEventDestroy(R)");
    }
}
} // namespace taco::tcl4::cuda_fcr
