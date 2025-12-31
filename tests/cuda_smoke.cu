#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

__global__ void axpby(const double* x, const double* y, double* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0 * x[i] + y[i];
}

bool check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(status));
    return false;
}

} // namespace

int main() {
    constexpr int n = 1 << 20;

    std::vector<double> host_x(n, 1.0);
    std::vector<double> host_y(n, 3.0);
    std::vector<double> host_out(n, 0.0);

    double* device_x = nullptr;
    double* device_y = nullptr;
    double* device_out = nullptr;

    if (!check(cudaMalloc(&device_x, sizeof(double) * n), "cudaMalloc(x)")) return 1;
    if (!check(cudaMalloc(&device_y, sizeof(double) * n), "cudaMalloc(y)")) return 1;
    if (!check(cudaMalloc(&device_out, sizeof(double) * n), "cudaMalloc(out)")) return 1;

    if (!check(cudaMemcpy(device_x, host_x.data(), sizeof(double) * n, cudaMemcpyHostToDevice), "cudaMemcpy(x)")) return 1;
    if (!check(cudaMemcpy(device_y, host_y.data(), sizeof(double) * n, cudaMemcpyHostToDevice), "cudaMemcpy(y)")) return 1;

    constexpr int block = 256;
    const int grid = (n + block - 1) / block;
    axpby<<<grid, block>>>(device_x, device_y, device_out, n);
    if (!check(cudaGetLastError(), "kernel launch")) return 1;
    if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return 1;

    if (!check(cudaMemcpy(host_out.data(), device_out, sizeof(double) * n, cudaMemcpyDeviceToHost), "cudaMemcpy(out)")) return 1;

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_out);

    for (int i = 0; i < n; ++i) {
        const double expected = 5.0;
        if (std::abs(host_out[i] - expected) > 0.0) {
            std::fprintf(stderr, "Mismatch at %d: got=%g expected=%g\n", i, host_out[i], expected);
            return 2;
        }
    }

    std::puts("cuda_smoke: ok");
    return 0;
}

