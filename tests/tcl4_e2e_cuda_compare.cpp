#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "taco/exec.hpp"
#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_mikx.hpp"
#ifdef TACO_HAS_CUDA
#include "taco/backend/cuda/tcl4_fused_cuda.hpp"
#endif

namespace {

double max_abs_diff(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return std::numeric_limits<double>::infinity();
    return (a - b).cwiseAbs().maxCoeff();
}

std::size_t clamp_tidx(std::size_t tidx, std::size_t Nt) {
    if (Nt == 0) return 0;
    return std::min(tidx, Nt - 1);
}

std::vector<std::size_t> parse_tidx_spec(const std::string& spec, std::size_t Nt) {
    if (spec.empty()) return {};

    std::vector<std::size_t> parts;
    parts.reserve(3);
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t next = spec.find(':', pos);
        const std::size_t len = (next == std::string::npos) ? (spec.size() - pos) : (next - pos);
        const std::string token = spec.substr(pos, len);
        if (token.empty()) {
            throw std::invalid_argument("invalid --tidx spec (empty token)");
        }
        parts.push_back(static_cast<std::size_t>(std::stoull(token)));
        if (next == std::string::npos) break;
        pos = next + 1;
    }

    if (parts.size() == 1) {
        return {clamp_tidx(parts[0], Nt)};
    }

    std::size_t start = 0;
    std::size_t step = 1;
    std::size_t end = 0;
    if (parts.size() == 2) {
        start = parts[0];
        end = parts[1];
    } else if (parts.size() == 3) {
        start = parts[0];
        step = parts[1];
        end = parts[2];
    } else {
        throw std::invalid_argument("invalid --tidx spec (expected k or a:b or a:step:b)");
    }

    if (step == 0) {
        throw std::invalid_argument("invalid --tidx spec (step must be > 0)");
    }

    start = clamp_tidx(start, Nt);
    end = clamp_tidx(end, Nt);
    if (start > end) {
        throw std::invalid_argument("invalid --tidx spec (start must be <= end)");
    }

    std::vector<std::size_t> out;
    out.reserve((end - start) / step + 1);
    for (std::size_t t = start; t <= end; t += step) {
        out.push_back(t);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    using namespace taco;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);

    std::size_t Nt_samples = 100000;
    double dt = 0.000625;
    double beta = 0.5;
    double omega_c = 10.0;
    std::string tidx_spec;
    bool has_tidx_spec = false;
    bool run_series = false;
    int gpu_id = 0;
    int threads = 0;
    int gpu_warmup = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--N=", 0) == 0) Nt_samples = static_cast<std::size_t>(std::stoull(arg.substr(4)));
        else if (arg.rfind("--dt=", 0) == 0) dt = std::stod(arg.substr(5));
        else if (arg.rfind("--beta=", 0) == 0) beta = std::stod(arg.substr(7));
        else if (arg.rfind("--omega_c=", 0) == 0) omega_c = std::stod(arg.substr(10));
        else if (arg.rfind("--tidx=", 0) == 0) {
            tidx_spec = arg.substr(7);
            has_tidx_spec = true;
        }
        else if (arg == "--series") run_series = true;
        else if (arg.rfind("--gpu_id=", 0) == 0) gpu_id = std::stoi(arg.substr(9));
        else if (arg.rfind("--threads=", 0) == 0) threads = std::stoi(arg.substr(10));
        else if (arg.rfind("--gpu_warmup=", 0) == 0) gpu_warmup = std::stoi(arg.substr(13));
    }

    Eigen::MatrixXcd H = 0.5 * ops::sigma_x();
    Eigen::MatrixXcd A = 0.5 * ops::sigma_z();
    sys::System system;
    system.build(H, {A}, 1e-9);

    std::vector<double> t;
    std::vector<std::complex<double>> C;
    auto J = [&](double w) { return (w > 0.0) ? (w * std::exp(-w / omega_c)) : 0.0; };
    bcf::bcf_fft_fun(Nt_samples, dt, J, beta, t, C);

    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;
    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(C, dt, omegas);
    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    if (Nt == 0) {
        std::cerr << "gamma_series is empty\n";
        return 1;
    }

    std::vector<std::size_t> tidx_list;
    if (run_series) {
        tidx_list.resize(Nt);
        for (std::size_t t = 0; t < Nt; ++t) tidx_list[t] = t;
    } else if (has_tidx_spec) {
        tidx_list = parse_tidx_spec(tidx_spec, Nt);
    } else {
        tidx_list = {0, Nt / 2, Nt - 1};
        std::sort(tidx_list.begin(), tidx_list.end());
        tidx_list.erase(std::unique(tidx_list.begin(), tidx_list.end()), tidx_list.end());
    }

    Exec exec_cpu;
#ifdef _OPENMP
    exec_cpu.backend = Backend::Omp;
    exec_cpu.threads = threads;
#else
    exec_cpu.backend = Backend::Serial;
#endif

#ifndef TACO_HAS_CUDA
    std::cout << "CUDA not enabled; skipping GPU compare\n";
    return 0;
#else
    Exec exec_gpu;
    exec_gpu.backend = Backend::Cuda;
    exec_gpu.gpu_id = gpu_id;

    double max_err = 0.0;
    double max_rel_err = 0.0;
    double cpu_total_ms = 0.0;
    double cpu_kernel_ms = 0.0;
    double gpu_total_ms = 0.0;
    double gpu_fcr_ms = 0.0;
    const double count = static_cast<double>(tidx_list.size());

    std::vector<Eigen::MatrixXcd> L4_cpu_list;
    L4_cpu_list.reserve(tidx_list.size());

    const auto t_cpu_kernel_start = std::chrono::high_resolution_clock::now();
    const auto kernels = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2,
                                                      tcl4::FCRMethod::Convolution, exec_cpu);
    const tcl4::Tcl4Map map = tcl4::build_map(system, /*time_grid*/{});
    const auto t_cpu_kernel_end = std::chrono::high_resolution_clock::now();
    cpu_kernel_ms = std::chrono::duration<double, std::milli>(t_cpu_kernel_end - t_cpu_kernel_start).count();

    for (std::size_t tidx : tidx_list) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto mikx = tcl4::build_mikx(map, kernels, tidx);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const Eigen::MatrixXcd GW = tcl4::assemble_liouvillian(mikx, system.A_eig);
        const auto t2 = std::chrono::high_resolution_clock::now();
        const Eigen::MatrixXcd L4_cpu = tcl4::gw_to_liouvillian(GW, system.eig.dim);
        const auto t3 = std::chrono::high_resolution_clock::now();

        const double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
        cpu_total_ms += cpu_ms;
        L4_cpu_list.push_back(L4_cpu);
    }

    for (int w = 0; w < gpu_warmup; ++w) {
        (void)tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, tidx_list,
                                                          tcl4::FCRMethod::Convolution, exec_gpu, nullptr);
    }
    const auto t_gpu_start = std::chrono::high_resolution_clock::now();
    const auto L4_gpu_list =
        tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, tidx_list,
                                                    tcl4::FCRMethod::Convolution, exec_gpu, &gpu_fcr_ms);
    const auto t_gpu_end = std::chrono::high_resolution_clock::now();
    const double gpu_total =
        std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();
    gpu_total_ms = gpu_total;
    const double gpu_avg = gpu_total / count;

    for (std::size_t idx = 0; idx < tidx_list.size(); ++idx) {
        const Eigen::MatrixXcd& L4_cpu = L4_cpu_list[idx];
        const Eigen::MatrixXcd& L4_gpu = L4_gpu_list[idx];
        const double err = max_abs_diff(L4_cpu, L4_gpu);
        const double ref = std::max(1.0, L4_cpu.cwiseAbs().maxCoeff());
        const double rel = err / ref;
        max_err = std::max(max_err, err);
        max_rel_err = std::max(max_rel_err, rel);
    }

    const double cpu_end_to_end = cpu_kernel_ms + cpu_total_ms;
    std::cout << "E2E L4 compare: max_abs=" << max_err
              << " max_rel=" << max_rel_err
              << " cpu_fcr_ms=" << cpu_kernel_ms
              << " gpu_fcr_ms=" << gpu_fcr_ms
              << " cpu_total_ms=" << cpu_end_to_end
              << " cpu_avg_ms=" << (cpu_end_to_end / count)
              << " gpu_total_ms=" << gpu_total_ms
              << " gpu_avg_ms=" << (gpu_total_ms / count)
              << "\n";

    const double tol = 1e-8;
    if (max_err > tol && max_rel_err > tol) {
        std::cerr << "FAIL: L4 mismatch above tolerance\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
#endif
}
