#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "taco/exec.hpp"
#include "taco/generator.hpp"
#include "taco/ops.hpp"
#include "taco/rk4_dense.hpp"
#include "taco/system.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_mikx.hpp"
#ifdef TACO_HAS_CUDA
#include <cuda_runtime_api.h>
#include "taco/backend/cuda/tcl4_fused_cuda.hpp"
#include "taco/backend/cuda/rk4_dense_cuda.hpp"
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

double max_abs_diff(const Eigen::VectorXcd& a, const Eigen::VectorXcd& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    return (a - b).cwiseAbs().maxCoeff();
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
    int gpu_warmup = 1;
    std::size_t rk4_steps = 50;
    int rk4_order = 4;
    std::string rk4_method = "warp";
    std::string precision = "fp64";
    bool no_check = false;

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
        else if (arg.rfind("--rk4_steps=", 0) == 0) rk4_steps = static_cast<std::size_t>(std::stoull(arg.substr(12)));
        else if (arg.rfind("--rk4_order=", 0) == 0) rk4_order = std::stoi(arg.substr(12));
        else if (arg.rfind("--rk4_method=", 0) == 0) rk4_method = arg.substr(13);
        else if (arg.rfind("--precision=", 0) == 0) precision = arg.substr(12);
        else if (arg.rfind("--cuda_precision=", 0) == 0) precision = arg.substr(17);
        else if (arg == "--no_check" || arg == "--no-check") no_check = true;
    }

    if (!(rk4_order == 0 || rk4_order == 2 || rk4_order == 4)) {
        throw std::invalid_argument("--rk4_order must be 0, 2, or 4");
    }
    if (rk4_steps == 0) {
        throw std::invalid_argument("--rk4_steps must be > 0");
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
    {
        std::string p = precision;
        std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (p == "fp64" || p == "f64" || p == "double") {
            exec_gpu.cuda_precision = CudaPrecision::Fp64;
        } else if (p == "fp32" || p == "f32" || p == "float") {
            exec_gpu.cuda_precision = CudaPrecision::Fp32;
        } else {
            throw std::invalid_argument("--precision must be 'fp64' or 'fp32'");
        }
    }

    tcl::Rk4DenseCudaMethod rk4_cuda_method = tcl::Rk4DenseCudaMethod::WarpKernel;
    {
        std::string m = rk4_method;
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (m == "warp" || m == "kernel") {
            rk4_cuda_method = tcl::Rk4DenseCudaMethod::WarpKernel;
        } else if (m == "cublas" || m == "cublasgemv") {
            rk4_cuda_method = tcl::Rk4DenseCudaMethod::CublasGemv;
        } else {
            throw std::invalid_argument("--rk4_method must be 'warp' or 'cublas'");
        }
    }

    bool ok = true;

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
    const char* gpu_prec_name = (exec_gpu.cuda_precision == CudaPrecision::Fp32) ? "fp32" : "fp64";
    std::cout << "E2E L4 compare: max_abs=" << max_err
              << " max_rel=" << max_rel_err
              << " cpu_fcr_ms=" << cpu_kernel_ms
              << " gpu_fcr_ms=" << gpu_fcr_ms
              << " cpu_total_ms=" << cpu_end_to_end
              << " cpu_avg_ms=" << (cpu_end_to_end / count)
              << " gpu_total_ms=" << gpu_total_ms
              << " gpu_avg_ms=" << (gpu_total_ms / count)
              << " gpu_precision=" << gpu_prec_name
              << "\n";

    const double tol = (exec_gpu.cuda_precision == CudaPrecision::Fp32) ? 1e-4 : 1e-8;
    if (!no_check) {
        if (max_err > tol && max_rel_err > tol) {
            std::cerr << "FAIL: L4 mismatch above tolerance\n";
            ok = false;
        }
    }

    if (Nt < 2) {
        std::cerr << "RK4 compare skipped: Nt < 2\n";
    } else {
        const std::size_t n_steps = std::min(rk4_steps, Nt - 1);
            const std::size_t dim = system.eig.dim;
            const std::size_t D_u = dim * dim;
            if (D_u == 0 || D_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("RK4 compare: state dimension too large for int indexing");
            }
            if (D_u > std::numeric_limits<std::size_t>::max() / D_u) {
                throw std::overflow_error("RK4 compare: dense matrix size overflow");
            }

            std::vector<std::size_t> rk_tidx(n_steps + 1);
            for (std::size_t k = 0; k <= n_steps; ++k) rk_tidx[k] = k;

            // Build TCL4 correction series on CPU/GPU for the propagation window.
            std::vector<Eigen::MatrixXcd> L4_cpu_series;
            std::vector<Eigen::MatrixXcd> L4_gpu_series;
            double rk4_gpu_fcr_ms = 0.0;

            if (rk4_order == 4) {
                L4_cpu_series.reserve(rk_tidx.size());
                for (std::size_t tidx : rk_tidx) {
                    auto mikx = tcl4::build_mikx(map, kernels, tidx);
                    const Eigen::MatrixXcd GW = tcl4::assemble_liouvillian(mikx, system.A_eig);
                    L4_cpu_series.push_back(tcl4::gw_to_liouvillian(GW, system.eig.dim));
                }

                for (int w = 0; w < gpu_warmup; ++w) {
                    (void)tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, rk_tidx,
                                                                      tcl4::FCRMethod::Convolution, exec_gpu, nullptr);
                }
                L4_gpu_series =
                    tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, rk_tidx,
                                                                tcl4::FCRMethod::Convolution, exec_gpu, &rk4_gpu_fcr_ms);
            } else {
                L4_cpu_series.assign(rk_tidx.size(), Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(D_u),
                                                                            static_cast<Eigen::Index>(D_u)));
                L4_gpu_series = L4_cpu_series;
            }

            const Eigen::MatrixXcd H0 = system.eig.eps.asDiagonal().toDenseMatrix().cast<std::complex<double>>();
            const Eigen::MatrixXcd L0 = tcl2::build_unitary_superop(system, H0);

            tcl2::SpectralKernels K2;
            K2.buckets.resize(nf);
            for (std::size_t b = 0; b < nf; ++b) {
                K2.buckets[b].omega = system.fidx.buckets[b].omega;
                K2.buckets[b].Gamma = Eigen::MatrixXcd::Zero(1, 1);
            }

            auto fill_tcl2_kernels = [&](std::size_t time_index) {
                for (std::size_t b = 0; b < nf; ++b) {
                    K2.buckets[b].Gamma(0, 0) = gamma_series(static_cast<Eigen::Index>(time_index), static_cast<Eigen::Index>(b));
                }
            };

            auto build_L_at = [&](std::size_t time_index, const std::vector<Eigen::MatrixXcd>& L4_series) -> Eigen::MatrixXcd {
                if (rk4_order == 0) return L0;
                fill_tcl2_kernels(time_index);
                const tcl2::TCL2Components comps2 = tcl2::build_tcl2_components(system, K2, /*cutoff=*/0.0);
                Eigen::MatrixXcd L = comps2.total();
                if (rk4_order == 4) {
                    L.noalias() += L4_series[time_index];
                }
                return L;
            };

            // Initial rho in lab basis: |0><0|
            Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
            rho0(0, 0) = 1.0;
            const Eigen::MatrixXcd rho0_eig = system.eig.rho_to_eigen(rho0);

            Eigen::VectorXcd r_cpu = ops::vec(rho0_eig);
            Eigen::VectorXcd r_gpu = r_cpu;

            // CPU RK4 propagation (dense, time-dependent via endpoint averaging).
            const auto cpu_start = std::chrono::high_resolution_clock::now();
            {
                tcl::Rk4DenseWorkspace ws;
                ws.resize(static_cast<Eigen::Index>(D_u));

                Eigen::MatrixXcd L_cur = build_L_at(0, L4_cpu_series);
                Eigen::MatrixXcd L_next = build_L_at(1, L4_cpu_series);

                for (std::size_t step = 0; step < n_steps; ++step) {
                    const Eigen::MatrixXcd Lhalf = 0.5 * (L_cur + L_next);
                    tcl::rk4_update_serial(L_cur, Lhalf, L_next, r_cpu, ws, dt);

                    const std::size_t step1 = step + 1;
                    if (step1 < n_steps) {
                        L_cur = std::move(L_next);
                        L_next = build_L_at(step + 2, L4_cpu_series);
                    }
                }
            }
            const auto cpu_end = std::chrono::high_resolution_clock::now();
            const double cpu_rk4_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

            // GPU RK4 propagation (host matrices, device integration).
            const auto gpu_start_rk4 = std::chrono::high_resolution_clock::now();
            {
                auto cuda_check = [](cudaError_t status, const char* what) {
                    if (status == cudaSuccess) return;
                    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
                };

                cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice(rk4)");

                const int D = static_cast<int>(D_u);
                const std::size_t L_elems = D_u * D_u;
                const cudaStream_t stream = 0;

                if (exec_gpu.cuda_precision == CudaPrecision::Fp32) {
                    const float dt_f32 = static_cast<float>(dt);
                    const std::size_t vbytes = D_u * sizeof(cuFloatComplex);
                    const std::size_t Lbytes = L_elems * sizeof(cuFloatComplex);

                    auto pack_vec = [&](const Eigen::VectorXcd& src, std::vector<cuFloatComplex>& dst) {
                        dst.resize(D_u);
                        for (std::size_t i = 0; i < D_u; ++i) {
                            const std::complex<double> z = src(static_cast<Eigen::Index>(i));
                            dst[i] = make_cuFloatComplex(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                        }
                    };
                    auto unpack_vec = [&](const std::vector<cuFloatComplex>& src, Eigen::VectorXcd& dst) {
                        for (std::size_t i = 0; i < D_u; ++i) {
                            const auto z = src[i];
                            dst(static_cast<Eigen::Index>(i)) =
                                std::complex<double>(static_cast<double>(z.x), static_cast<double>(z.y));
                        }
                    };
                    auto pack_mat = [&](const Eigen::MatrixXcd& src, std::vector<cuFloatComplex>& dst) {
                        dst.resize(L_elems);
                        const auto* p = src.data(); // column-major
                        for (std::size_t i = 0; i < L_elems; ++i) {
                            dst[i] = make_cuFloatComplex(static_cast<float>(p[i].real()), static_cast<float>(p[i].imag()));
                        }
                    };

                    std::vector<cuFloatComplex> h_r;
                    std::vector<cuFloatComplex> h_L;

                    cuFloatComplex* d_r = nullptr;
                    cuda_check(cudaMalloc(&d_r, vbytes), "cudaMalloc(r_f32)");
                    auto free_r = [&] {
                        if (d_r) cudaFree(d_r);
                        d_r = nullptr;
                    };

                    tcl::Rk4DenseCudaWorkspaceF32 ws_cuda;

                    try {
                        pack_vec(r_gpu, h_r);
                        cuda_check(cudaMemcpy(d_r, h_r.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r_f32 H2D)");

                        if (rk4_order == 0) {
                            cuFloatComplex* d_L = nullptr;
                            cuda_check(cudaMalloc(&d_L, Lbytes), "cudaMalloc(L_f32)");
                            auto free_L = [&] {
                                if (d_L) cudaFree(d_L);
                                d_L = nullptr;
                            };

                            try {
                                pack_mat(L0, h_L);
                                cuda_check(cudaMemcpy(d_L, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L_f32 H2D)");

                                for (std::size_t step = 0; step < n_steps; ++step) {
                                    tcl::rk4_update_cuda_f32(d_L, d_r, D, ws_cuda, dt_f32, stream, rk4_cuda_method);
                                }
                            } catch (...) {
                                free_L();
                                throw;
                            }
                            free_L();
                        } else {
                            cuFloatComplex* d_L0 = nullptr;
                            cuFloatComplex* d_L1 = nullptr;
                            cuFloatComplex* d_Lhalf = nullptr;
                            cuda_check(cudaMalloc(&d_L0, Lbytes), "cudaMalloc(L0_f32)");
                            cuda_check(cudaMalloc(&d_L1, Lbytes), "cudaMalloc(L1_f32)");
                            cuda_check(cudaMalloc(&d_Lhalf, Lbytes), "cudaMalloc(Lhalf_f32)");

                            auto free_mats = [&] {
                                if (d_L0) cudaFree(d_L0);
                                if (d_L1) cudaFree(d_L1);
                                if (d_Lhalf) cudaFree(d_Lhalf);
                            };

                            try {
                                Eigen::MatrixXcd L_cur = build_L_at(0, L4_gpu_series);
                                Eigen::MatrixXcd L_next = build_L_at(1, L4_gpu_series);

                                pack_mat(L_cur, h_L);
                                cuda_check(cudaMemcpy(d_L0, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L0_f32 H2D)");
                                pack_mat(L_next, h_L);
                                cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L1_f32 H2D)");

                                for (std::size_t step = 0; step < n_steps; ++step) {
                                    tcl::half_sum_cuda_f32(d_L0, d_L1, d_Lhalf, L_elems, stream);
                                    tcl::rk4_update_cuda_f32(d_L0, d_Lhalf, d_L1, d_r, D, ws_cuda, dt_f32, stream, rk4_cuda_method);

                                    const std::size_t step1 = step + 1;
                                    if (step1 < n_steps) {
                                        L_cur = std::move(L_next);
                                        L_next = build_L_at(step + 2, L4_gpu_series);

                                        std::swap(d_L0, d_L1);
                                        pack_mat(L_next, h_L);
                                        cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice),
                                                   "cudaMemcpy(Lnext_f32 H2D)");
                                    }
                                }
                            } catch (...) {
                                free_mats();
                                throw;
                            }

                            free_mats();
                        }

                        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(rk4_f32)");
                        cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r_f32 D2H)");
                        unpack_vec(h_r, r_gpu);
                    } catch (...) {
                        free_r();
                        throw;
                    }

                    free_r();
                } else {
                    const std::size_t vbytes = D_u * sizeof(cuDoubleComplex);
                    const std::size_t Lbytes = L_elems * sizeof(cuDoubleComplex);

                    auto pack_vec = [&](const Eigen::VectorXcd& src, std::vector<cuDoubleComplex>& dst) {
                        dst.resize(D_u);
                        for (std::size_t i = 0; i < D_u; ++i) {
                            const std::complex<double> z = src(static_cast<Eigen::Index>(i));
                            dst[i] = make_cuDoubleComplex(z.real(), z.imag());
                        }
                    };
                    auto unpack_vec = [&](const std::vector<cuDoubleComplex>& src, Eigen::VectorXcd& dst) {
                        for (std::size_t i = 0; i < D_u; ++i) {
                            const auto z = src[i];
                            dst(static_cast<Eigen::Index>(i)) = std::complex<double>(z.x, z.y);
                        }
                    };
                    auto pack_mat = [&](const Eigen::MatrixXcd& src, std::vector<cuDoubleComplex>& dst) {
                        dst.resize(L_elems);
                        const auto* p = src.data(); // column-major
                        for (std::size_t i = 0; i < L_elems; ++i) dst[i] = make_cuDoubleComplex(p[i].real(), p[i].imag());
                    };

                    std::vector<cuDoubleComplex> h_r;
                    std::vector<cuDoubleComplex> h_L;

                    cuDoubleComplex* d_r = nullptr;
                    cuda_check(cudaMalloc(&d_r, vbytes), "cudaMalloc(r)");
                    auto free_r = [&] {
                        if (d_r) cudaFree(d_r);
                        d_r = nullptr;
                    };

                    tcl::Rk4DenseCudaWorkspace ws_cuda;

                    try {
                        pack_vec(r_gpu, h_r);
                        cuda_check(cudaMemcpy(d_r, h_r.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r H2D)");

                        if (rk4_order == 0) {
                            cuDoubleComplex* d_L = nullptr;
                            cuda_check(cudaMalloc(&d_L, Lbytes), "cudaMalloc(L)");
                            auto free_L = [&] {
                                if (d_L) cudaFree(d_L);
                                d_L = nullptr;
                            };

                            try {
                                pack_mat(L0, h_L);
                                cuda_check(cudaMemcpy(d_L, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L H2D)");

                                for (std::size_t step = 0; step < n_steps; ++step) {
                                    tcl::rk4_update_cuda(d_L, d_r, D, ws_cuda, dt, stream, rk4_cuda_method);
                                }
                            } catch (...) {
                                free_L();
                                throw;
                            }
                            free_L();
                        } else {
                            cuDoubleComplex* d_L0 = nullptr;
                            cuDoubleComplex* d_L1 = nullptr;
                            cuDoubleComplex* d_Lhalf = nullptr;
                            cuda_check(cudaMalloc(&d_L0, Lbytes), "cudaMalloc(L0)");
                            cuda_check(cudaMalloc(&d_L1, Lbytes), "cudaMalloc(L1)");
                            cuda_check(cudaMalloc(&d_Lhalf, Lbytes), "cudaMalloc(Lhalf)");

                            auto free_mats = [&] {
                                if (d_L0) cudaFree(d_L0);
                                if (d_L1) cudaFree(d_L1);
                                if (d_Lhalf) cudaFree(d_Lhalf);
                            };

                            try {
                                Eigen::MatrixXcd L_cur = build_L_at(0, L4_gpu_series);
                                Eigen::MatrixXcd L_next = build_L_at(1, L4_gpu_series);

                                pack_mat(L_cur, h_L);
                                cuda_check(cudaMemcpy(d_L0, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L0 H2D)");
                                pack_mat(L_next, h_L);
                                cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L1 H2D)");

                                for (std::size_t step = 0; step < n_steps; ++step) {
                                    tcl::half_sum_cuda(d_L0, d_L1, d_Lhalf, L_elems, stream);
                                    tcl::rk4_update_cuda(d_L0, d_Lhalf, d_L1, d_r, D, ws_cuda, dt, stream, rk4_cuda_method);

                                    const std::size_t step1 = step + 1;
                                    if (step1 < n_steps) {
                                        L_cur = std::move(L_next);
                                        L_next = build_L_at(step + 2, L4_gpu_series);

                                        std::swap(d_L0, d_L1);
                                        pack_mat(L_next, h_L);
                                        cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice),
                                                   "cudaMemcpy(Lnext H2D)");
                                    }
                                }
                            } catch (...) {
                                free_mats();
                                throw;
                            }

                            free_mats();
                        }

                        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(rk4)");
                        cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r D2H)");
                        unpack_vec(h_r, r_gpu);
                    } catch (...) {
                        free_r();
                        throw;
                    }

                    free_r();
                }
            }
            const auto gpu_end_rk4 = std::chrono::high_resolution_clock::now();
            const double gpu_rk4_ms = std::chrono::duration<double, std::milli>(gpu_end_rk4 - gpu_start_rk4).count();

            const auto rho_from_r = [&](const Eigen::VectorXcd& r) {
                Eigen::MatrixXcd rho_eig = ops::unvec(r, static_cast<std::size_t>(dim));
                rho_eig = ops::hermitize_and_normalize(rho_eig);
                return system.eig.rho_to_lab(rho_eig);
            };

            const Eigen::MatrixXcd rho_cpu = rho_from_r(r_cpu);
            const Eigen::MatrixXcd rho_gpu = rho_from_r(r_gpu);

            const double r_abs = max_abs_diff(r_cpu, r_gpu);
            const double rho_abs = max_abs_diff(rho_cpu, rho_gpu);
            const double rho_ref = std::max(1.0, rho_cpu.cwiseAbs().maxCoeff());
            const double rho_rel = rho_abs / rho_ref;

            std::cout << "RK4 compare: steps=" << n_steps
                      << " order=" << rk4_order
                      << " method=" << rk4_method
                      << " max_abs_r=" << r_abs
                      << " max_abs_rho=" << rho_abs
                      << " max_rel_rho=" << rho_rel
                      << " cpu_rk4_ms=" << cpu_rk4_ms
                      << " gpu_rk4_ms=" << gpu_rk4_ms
                      << " gpu_fcr_ms_rk4=" << rk4_gpu_fcr_ms
                      << " gpu_precision=" << ((exec_gpu.cuda_precision == CudaPrecision::Fp32) ? "fp32" : "fp64")
                      << "\n";

            const double rk4_tol = (exec_gpu.cuda_precision == CudaPrecision::Fp32) ? 1e-4 : 1e-5;
            if (!no_check) {
                if (rho_abs > rk4_tol && rho_rel > rk4_tol) {
                    std::cerr << "FAIL: RK4 propagation mismatch above tolerance\n";
                    ok = false;
                }
            }
    }

    if (!ok) {
        std::cout << "FAIL\n";
        return 1;
    }
    if (no_check) {
        std::cout << "PASS (checks skipped)\n";
    } else {
        std::cout << "PASS\n";
    }
    return 0;
#endif
}
