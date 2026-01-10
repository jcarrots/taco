#include <algorithm>
#include <chrono>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"
#ifdef TACO_HAS_CUDA
#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"
#endif

static inline double rel_err(std::complex<double> a, std::complex<double> b) {
    const double den = std::max(1.0, std::abs(b));
    return std::abs(a - b) / den;
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static std::size_t parse_tidx_token(const std::string& token, std::size_t Nt) {
    const std::string t = lower_copy(token);
    if (t == "last") return (Nt > 0 ? Nt - 1 : 0);
    if (t == "mid") return (Nt > 0 ? Nt / 2 : 0);
    return static_cast<std::size_t>(std::stoull(token));
}

static void append_tidx_list(std::vector<std::size_t>& out,
                             const std::string& spec,
                             std::size_t Nt) {
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string token = (comma == std::string::npos)
            ? spec.substr(pos)
            : spec.substr(pos, comma - pos);
        if (!token.empty()) out.push_back(parse_tidx_token(token, Nt));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
}

static void append_tidx_range(std::vector<std::size_t>& out,
                              const std::string& spec,
                              std::size_t Nt) {
    const std::size_t c1 = spec.find(':');
    if (c1 == std::string::npos) return;
    const std::size_t c2 = spec.find(':', c1 + 1);
    const std::string s_start = spec.substr(0, c1);
    const std::string s_stop = (c2 == std::string::npos)
        ? spec.substr(c1 + 1)
        : spec.substr(c1 + 1, c2 - c1 - 1);
    const std::string s_step = (c2 == std::string::npos) ? "" : spec.substr(c2 + 1);
    std::size_t start = static_cast<std::size_t>(std::stoull(s_start));
    std::size_t stop = static_cast<std::size_t>(std::stoull(s_stop));
    std::size_t step = s_step.empty() ? 1 : static_cast<std::size_t>(std::stoull(s_step));
    if (step == 0) step = 1;
    if (start > stop) std::swap(start, stop);
    for (std::size_t v = start; v <= stop; v += step) {
        out.push_back(v);
        if (stop - v < step) break;
    }
}

static std::vector<std::size_t> resolve_tidx_list(std::size_t Nt,
                                                  const std::string& single_spec,
                                                  const std::string& list_spec,
                                                  const std::string& range_spec) {
    std::vector<std::size_t> out;
    if (!list_spec.empty()) append_tidx_list(out, list_spec, Nt);
    if (!range_spec.empty()) append_tidx_range(out, range_spec, Nt);
    if (out.empty()) {
        if (!single_spec.empty()) out.push_back(parse_tidx_token(single_spec, Nt));
        else out.push_back(Nt > 0 ? Nt - 1 : 0);
    }
    if (Nt == 0) return std::vector<std::size_t>{0};
    bool clamped = false;
    for (auto& idx : out) {
        if (idx >= Nt) { idx = Nt - 1; clamped = true; }
    }
    if (clamped) {
        std::cout << "tidx list clamped to last index " << (Nt - 1) << "\n";
    }
    return out;
}

struct ErrTriple {
    double F{0.0};
    double C{0.0};
    double R{0.0};
};

static ErrTriple max_rel_err(const taco::tcl4::TripleKernelSeries& a,
                             const taco::tcl4::TripleKernelSeries& b,
                             std::size_t tidx)
{
    ErrTriple out;
    const std::size_t nf = a.F.size();
    if (nf == 0) return out;
    const Eigen::Index t = static_cast<Eigen::Index>(tidx);
    for (std::size_t i = 0; i < nf; ++i) {
        for (std::size_t j = 0; j < nf; ++j) {
            for (std::size_t k = 0; k < nf; ++k) {
                out.F = std::max(out.F, rel_err(a.F[i][j][k](t), b.F[i][j][k](t)));
                out.C = std::max(out.C, rel_err(a.C[i][j][k](t), b.C[i][j][k](t)));
                out.R = std::max(out.R, rel_err(a.R[i][j][k](t), b.R[i][j][k](t)));
            }
        }
    }
    return out;
}

int main(int argc, char** argv) {
    using namespace taco;
    std::cout.setf(std::ios::fixed); std::cout.precision(9);

    // Simple 2-level model
    Eigen::MatrixXcd H = 0.5 * ops::sigma_x();
    Eigen::MatrixXcd A = 0.5 * ops::sigma_z();
    sys::System system; system.build(H, {A}, 1e-9);

    // Build C(t) via FFT
    double beta = 0.5, omega_c = 10.0, dt = 0.000625; std::size_t N = 4096;
    bool use_cuda = false;
    bool force_serial = false;
    bool force_omp = false;
    bool run_direct = true;
    bool run_mikx = false;
    bool mikx_serial = false;
    bool mikx_omp = false;
    bool mikx_cuda = false;
    int threads = 0;
    int gpu_id = 0;
    std::string tidx_spec;
    std::string tidx_list_spec;
    std::string tidx_range_spec;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--dt=",0)==0) dt = std::stod(arg.substr(5));
        else if (arg.rfind("--N=",0)==0) N = static_cast<std::size_t>(std::stoull(arg.substr(4)));
        else if (arg == "--cuda") use_cuda = true;
        else if (arg == "--serial") force_serial = true;
        else if (arg == "--omp") force_omp = true;
        else if (arg == "--no-direct") run_direct = false;
        else if (arg == "--mikx") run_mikx = true;
        else if (arg == "--mikx-serial") { run_mikx = true; mikx_serial = true; }
        else if (arg == "--mikx-omp") { run_mikx = true; mikx_omp = true; }
        else if (arg == "--mikx-cuda") { run_mikx = true; mikx_cuda = true; }
        else if (arg.rfind("--backend=",0)==0) {
            const std::string v = arg.substr(10);
            if (v == "cuda") use_cuda = true;
            else if (v == "serial") force_serial = true;
            else if (v == "omp") force_omp = true;
        } else if (arg.rfind("--threads=",0)==0) {
            threads = std::stoi(arg.substr(10));
        } else if (arg.rfind("--gpu_id=",0)==0) {
            gpu_id = std::stoi(arg.substr(9));
        } else if (arg.rfind("--tidx=",0)==0) {
            tidx_spec = arg.substr(7);
        } else if (arg.rfind("--tidx-list=",0)==0) {
            tidx_list_spec = arg.substr(12);
        } else if (arg.rfind("--tidx-range=",0)==0) {
            tidx_range_spec = arg.substr(13);
        }
    }
    std::vector<double> t; std::vector<std::complex<double>> C;
    auto J = [&](double w){ return (w>0.0) ? (w*std::exp(-w/omega_c)) : 0.0; };
    bcf::bcf_fft_fun(N, dt, J, beta, t, C);

    // Gamma series per bucket
    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;
    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(C, dt, omegas);

    std::vector<int> thread_cases;
#ifdef _OPENMP
    if (threads > 0) {
        thread_cases = {threads};
    } else {
        thread_cases = {1, std::max(1, omp_get_max_threads()/2), omp_get_max_threads()};
    }
#else
    thread_cases = {1};
#endif

    tcl4::TripleKernelSeries k_cpu_conv_last;
    bool have_cpu_conv = false;
    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    const std::vector<std::size_t> tidx_list =
        resolve_tidx_list(Nt, tidx_spec, tidx_list_spec, tidx_range_spec);

    for (int th : thread_cases) {
        Exec exec;
#ifdef _OPENMP
        if (force_serial) exec.backend = Backend::Serial;
        else exec.backend = Backend::Omp;
#else
        exec.backend = Backend::Serial;
#endif
        if (force_omp) exec.backend = Backend::Omp;
        exec.threads = th;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto k_conv = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Convolution, exec);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_fft = std::chrono::duration<double>(t1 - t0).count();

        ErrTriple err_direct;
        double t_dir = 0.0;
        if (run_direct) {
            auto t2 = std::chrono::high_resolution_clock::now();
            auto k_dir = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Direct, exec);
            auto t3 = std::chrono::high_resolution_clock::now();
            t_dir = std::chrono::duration<double>(t3 - t2).count();
            for (std::size_t ti : tidx_list) {
                ErrTriple e = max_rel_err(k_conv, k_dir, ti);
                err_direct.F = std::max(err_direct.F, e.F);
                err_direct.C = std::max(err_direct.C, e.C);
                err_direct.R = std::max(err_direct.R, e.R);
            }
        }

        std::cout << "cpu threads=" << th
                  << ", t_fft=" << t_fft << "s";
        if (run_direct) {
            std::cout << ", t_dir=" << t_dir
                      << "s, errF=" << err_direct.F
                      << ", errC=" << err_direct.C
                      << ", errR=" << err_direct.R;
        }
        std::cout << "\n";

        k_cpu_conv_last = std::move(k_conv);
        have_cpu_conv = true;
    }

        if (run_mikx) {
        if (!have_cpu_conv) {
            std::cout << "mikx requested but no CPU kernels are available\n";
        } else {
            tcl4::Tcl4Map map = tcl4::build_map(system, t);
            const bool run_cpu_default = (!mikx_serial && !mikx_omp && !mikx_cuda);
            bool want_serial = mikx_serial;
            bool want_omp = mikx_omp;
            if (run_cpu_default) {
#ifdef _OPENMP
                want_omp = true;
#else
                want_serial = true;
#endif
            }
            if (want_serial) {
                double total = 0.0;
                taco::tcl4::MikxTensors mikx;
                for (std::size_t ti : tidx_list) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    mikx = tcl4::build_mikx_serial(map, k_cpu_conv_last, ti);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    total += std::chrono::duration<double>(t1 - t0).count();
                }
                const double avg = total / static_cast<double>(tidx_list.size());
                std::cout << "mikx serial t_total=" << total << "s, avg=" << avg
                          << "s, n=" << tidx_list.size() << " (N=" << mikx.N << ")\n";
            }
#ifdef _OPENMP
            if (want_omp) {
                for (int th : thread_cases) {
                    if (th > 0 && !omp_in_parallel()) omp_set_num_threads(th);
                    double total = 0.0;
                    taco::tcl4::MikxTensors mikx;
                    for (std::size_t ti : tidx_list) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        mikx = tcl4::build_mikx_omp(map, k_cpu_conv_last, ti);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        total += std::chrono::duration<double>(t1 - t0).count();
                    }
                    const double avg = total / static_cast<double>(tidx_list.size());
                    std::cout << "mikx omp threads=" << th << ", t_total=" << total
                              << "s, avg=" << avg << "s, n=" << tidx_list.size()
                              << " (N=" << mikx.N << ")\n";
                }
            }
#endif
            if (mikx_cuda) {
#ifdef TACO_HAS_CUDA
                Exec exec;
                exec.backend = Backend::Cuda;
                exec.gpu_id = gpu_id;
                double total = 0.0;
                double kernel_ms_total = 0.0;
                int N_out = map.N;
                if (tidx_list.size() > 1) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto mikx_list = tcl4::build_mikx_cuda_batch(map, k_cpu_conv_last, tidx_list, exec, 0, &kernel_ms_total);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    total = std::chrono::duration<double>(t1 - t0).count();
                    if (!mikx_list.empty()) N_out = mikx_list.back().N;
                    const double avg = total / static_cast<double>(tidx_list.size());
                    const double kernel_total = kernel_ms_total * 1e-3;
                    const double kernel_avg = kernel_total / static_cast<double>(tidx_list.size());
                    std::cout << "mikx cuda batch gpu_id=" << gpu_id << ", t_total=" << total
                              << "s, t_kernel_total=" << kernel_total << "s, avg=" << avg
                              << "s, avg_kernel=" << kernel_avg << "s, n=" << tidx_list.size()
                              << " (N=" << N_out << ")\n";
                } else {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    double kernel_ms = 0.0;
                    auto mikx = tcl4::build_mikx_cuda(map, k_cpu_conv_last, tidx_list.front(), exec, &kernel_ms);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    total = std::chrono::duration<double>(t1 - t0).count();
                    const double kernel_total = kernel_ms * 1e-3;
                    std::cout << "mikx cuda gpu_id=" << gpu_id << ", t_total=" << total
                              << "s, t_kernel_total=" << kernel_total << "s, avg=" << total
                              << "s, avg_kernel=" << kernel_total << "s, n=1 (N=" << mikx.N << ")\n";
                }
#else
                std::cout << "mikx cuda requested but TACO_HAS_CUDA is not enabled in this build\n";
#endif
            }
        }
    }

    if (use_cuda) {
#ifdef TACO_HAS_CUDA
        Exec exec;
        exec.backend = Backend::Cuda;
        exec.gpu_id = gpu_id;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto k_gpu = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, tcl4::FCRMethod::Convolution, exec);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_gpu = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "gpu id=" << gpu_id << ", t_fft=" << t_gpu << "s";
        if (have_cpu_conv) {
            ErrTriple err_gpu;
            for (std::size_t ti : tidx_list) {
                ErrTriple e = max_rel_err(k_cpu_conv_last, k_gpu, ti);
                err_gpu.F = std::max(err_gpu.F, e.F);
                err_gpu.C = std::max(err_gpu.C, e.C);
                err_gpu.R = std::max(err_gpu.R, e.R);
            }
            std::cout << ", errF=" << err_gpu.F
                      << ", errC=" << err_gpu.C
                      << ", errR=" << err_gpu.R;
        }
        std::cout << "\n";
#else
        std::cout << "gpu requested but TACO_HAS_CUDA is not enabled in this build\n";
#endif
    }

    return 0;
}
