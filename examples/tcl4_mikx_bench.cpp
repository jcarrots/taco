#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <complex>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_mikx.hpp"
#ifdef TACO_HAS_CUDA
#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"
#endif

namespace {

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::size_t parse_tidx_token(const std::string& token, std::size_t Nt) {
    const std::string t = lower_copy(token);
    if (t == "last") return (Nt > 0 ? Nt - 1 : 0);
    if (t == "mid") return (Nt > 0 ? Nt / 2 : 0);
    return static_cast<std::size_t>(std::stoull(token));
}

void append_tidx_list(std::vector<std::size_t>& out,
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

void append_tidx_range(std::vector<std::size_t>& out,
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

std::vector<std::size_t> resolve_tidx_list(std::size_t Nt,
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

} // namespace

int main(int argc, char** argv) {
    using namespace taco;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);

    double dt = 0.000625;
    double beta = 0.5;
    double omega_c = 10.0;
    std::size_t Nt = 4096;
    int dim = 2;
    bool run_serial = false;
    bool run_omp = false;
    bool run_cuda = false;
    int threads = 0;
    int gpu_id = 0;
    std::string tidx_spec;
    std::string tidx_list_spec;
    std::string tidx_range_spec;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--dt=", 0) == 0) dt = std::stod(arg.substr(5));
        else if (arg.rfind("--beta=", 0) == 0) beta = std::stod(arg.substr(7));
        else if (arg.rfind("--omega_c=", 0) == 0) omega_c = std::stod(arg.substr(10));
        else if (arg.rfind("--N=", 0) == 0) Nt = static_cast<std::size_t>(std::stoull(arg.substr(4)));
        else if (arg.rfind("--dim=", 0) == 0) dim = std::stoi(arg.substr(6));
        else if (arg == "--serial") run_serial = true;
        else if (arg == "--omp") run_omp = true;
        else if (arg == "--cuda") run_cuda = true;
        else if (arg.rfind("--threads=", 0) == 0) threads = std::stoi(arg.substr(10));
        else if (arg.rfind("--gpu_id=", 0) == 0) gpu_id = std::stoi(arg.substr(9));
        else if (arg.rfind("--tidx=", 0) == 0) tidx_spec = arg.substr(7);
        else if (arg.rfind("--tidx-list=", 0) == 0) tidx_list_spec = arg.substr(12);
        else if (arg.rfind("--tidx-range=", 0) == 0) tidx_range_spec = arg.substr(13);
    }

    if (!run_serial && !run_omp && !run_cuda) {
#ifdef _OPENMP
        run_omp = true;
#else
        run_serial = true;
#endif
    }

    if (dim <= 0) {
        std::cerr << "dim must be > 0\n";
        return 1;
    }

    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) H(i, i) = static_cast<double>(i);

    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(dim, dim);
    if (dim == 1) {
        A(0, 0) = 1.0;
    } else {
        for (int i = 0; i + 1 < dim; ++i) {
            A(i, i + 1) = 1.0;
            A(i + 1, i) = 1.0;
        }
    }

    sys::System system;
    system.build(H, {A}, 1e-9);

    std::vector<double> tgrid;
    std::vector<std::complex<double>> Ccorr;
    auto J = [&](double w) { return (w > 0.0) ? (w * std::exp(-w / omega_c)) : 0.0; };
    bcf::bcf_fft_fun(Nt, dt, J, beta, tgrid, Ccorr);

    const std::size_t nf = system.fidx.buckets.size();
    std::vector<double> omegas(nf);
    for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;

    Eigen::MatrixXcd gamma_series = gamma::compute_trapz_prefix_multi_matrix(Ccorr, dt, omegas);
    const std::size_t Nt_series = static_cast<std::size_t>(gamma_series.rows());
    const std::vector<std::size_t> tidx_list =
        resolve_tidx_list(Nt_series, tidx_spec, tidx_list_spec, tidx_range_spec);

    Exec exec;
#ifdef _OPENMP
    exec.backend = Backend::Omp;
    exec.threads = threads;
#else
    exec.backend = Backend::Serial;
#endif

    auto t0 = std::chrono::high_resolution_clock::now();
    auto kernels = tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/2,
                                                tcl4::FCRMethod::Convolution, exec);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_kernels = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "kernels t=" << t_kernels << "s, nf=" << nf << ", Nt=" << Nt_series << "\n";

    tcl4::Tcl4Map map = tcl4::build_map(system, tgrid);
    const std::size_t tidx_first = tidx_list.front();
    const std::size_t tidx_last = tidx_list.back();
    std::cout << "mikx tidx_count=" << tidx_list.size()
              << " [first=" << tidx_first
              << ", last=" << tidx_last << "]\n";

#ifdef _OPENMP
    std::vector<int> thread_cases;
    if (threads > 0) {
        thread_cases = {threads};
    } else {
        thread_cases = {1, std::max(1, omp_get_max_threads() / 2), omp_get_max_threads()};
    }
#else
    std::vector<int> thread_cases = {1};
#endif

    if (run_serial) {
        double total = 0.0;
        taco::tcl4::MikxTensors mikx;
        for (std::size_t ti : tidx_list) {
            auto t2 = std::chrono::high_resolution_clock::now();
            mikx = tcl4::build_mikx_serial(map, kernels, ti);
            auto t3 = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double>(t3 - t2).count();
        }
        const double avg = total / static_cast<double>(tidx_list.size());
        std::cout << "mikx serial t_total=" << total << "s, avg=" << avg
                  << "s, n=" << tidx_list.size() << " (N=" << mikx.N << ")\n";
    }

#ifdef _OPENMP
    if (run_omp) {
        for (int th : thread_cases) {
            if (th > 0 && !omp_in_parallel()) omp_set_num_threads(th);
            double total = 0.0;
            taco::tcl4::MikxTensors mikx;
            for (std::size_t ti : tidx_list) {
                auto t2 = std::chrono::high_resolution_clock::now();
                mikx = tcl4::build_mikx_omp(map, kernels, ti);
                auto t3 = std::chrono::high_resolution_clock::now();
                total += std::chrono::duration<double>(t3 - t2).count();
            }
            const double avg = total / static_cast<double>(tidx_list.size());
            std::cout << "mikx omp threads=" << th << ", t_total=" << total
                      << "s, avg=" << avg << "s, n=" << tidx_list.size()
                      << " (N=" << mikx.N << ")\n";
        }
    }
#endif

    if (run_cuda) {
#ifdef TACO_HAS_CUDA
        Exec exec_cuda;
        exec_cuda.backend = Backend::Cuda;
        exec_cuda.gpu_id = gpu_id;
        double total = 0.0;
        double kernel_ms_total = 0.0;
        int N_out = map.N;
        if (tidx_list.size() > 1) {
            auto t2 = std::chrono::high_resolution_clock::now();
            auto mikx_list = tcl4::build_mikx_cuda_batch(map, kernels, tidx_list, exec_cuda, 0, &kernel_ms_total);
            auto t3 = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<double>(t3 - t2).count();
            if (!mikx_list.empty()) N_out = mikx_list.back().N;
            const double avg = total / static_cast<double>(tidx_list.size());
            const double kernel_total = kernel_ms_total * 1e-3;
            const double kernel_avg = kernel_total / static_cast<double>(tidx_list.size());
            std::cout << "mikx cuda batch gpu_id=" << gpu_id << ", t_total=" << total
                      << "s, t_kernel_total=" << kernel_total << "s, avg=" << avg
                      << "s, avg_kernel=" << kernel_avg << "s, n=" << tidx_list.size()
                      << " (N=" << N_out << ")\n";
        } else {
            auto t2 = std::chrono::high_resolution_clock::now();
            double kernel_ms = 0.0;
            auto mikx = tcl4::build_mikx_cuda(map, kernels, tidx_list.front(), exec_cuda, &kernel_ms);
            auto t3 = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<double>(t3 - t2).count();
            const double kernel_total = kernel_ms * 1e-3;
            std::cout << "mikx cuda gpu_id=" << gpu_id << ", t_total=" << total
                      << "s, t_kernel_total=" << kernel_total << "s, avg=" << total
                      << "s, avg_kernel=" << kernel_total << "s, n=1 (N=" << mikx.N << ")\n";
        }
#else
        std::cout << "mikx cuda requested but TACO_HAS_CUDA is not enabled in this build\n";
#endif
    }

    return 0;
}
