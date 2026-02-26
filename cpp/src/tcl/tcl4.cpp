#include "taco/tcl4.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "taco/backend/omp/tcl4_omp.hpp"
#include "taco/backend/serial/tcl4_serial.hpp"
#include "taco/tcl4_kernels.hpp"
#ifdef TACO_HAS_MPI
#include "taco/backend/cpu/tcl4_mpi_omp.hpp"
#endif
#ifdef TACO_HAS_CUDA
#include "taco/backend/cuda/tcl4_kernels_cuda.hpp"
#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"
#include "taco/backend/cuda/tcl4_assemble_cuda.hpp"
#include "taco/backend/cuda/tcl4_fused_cuda.hpp"
#endif
#include "taco/tcl4_mikx.hpp"
#include "taco/tcl4_assemble.hpp"

namespace taco::tcl4 {

namespace {
MikxTensors build_mikx_cpu(const Tcl4Map& map,
                           const TripleKernelSeries& kernels,
                           std::size_t time_index,
                           const Exec& exec)
{
#ifdef _OPENMP
    if (exec.backend == Backend::Omp) {
        return build_mikx_omp(map, kernels, time_index);
    }
#endif
    return build_mikx_serial(map, kernels, time_index);
}
} // namespace

Tcl4Map build_map(const sys::System& system, const std::vector<double>& time_grid)
{
    Tcl4Map map;
    map.N = static_cast<int>(system.eig.dim);
    map.nf = static_cast<int>(system.fidx.buckets.size());
    map.time_grid = time_grid;
    map.omegas.reserve(map.nf);

    map.pair_to_freq = Eigen::MatrixXi::Constant(map.N, map.N, -1);
    map.freq_to_pair.reserve(map.nf);
    map.mirror_index.assign(static_cast<std::size_t>(map.nf), -1);

    for (std::size_t b = 0; b < system.fidx.buckets.size(); ++b) {
        map.omegas.push_back(system.fidx.buckets[b].omega);
        map.freq_to_pair.insert(map.freq_to_pair.end(),
                                system.fidx.buckets[b].pairs.begin(),
                                system.fidx.buckets[b].pairs.end());
        for (const auto& pair : system.fidx.buckets[b].pairs) {
            map.pair_to_freq(pair.first, pair.second) = static_cast<int>(b);
        }
    }

    // Build mirror_index and locate zero bucket.
    // For each b, find b' with omegas[b'] ≈ -omegas[b]. If |w|≈0, map to itself.
    const double tol = std::max(1e-12, system.fidx.tol);
    map.zero_index = -1;
    for (int b = 0; b < map.nf; ++b) {
        const double w = map.omegas[static_cast<std::size_t>(b)];
        int best = -1;
        if (std::abs(w) <= tol) {
            best = b;
            if (map.zero_index < 0) map.zero_index = b;
        } else {
            const double target = -w;
            double best_abs = std::numeric_limits<double>::infinity();
            for (int bp = 0; bp < map.nf; ++bp) {
                const double dw = std::abs(map.omegas[static_cast<std::size_t>(bp)] - target);
                if (dw < best_abs) {
                    best_abs = dw; best = bp;
                }
            }
            if (best_abs > tol) {
                // no clean mirror found within tolerance; leave best as closest match
            }
        }
        map.mirror_index[static_cast<std::size_t>(b)] = best;
    }

    // Sanity: zero bucket must exist due to diagonal ω_{mm}=0
    if (map.zero_index < 0) {
        // Fallback: find closest to zero
        double best_abs = std::numeric_limits<double>::infinity();
        int best = -1;
        for (int b = 0; b < map.nf; ++b) {
            double a = std::abs(map.omegas[static_cast<std::size_t>(b)]);
            if (a < best_abs) { best_abs = a; best = b; }
        }
        map.zero_index = (best >= 0 ? best : 0);
    }

    return map;
}

TripleKernelSeries compute_triple_kernels(const sys::System& system,
                                          const Eigen::MatrixXcd& gamma_series,
                                          double dt,
                                          int nmax,
                                          FCRMethod method,
                                          Exec exec)
{
    if (exec.backend == Backend::Cuda) {
        #ifdef TACO_HAS_CUDA
        return compute_triple_kernels_cuda(system, gamma_series, dt, nmax, method, exec);
        #else
        throw std::invalid_argument("compute_triple_kernels: CUDA backend requested but taco_tcl was built without CUDA");
        #endif
    }
    if (exec.backend == Backend::Omp) {
        return compute_triple_kernels_omp(system, gamma_series, dt, nmax, method, exec);
    }
    if (exec.backend == Backend::Serial) {
        return compute_triple_kernels_serial(system, gamma_series, dt, nmax, method, exec);
    }
    throw std::invalid_argument("compute_triple_kernels: unsupported backend (expected Serial/Omp/Cuda)");
}

// ---------------- Convenience rebuild helpers ----------------

namespace {
inline std::size_t flat6(std::size_t N,
                         int j,int k,int p,int q,int r,int s)
{
    const std::size_t NN = N;
    return static_cast<std::size_t>(j) +
           NN * (static_cast<std::size_t>(k) +
           NN * (static_cast<std::size_t>(p) +
           NN * (static_cast<std::size_t>(q) +
           NN * (static_cast<std::size_t>(r) +
           NN * static_cast<std::size_t>(s)))));
}
} // namespace

Eigen::MatrixXcd build_gamma_matrix_at(const Tcl4Map& map,
                                       const Eigen::MatrixXcd& gamma_series,
                                       std::size_t time_index)
{
    if (time_index >= static_cast<std::size_t>(gamma_series.rows())) {
        throw std::out_of_range("build_gamma_matrix_at: time_index out of range");
    }
    const std::size_t N = static_cast<std::size_t>(map.N);
    Eigen::MatrixXcd G(static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(N));
    for (int j = 0; j < map.N; ++j) {
        for (int k = 0; k < map.N; ++k) {
            const int b = map.pair_to_freq(j, k);
            if (b < 0) { G(j,k) = 0.0; continue; }
            G(j,k) = gamma_series(static_cast<Eigen::Index>(time_index), static_cast<Eigen::Index>(b));
        }
    }
    return G;
}

void build_FCR_6d_at(const Tcl4Map& map,
                     const TripleKernelSeries& kernels,
                     std::size_t time_index,
                     std::vector<std::complex<double>>& F_out,
                     std::vector<std::complex<double>>& C_out,
                     std::vector<std::complex<double>>& R_out)
{
    if (map.N <= 0) throw std::invalid_argument("build_FCR_6d_at: map.N must be > 0");
    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t total = N*N*N*N*N*N;
    F_out.assign(total, std::complex<double>(0.0, 0.0));
    C_out.assign(total, std::complex<double>(0.0, 0.0));
    R_out.assign(total, std::complex<double>(0.0, 0.0));

    for (int j = 0; j < map.N; ++j) {
        for (int k = 0; k < map.N; ++k) {
            const int f1 = map.pair_to_freq(j, k);
            if (f1 < 0) continue;
            for (int p = 0; p < map.N; ++p) {
                for (int q = 0; q < map.N; ++q) {
                    const int f2 = map.pair_to_freq(p, q);
                    if (f2 < 0) continue;
                    for (int r = 0; r < map.N; ++r) {
                        for (int s = 0; s < map.N; ++s) {
                            const int f3 = map.pair_to_freq(r, s);
                            if (f3 < 0) continue;
                            const auto idx = flat6(N, j,k,p,q,r,s);
                            F_out[idx] = kernels.F[static_cast<std::size_t>(f1)][static_cast<std::size_t>(f2)][static_cast<std::size_t>(f3)](static_cast<Eigen::Index>(time_index));
                            C_out[idx] = kernels.C[static_cast<std::size_t>(f1)][static_cast<std::size_t>(f2)][static_cast<std::size_t>(f3)](static_cast<Eigen::Index>(time_index));
                            R_out[idx] = kernels.R[static_cast<std::size_t>(f1)][static_cast<std::size_t>(f2)][static_cast<std::size_t>(f3)](static_cast<Eigen::Index>(time_index));
                        }
                    }
                }
            }
        }
    }
}

void build_FCR_6d_final(const Tcl4Map& map,
                        const TripleKernelSeries& kernels,
                        std::vector<std::complex<double>>& F_out,
                        std::vector<std::complex<double>>& C_out,
                        std::vector<std::complex<double>>& R_out)
{
    // Use the last available time index from any one entry (assume consistent length)
    if (kernels.F.empty() || kernels.F.front().empty() || kernels.F.front().front().size() == 0) {
        F_out.clear(); C_out.clear(); R_out.clear(); return;
    }
    const auto& v = kernels.F.front().front().front();
    const std::size_t last = static_cast<std::size_t>(std::max<Eigen::Index>(0, v.size() - 1));
    build_FCR_6d_at(map, kernels, last, F_out, C_out, R_out);
}

void build_FCR_6d_series(const Tcl4Map& map,
                         const TripleKernelSeries& kernels,
                         std::vector<std::vector<std::complex<double>>>& F_series,
                         std::vector<std::vector<std::complex<double>>>& C_series,
                         std::vector<std::vector<std::complex<double>>>& R_series)
{
    // Deduce Nt from any one F entry
    if (kernels.F.empty() || kernels.F.front().empty() || kernels.F.front().front().size() == 0) {
        F_series.clear(); C_series.clear(); R_series.clear(); return;
    }
    const auto& v = kernels.F.front().front().front();
    const std::size_t Nt = static_cast<std::size_t>(v.size());
    F_series.resize(Nt);
    C_series.resize(Nt);
    R_series.resize(Nt);
    for (std::size_t t = 0; t < Nt; ++t) {
        build_FCR_6d_at(map, kernels, t, F_series[t], C_series[t], R_series[t]);
    }
}

Eigen::MatrixXcd build_TCL4_generator(const sys::System& system,
                                      const Eigen::MatrixXcd& gamma_series,
                                      double dt,
                                      std::size_t time_index,
                                      FCRMethod method,
                                      Exec exec)
{
    if (time_index >= static_cast<std::size_t>(gamma_series.rows())) {
        throw std::out_of_range("build_TCL4_generator: time_index out of range");
    }

    if (exec.backend == Backend::Cuda) {
        #ifdef TACO_HAS_CUDA
        return build_TCL4_generator_cuda_fused(system, gamma_series, dt, time_index, method, exec);
        #else
        throw std::invalid_argument("build_TCL4_generator: CUDA backend requested but taco_tcl was built without CUDA");
        #endif
    }

    if (exec.backend == Backend::MpiOmp) {
        #ifdef TACO_HAS_MPI
        // NOTE: the current MPI decomposition is over time indices. For a single time index, only
        // the root rank does work; broadcast the result so the return value is valid on all ranks.
        const auto out = build_TCL4_generator_cpu_mpi_omp_batch(system,
                                                                gamma_series,
                                                                dt,
                                                                /*time_indices=*/{time_index},
                                                                method,
                                                                MPI_COMM_WORLD,
                                                                exec);

        int rank = 0;
        int size = 1;
        if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
            throw std::runtime_error("build_TCL4_generator: MPI_Comm_rank failed");
        }
        if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
            throw std::runtime_error("build_TCL4_generator: MPI_Comm_size failed");
        }

        const std::size_t N = static_cast<std::size_t>(system.eig.dim);
        const std::size_t N2 = N * N;
        const std::size_t elems_per = N2 * N2;
        const std::size_t scalar_bytes = sizeof(Eigen::MatrixXcd::Scalar);

        Eigen::MatrixXcd L4;
        if (rank == 0) {
            if (out.size() != 1) {
                throw std::runtime_error("build_TCL4_generator: MPI root did not return L4");
            }
            L4 = out[0];
        } else {
            L4.resize(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        }

        if (L4.rows() != static_cast<Eigen::Index>(N2) || L4.cols() != static_cast<Eigen::Index>(N2)) {
            throw std::runtime_error("build_TCL4_generator: unexpected L4 shape in MPI backend");
        }

        if (elems_per > 0 && scalar_bytes > (std::numeric_limits<std::size_t>::max() / elems_per)) {
            throw std::overflow_error("build_TCL4_generator: MPI broadcast buffer too large");
        }
        const std::size_t bytes_u = elems_per * scalar_bytes;
        if (bytes_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("build_TCL4_generator: MPI broadcast buffer too large");
        }
        const int bytes = static_cast<int>(bytes_u);

        if (size > 1) {
            if (MPI_Bcast(reinterpret_cast<void*>(L4.data()), bytes, MPI_BYTE, /*root=*/0, MPI_COMM_WORLD) !=
                MPI_SUCCESS) {
                throw std::runtime_error("build_TCL4_generator: MPI_Bcast failed");
            }
        }

        return L4;
        #else
        throw std::invalid_argument("build_TCL4_generator: MPI backend requested but taco_tcl was built without MPI");
        #endif
    }

    auto kernels = compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, method, exec);
    Tcl4Map map = build_map(system, /*time_grid*/{});

    MikxTensors mikx;
    Eigen::MatrixXcd GW;
    mikx = build_mikx_cpu(map, kernels, time_index, exec);
    GW = assemble_liouvillian(mikx, system.A_eig); // (n,i;m,j)
    return gw_to_liouvillian(GW, system.eig.dim);                         // (n,m;i,j)
}

std::vector<Eigen::MatrixXcd> build_correction_series(const sys::System& system,
                                                      const Eigen::MatrixXcd& gamma_series,
                                                      double dt,
                                                      FCRMethod method,
                                                      Exec exec)
{
    if (exec.backend == Backend::MpiOmp) {
        #ifdef TACO_HAS_MPI
        return build_TCL4_generator_cpu_mpi_omp_batch(system,
                                                      gamma_series,
                                                      dt,
                                                      /*time_indices=*/{},
                                                      method,
                                                      MPI_COMM_WORLD,
                                                      exec);
        #else
        throw std::invalid_argument("build_correction_series: MPI backend requested but taco_tcl was built without MPI");
        #endif
    }

    if (exec.backend == Backend::Omp) {
        const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
        auto kernels = compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, method, exec);
        Tcl4Map map = build_map(system, /*time_grid*/{});
        return build_correction_series_omp(system, kernels, map, Nt, exec);
    }
    if (exec.backend == Backend::Serial) {
        const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
        auto kernels = compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, method, exec);
        Tcl4Map map = build_map(system, /*time_grid*/{});
        return build_correction_series_serial(system, kernels, map, Nt, exec);
    }
    if (exec.backend != Backend::Cuda) {
        throw std::invalid_argument("build_correction_series: unsupported backend (expected Serial/Omp/Cuda/MpiOmp)");
    }

#ifdef TACO_HAS_CUDA
    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    std::vector<Eigen::MatrixXcd> out(Nt);
    auto kernels = compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, method, exec);
    Tcl4Map map = build_map(system, /*time_grid*/{});

    for (std::ptrdiff_t tt = 0; tt < static_cast<std::ptrdiff_t>(Nt); ++tt) {
        const std::size_t t = static_cast<std::size_t>(tt);
        auto mikx = build_mikx_cuda(map, kernels, t, exec);
        const Eigen::MatrixXcd GW = assemble_liouvillian_cuda(mikx, system.A_eig, exec); // (n,i;m,j)
        out[t] = gw_to_liouvillian(GW, system.eig.dim);                                   // (n,m;i,j)
    }
    return out;
#else
    throw std::invalid_argument("build_correction_series: CUDA backend requested but taco_tcl was built without CUDA");
#endif
}

// Intentionally no combined TCL2+TCL4 builder here; see examples/tcl_driver.cpp.
} // namespace taco::tcl4
