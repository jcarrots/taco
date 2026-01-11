#include "taco/backend/cpu/tcl4_mpi_omp.hpp"

#ifdef TACO_HAS_MPI

#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_mikx.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace taco::tcl4 {

namespace {

struct MpiInfo {
    int rank{0};
    int size{1};
};

[[noreturn]] void throw_mpi_error(int err, const char* what) {
    char buf[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(err, buf, &len);
    throw std::runtime_error(std::string(what) + ": " + std::string(buf, buf + len));
}

inline void mpi_check(int err, const char* what) {
    if (err == MPI_SUCCESS) return;
    throw_mpi_error(err, what);
}

MpiInfo get_mpi_info(MPI_Comm comm) {
    int initialized = 0;
    mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized) {
        throw std::runtime_error("build_TCL4_generator_cpu_mpi_omp_batch: MPI is not initialized (call MPI_Init)");
    }
    if (comm == MPI_COMM_NULL) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: comm is MPI_COMM_NULL");
    }

    MpiInfo info;
    mpi_check(MPI_Comm_rank(comm, &info.rank), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(comm, &info.size), "MPI_Comm_size");
    if (info.size <= 0) {
        throw std::runtime_error("build_TCL4_generator_cpu_mpi_omp_batch: invalid MPI communicator size");
    }
    return info;
}

void decompose_contiguous(std::size_t n,
                          int comm_size,
                          int comm_rank,
                          std::size_t& start,
                          std::size_t& count) {
    if (comm_size <= 0) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: invalid comm size");
    }
    if (comm_rank < 0 || comm_rank >= comm_size) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: invalid comm rank");
    }
    const std::size_t size_u = static_cast<std::size_t>(comm_size);
    const std::size_t rank_u = static_cast<std::size_t>(comm_rank);

    const std::size_t base = n / size_u;
    const std::size_t rem = n % size_u;
    count = base + (rank_u < rem ? 1 : 0);
    start = base * rank_u + std::min(rank_u, rem);
}

TripleKernelSeries compute_triple_kernels_sliced(const sys::System& system,
                                                 const Eigen::MatrixXcd& gamma_series,
                                                 double dt,
                                                 FCRMethod method,
                                                 const std::vector<std::size_t>& time_indices,
                                                 const std::vector<int>& mirror_idx)
{
    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    const std::size_t nf = static_cast<std::size_t>(gamma_series.cols());

    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: gamma_series column count does not match frequency buckets");
    }
    if (mirror_idx.size() != nf) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: mirror index list has wrong size");
    }

    const std::size_t num_times = time_indices.size();
    std::vector<Eigen::Index> tids_e;
    tids_e.reserve(num_times);
    for (std::size_t t : time_indices) {
        if (t >= Nt) {
            throw std::out_of_range("build_TCL4_generator_cpu_mpi_omp_batch: time_index out of range");
        }
        tids_e.push_back(static_cast<Eigen::Index>(t));
    }

    TripleKernelSeries result;
    result.F.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.C.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.R.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));

    const std::ptrdiff_t nf_i = static_cast<std::ptrdiff_t>(nf);
    const std::ptrdiff_t total = nf_i * nf_i;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    #endif
    for (std::ptrdiff_t idx = 0; idx < total; ++idx) {
        const std::size_t i = static_cast<std::size_t>(idx / nf_i);
        const std::size_t j = static_cast<std::size_t>(idx % nf_i);

        const auto g1col = gamma_series.col(static_cast<Eigen::Index>(i));
        const auto g2col = gamma_series.col(static_cast<Eigen::Index>(j));
        const int j_mirror = mirror_idx[j];
        const auto g2mcol =
            gamma_series.col(static_cast<Eigen::Index>(j_mirror >= 0 ? j_mirror : static_cast<int>(j)));

        const double wi = system.fidx.buckets[i].omega;
        const double wj = system.fidx.buckets[j].omega;
        for (std::size_t k = 0; k < nf; ++k) {
            const double omega = wi + wj + system.fidx.buckets[k].omega;

            Eigen::VectorXcd Ft_full = compute_F_series(g1col, g2mcol, omega, dt, method);
            Eigen::VectorXcd Ct_full = compute_C_series(g1col, g2col, omega, dt, method);
            Eigen::VectorXcd Rt_full = compute_R_series(g1col, g2col, omega, dt, method);

            Eigen::VectorXcd Ft(static_cast<Eigen::Index>(num_times));
            Eigen::VectorXcd Ct(static_cast<Eigen::Index>(num_times));
            Eigen::VectorXcd Rt(static_cast<Eigen::Index>(num_times));
            for (std::size_t tpos = 0; tpos < num_times; ++tpos) {
                const Eigen::Index ti = tids_e[tpos];
                Ft(static_cast<Eigen::Index>(tpos)) = Ft_full(ti);
                Ct(static_cast<Eigen::Index>(tpos)) = Ct_full(ti);
                Rt(static_cast<Eigen::Index>(tpos)) = Rt_full(ti);
            }

            result.F[i][j][k] = std::move(Ft);
            result.C[i][j][k] = std::move(Ct);
            result.R[i][j][k] = std::move(Rt);
        }
    }

    return result;
}

} // namespace

std::vector<Eigen::MatrixXcd>
build_TCL4_generator_cpu_mpi_omp_batch(const sys::System& system,
                                       const Eigen::MatrixXcd& gamma_series,
                                       double dt,
                                       const std::vector<std::size_t>& time_indices,
                                       FCRMethod method,
                                       MPI_Comm comm)
{
    if (dt <= 0.0) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: dt must be > 0");
    }

    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    const std::size_t nf = static_cast<std::size_t>(gamma_series.cols());
    if (Nt == 0 || nf == 0) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: empty gamma_series");
    }
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: gamma_series column count does not match frequency buckets");
    }

    std::vector<std::size_t> tids = time_indices;
    if (tids.empty()) {
        tids.resize(Nt);
        std::iota(tids.begin(), tids.end(), std::size_t{0});
    }
    for (std::size_t t : tids) {
        if (t >= Nt) {
            throw std::out_of_range("build_TCL4_generator_cpu_mpi_omp_batch: time_index out of range");
        }
    }
    if (tids.empty()) {
        return {};
    }

    Tcl4Map map = build_map(system, /*time_grid*/{});
    if (map.N <= 0) throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: map.N must be > 0");
    if (map.nf <= 0) throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: map.nf must be > 0");
    if (static_cast<std::size_t>(map.nf) != nf) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: map.nf mismatch");
    }
    if (map.pair_to_freq.rows() != static_cast<Eigen::Index>(map.N) ||
        map.pair_to_freq.cols() != static_cast<Eigen::Index>(map.N)) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: map.pair_to_freq has wrong shape");
    }
    if (map.pair_to_freq.minCoeff() < 0) {
        throw std::runtime_error("build_TCL4_generator_cpu_mpi_omp_batch: map.pair_to_freq contains -1 (missing frequency buckets)");
    }
    if (system.A_eig.empty()) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: coupling_ops must be non-empty");
    }

    const MpiInfo mpi = get_mpi_info(comm);

    std::size_t local_start = 0;
    std::size_t local_count = 0;
    decompose_contiguous(tids.size(), mpi.size, mpi.rank, local_start, local_count);

    std::vector<std::size_t> local_tids;
    local_tids.reserve(local_count);
    for (std::size_t i = 0; i < local_count; ++i) {
        local_tids.push_back(tids[local_start + i]);
    }

    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t N2 = N * N;
    const std::size_t elems_per = N2 * N2;
    if (elems_per == 0) {
        throw std::invalid_argument("build_TCL4_generator_cpu_mpi_omp_batch: invalid system dimension");
    }

    std::vector<std::complex<double>> local_L4;
    local_L4.resize(local_count * elems_per);

    if (local_count > 0) {
        TripleKernelSeries kernels =
            compute_triple_kernels_sliced(system, gamma_series, dt, method, local_tids, map.mirror_index);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(!omp_in_parallel())
        #endif
        for (std::ptrdiff_t tt = 0; tt < static_cast<std::ptrdiff_t>(local_count); ++tt) {
            const std::size_t tpos = static_cast<std::size_t>(tt);
            const auto mikx = build_mikx_serial(map, kernels, tpos);
            const Eigen::MatrixXcd GW = assemble_liouvillian(mikx, system.A_eig);
            const Eigen::MatrixXcd L4 = gw_to_liouvillian(GW, N);
            std::complex<double>* dst = local_L4.data() + tpos * elems_per;
            std::copy(L4.data(), L4.data() + static_cast<Eigen::Index>(elems_per), dst);
        }
    }

    if (mpi.size == 1) {
        std::vector<Eigen::MatrixXcd> out_series;
        out_series.resize(tids.size());
        for (std::size_t idx = 0; idx < tids.size(); ++idx) {
            Eigen::MatrixXcd L4(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
            const std::complex<double>* src = local_L4.data() + idx * elems_per;
            std::copy(src, src + elems_per, L4.data());
            out_series[idx] = std::move(L4);
        }
        return out_series;
    }

    const std::size_t complex_bytes = sizeof(std::complex<double>);
    if (elems_per > 0 && local_count > (std::numeric_limits<std::size_t>::max() / elems_per)) {
        throw std::overflow_error("build_TCL4_generator_cpu_mpi_omp_batch: local output too large");
    }
    if (elems_per > 0 && tids.size() > (std::numeric_limits<std::size_t>::max() / elems_per)) {
        throw std::overflow_error("build_TCL4_generator_cpu_mpi_omp_batch: global output too large");
    }

    const std::size_t local_bytes_u = local_count * elems_per * complex_bytes;
    if (local_bytes_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("build_TCL4_generator_cpu_mpi_omp_batch: MPI send buffer too large");
    }
    const int local_bytes = static_cast<int>(local_bytes_u);

    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<std::complex<double>> all_L4;
    if (mpi.rank == 0) {
        recv_counts.assign(static_cast<std::size_t>(mpi.size), 0);
        recv_displs.assign(static_cast<std::size_t>(mpi.size), 0);

        std::size_t disp_bytes = 0;
        for (int r = 0; r < mpi.size; ++r) {
            std::size_t rs = 0;
            std::size_t rc = 0;
            decompose_contiguous(tids.size(), mpi.size, r, rs, rc);

            const std::size_t bytes_u = rc * elems_per * complex_bytes;
            if (bytes_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error("build_TCL4_generator_cpu_mpi_omp_batch: MPI recv buffer too large");
            }
            if (disp_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error("build_TCL4_generator_cpu_mpi_omp_batch: MPI displacement too large");
            }
            recv_counts[static_cast<std::size_t>(r)] = static_cast<int>(bytes_u);
            recv_displs[static_cast<std::size_t>(r)] = static_cast<int>(disp_bytes);
            disp_bytes += bytes_u;
        }

        all_L4.resize(tids.size() * elems_per);
    }

    mpi_check(MPI_Gatherv(reinterpret_cast<const void*>(local_L4.data()),
                          local_bytes,
                          MPI_BYTE,
                          reinterpret_cast<void*>(all_L4.data()),
                          recv_counts.empty() ? nullptr : recv_counts.data(),
                          recv_displs.empty() ? nullptr : recv_displs.data(),
                          MPI_BYTE,
                          /*root=*/0,
                          comm),
              "MPI_Gatherv(L4)");

    if (mpi.rank != 0) {
        return {};
    }

    std::vector<Eigen::MatrixXcd> out_series;
    out_series.resize(tids.size());
    for (std::size_t idx = 0; idx < tids.size(); ++idx) {
        Eigen::MatrixXcd L4(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        const std::complex<double>* src = all_L4.data() + idx * elems_per;
        std::copy(src, src + elems_per, L4.data());
        out_series[idx] = std::move(L4);
    }
    return out_series;
}

} // namespace taco::tcl4

#endif // TACO_HAS_MPI

