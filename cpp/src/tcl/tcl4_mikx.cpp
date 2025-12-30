#include "taco/tcl4_mikx.hpp"

#include <cstddef>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace taco::tcl4 {

namespace {

inline std::size_t flat6(std::size_t N,
                         int j,int k,int p,int q,int r,int s)
{
    // Column-major index for 6D tensor of size N in each dim (first index varies fastest)
    const std::size_t NN = static_cast<std::size_t>(N);
    return static_cast<std::size_t>(j) +
           NN * (static_cast<std::size_t>(k) +
           NN * (static_cast<std::size_t>(p) +
           NN * (static_cast<std::size_t>(q) +
           NN * (static_cast<std::size_t>(r) +
           NN * static_cast<std::size_t>(s)))));
}

// Column-major pair flattening: idx(row,col) = row + col*N
inline std::size_t flat2(std::size_t N, int row, int col)
{
    return static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * N;
}

inline std::size_t freq_idx_checked(const Eigen::MatrixXi& pair_to_freq,
                                    int a, int b)
{
    int idx = pair_to_freq(a, b);
    if (idx < 0) {
        throw std::runtime_error("tcl4::build_mikx_serial: pair_to_freq contains -1 for a required pair");
    }
    return static_cast<std::size_t>(idx);
}

} // namespace

MikxTensors build_mikx_serial(const Tcl4Map& map,
                              const TripleKernelSeries& kernels,
                              std::size_t time_index)
{
    MikxTensors tensors;
    tensors.N = map.N;

    if (map.N <= 0) {
        throw std::invalid_argument("build_mikx_serial: map.N must be > 0");
    }

    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t N2 = N * N;

    // Validate time_index against any one series (if available)
    if (!kernels.F.empty() && !kernels.F.front().empty() && !kernels.F.front().front().empty()) {
        const auto& v = kernels.F.front().front().front();
        if (time_index >= static_cast<std::size_t>(v.size())) {
            throw std::out_of_range("build_mikx_serial: time_index out of range for kernel series");
        }
    }

    tensors.M = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
    tensors.I = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
    tensors.K = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));

    // Pre-size X (N^6) and fill below
    std::size_t totalX = 1;
    for (int d = 0; d < 6; ++d) totalX *= N;
    tensors.X.assign(totalX, std::complex<double>(0.0, 0.0));

    // Convenience alias
    const auto& F = kernels.F;
    const auto& C = kernels.C;
    const auto& R = kernels.R;

    // Build M, I, K, X with explicit contractions mirroring MIKX.m.
    // Parallelize over (j,k,p,q) coefficients (N^4 tasks) for better scaling on small N.
    const std::size_t N4 = N2 * N2;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    #endif
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(N4); ++idx) {
        const std::size_t row_u = static_cast<std::size_t>(idx) / N2; // (j,k) flattened column-major
        const std::size_t col_u = static_cast<std::size_t>(idx) % N2; // (p,q) flattened column-major

        const int j = static_cast<int>(row_u % N);
        const int k = static_cast<int>(row_u / N);
        const int p = static_cast<int>(col_u % N);
        const int q = static_cast<int>(col_u / N);

        const auto row = static_cast<Eigen::Index>(row_u);
        const auto col = static_cast<Eigen::Index>(col_u);

        const auto f_jk = freq_idx_checked(map.pair_to_freq, j, k);
        const auto f_jq = freq_idx_checked(map.pair_to_freq, j, q);
        const auto f_pj = freq_idx_checked(map.pair_to_freq, p, j);
        const auto f_pq = freq_idx_checked(map.pair_to_freq, p, q);
        const auto f_qk = freq_idx_checked(map.pair_to_freq, q, k);
        const auto f_kq = freq_idx_checked(map.pair_to_freq, k, q);
        const auto f_qp = freq_idx_checked(map.pair_to_freq, q, p);
        const auto f_qj = freq_idx_checked(map.pair_to_freq, q, j);

        // M = M1 - M2
        // M1(j,k,p,q) = A(j,q,j,k,p,j) -> F[f(j,q)][f(j,k)][f(p,j)]
        // M2(j,k,p,q) = B(j,q,p,q,q,k) -> R[f(j,q)][f(p,q)][f(q,k)]
        const std::complex<double> M1 = F[f_jq][f_jk][f_pj](static_cast<Eigen::Index>(time_index));
        const std::complex<double> M2 = R[f_jq][f_pq][f_qk](static_cast<Eigen::Index>(time_index));
        tensors.M(row, col) = M1 - M2;

        // I(j,k,p,q) = A(j,k,q,p,k,q) -> F[f(j,k)][f(q,p)][f(k,q)]
        tensors.I(row, col) = F[f_jk][f_qp][f_kq](static_cast<Eigen::Index>(time_index));

        // K(j,k,p,q) = B(j,k,p,q,q,j) -> R[f(j,k)][f(p,q)][f(q,j)]
        tensors.K(row, col) = R[f_jk][f_pq][f_qj](static_cast<Eigen::Index>(time_index));

        // Fill X for all r,s at fixed (j,k,p,q)
        for (int r = 0; r < map.N; ++r) {
            for (int s = 0; s < map.N; ++s) {
                const auto f_rs = freq_idx_checked(map.pair_to_freq, r, s);
                const std::complex<double> Xval =
                    C[f_jk][f_pq][f_rs](static_cast<Eigen::Index>(time_index)) +
                    R[f_jk][f_pq][f_rs](static_cast<Eigen::Index>(time_index));
                tensors.X[flat6(map.N, j, k, p, q, r, s)] = Xval;
            }
        }
    }

    return tensors;
}

} // namespace taco::tcl4
