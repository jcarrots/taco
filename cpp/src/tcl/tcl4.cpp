#include "taco/tcl4.hpp"

#include <stdexcept>
#include <limits>
#include <cmath>

#include "taco/tcl4_kernels.hpp"
#include "taco/tcl4_mikx.hpp"
#include "taco/tcl4_assemble.hpp"

namespace taco::tcl4 {

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
                                          int /*nmax*/,
                                          FCRMethod method)
{
    const std::size_t nf = gamma_series.cols();
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("compute_triple_kernels: gamma_series column count does not match frequency buckets");
    }

    TripleKernelSeries result;
    result.F.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.C.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.R.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));

    // Precompute mirror indices using system frequencies (ω -> -ω)
    std::vector<int> mirror_idx(nf, -1);
    const double tol = std::max(1e-12, system.fidx.tol);
    for (std::size_t j = 0; j < nf; ++j) {
        const double w = system.fidx.buckets[j].omega;
        if (std::abs(w) <= tol) { mirror_idx[j] = static_cast<int>(j); continue; }
        const double target = -w;
        double best = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (std::size_t jp = 0; jp < nf; ++jp) {
            double dw = std::abs(system.fidx.buckets[jp].omega - target);
            if (dw < best) { best = dw; best_idx = static_cast<int>(jp); }
        }
        mirror_idx[j] = (best_idx >= 0 ? best_idx : static_cast<int>(j));
    }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(nf); ++ii) {
        for (std::ptrdiff_t jj = 0; jj < static_cast<std::ptrdiff_t>(nf); ++jj) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const std::size_t j = static_cast<std::size_t>(jj);
            const Eigen::VectorXcd g1col = gamma_series.col(static_cast<Eigen::Index>(i));
            const int j_mirror = mirror_idx[j];
            const Eigen::VectorXcd g2col = gamma_series.col(static_cast<Eigen::Index>(j));
            const Eigen::VectorXcd g2mcol = gamma_series.col(static_cast<Eigen::Index>(j_mirror >= 0 ? j_mirror : static_cast<int>(j)));
            for (std::size_t k = 0; k < nf; ++k) {
                double omega = system.fidx.buckets[i].omega +
                               system.fidx.buckets[j].omega +
                               system.fidx.buckets[k].omega;
                Eigen::VectorXcd Ft = compute_F_series(g1col, g2mcol, omega, dt, method);
                Eigen::VectorXcd Ct = compute_C_series(g1col, g2col, omega, dt, method);
                Eigen::VectorXcd Rt = compute_R_series(g1col, g2col, omega, dt, method);
                result.F[i][j][k] = std::move(Ft);
                result.C[i][j][k] = std::move(Ct);
                result.R[i][j][k] = std::move(Rt);
            }
        }
    }

    return result;
}

// ---------------- Convenience rebuild helpers ----------------

namespace {
inline std::size_t flat6(std::size_t N,
                         int j,int k,int p,int q,int r,int s)
{
    const std::size_t NN = N;
    return (((((static_cast<std::size_t>(j) * NN + static_cast<std::size_t>(k)) * NN
                + static_cast<std::size_t>(p)) * NN + static_cast<std::size_t>(q)) * NN
              + static_cast<std::size_t>(r)) * NN + static_cast<std::size_t>(s));
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
                                      FCRMethod method)
{
    if (time_index >= static_cast<std::size_t>(gamma_series.rows())) {
        throw std::out_of_range("build_TCL4_generator: time_index out of range");
    }
    auto kernels = compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, method);
    Tcl4Map map = build_map(system, /*time_grid*/{});
    auto mikx = build_mikx_serial(map, kernels, time_index);
    return assemble_liouvillian(mikx, system.A_eig);
}

std::vector<Eigen::MatrixXcd> build_correction_series(const sys::System& system,
                                                      const Eigen::MatrixXcd& gamma_series,
                                                      double dt,
                                                      FCRMethod method)
{
    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    std::vector<Eigen::MatrixXcd> out; out.reserve(Nt);
    auto kernels = compute_triple_kernels(system, gamma_series, dt, /*nmax*/2, method);
    Tcl4Map map = build_map(system, /*time_grid*/{});
    for (std::size_t t = 0; t < Nt; ++t) {
        auto mikx = build_mikx_serial(map, kernels, t);
        out.emplace_back(assemble_liouvillian(mikx, system.A_eig));
    }
    return out;
}

// Intentionally no combined TCL2+TCL4 builder here; see examples/tcl4_driver.cpp.
} // namespace taco::tcl4
