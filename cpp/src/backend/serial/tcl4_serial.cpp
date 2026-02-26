#include "taco/backend/serial/tcl4_serial.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_kernels.hpp"
#include "taco/tcl4_mikx.hpp"

namespace taco::tcl4 {

TripleKernelSeries compute_triple_kernels_serial(const sys::System& system,
                                                 const Eigen::MatrixXcd& gamma_series,
                                                 double dt,
                                                 int nmax,
                                                 FCRMethod method,
                                                 Exec exec)
{
    (void)nmax;
    if (exec.backend != Backend::Serial) {
        throw std::invalid_argument("compute_triple_kernels_serial: expected Backend::Serial");
    }

    const std::size_t nf = static_cast<std::size_t>(gamma_series.cols());
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("compute_triple_kernels_serial: gamma_series column count does not match frequency buckets");
    }

    TripleKernelSeries result;
    result.F.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.C.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.R.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));

    std::vector<int> mirror_idx(nf, -1);
    const double tol = std::max(1e-12, system.fidx.tol);
    for (std::size_t j = 0; j < nf; ++j) {
        const double w = system.fidx.buckets[j].omega;
        if (std::abs(w) <= tol) {
            mirror_idx[j] = static_cast<int>(j);
            continue;
        }

        const double target = -w;
        double best = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (std::size_t jp = 0; jp < nf; ++jp) {
            const double dw = std::abs(system.fidx.buckets[jp].omega - target);
            if (dw < best) {
                best = dw;
                best_idx = static_cast<int>(jp);
            }
        }
        mirror_idx[j] = (best_idx >= 0 ? best_idx : static_cast<int>(j));
    }

    const std::ptrdiff_t nf_i = static_cast<std::ptrdiff_t>(nf);
    const std::ptrdiff_t total = nf_i * nf_i;
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
            Eigen::VectorXcd Ft = compute_F_series(g1col, g2mcol, omega, dt, method);
            Eigen::VectorXcd Ct = compute_C_series(g1col, g2col, omega, dt, method);
            Eigen::VectorXcd Rt = compute_R_series(g1col, g2col, omega, dt, method);
            result.F[i][j][k] = std::move(Ft);
            result.C[i][j][k] = std::move(Ct);
            result.R[i][j][k] = std::move(Rt);
        }
    }

    return result;
}

std::vector<Eigen::MatrixXcd> build_correction_series_serial(const sys::System& system,
                                                             const TripleKernelSeries& kernels,
                                                             const Tcl4Map& map,
                                                             std::size_t Nt,
                                                             Exec exec)
{
    if (exec.backend != Backend::Serial) {
        throw std::invalid_argument("build_correction_series_serial: expected Backend::Serial");
    }

    std::vector<Eigen::MatrixXcd> out(Nt);
    for (std::size_t t = 0; t < Nt; ++t) {
        const auto mikx = build_mikx_serial(map, kernels, t);
        const Eigen::MatrixXcd GW = assemble_liouvillian(mikx, system.A_eig);
        out[t] = gw_to_liouvillian(GW, system.eig.dim);
    }

    return out;
}

} // namespace taco::tcl4

