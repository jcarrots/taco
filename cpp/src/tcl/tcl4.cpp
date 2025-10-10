#include "taco/tcl4.hpp"

#include <stdexcept>

#include "taco/tcl4_kernels.hpp"

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

    for (std::size_t b = 0; b < system.fidx.buckets.size(); ++b) {
        map.omegas.push_back(system.fidx.buckets[b].omega);
        map.freq_to_pair.insert(map.freq_to_pair.end(),
                                system.fidx.buckets[b].pairs.begin(),
                                system.fidx.buckets[b].pairs.end());
        for (const auto& pair : system.fidx.buckets[b].pairs) {
            map.pair_to_freq(pair.first, pair.second) = static_cast<int>(b);
        }
    }

    return map;
}

TripleKernelSeries compute_triple_kernels(const sys::System& system,
                                          const Eigen::MatrixXcd& gamma_series,
                                          double dt,
                                          int /*nmax*/)
{
    const std::size_t Nt = gamma_series.rows();
    const std::size_t nf = gamma_series.cols();
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("compute_triple_kernels: gamma_series column count does not match frequency buckets");
    }

    TripleKernelSeries result;
    result.F.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.C.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    result.R.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));

    std::vector<Eigen::MatrixXcd> g1(Nt, Eigen::MatrixXcd(1,1));
    std::vector<Eigen::MatrixXcd> g2(Nt, Eigen::MatrixXcd(1,1));

    for (std::size_t i = 0; i < nf; ++i) {
        for (std::size_t j = 0; j < nf; ++j) {
            for (std::size_t t = 0; t < Nt; ++t) {
                g1[t](0,0) = gamma_series(t, static_cast<Eigen::Index>(i));
                g2[t](0,0) = gamma_series(t, static_cast<Eigen::Index>(j));
            }
            for (std::size_t k = 0; k < nf; ++k) {
                double omega = system.fidx.buckets[i].omega +
                               system.fidx.buckets[j].omega +
                               system.fidx.buckets[k].omega;
                auto fcr = compute_FCR_time_series(g1, g2, omega, dt, SpectralOp::Transpose);
                Eigen::VectorXcd Ft(Nt), Ct(Nt), Rt(Nt);
                for (std::size_t t = 0; t < Nt; ++t) {
                    Ft(static_cast<Eigen::Index>(t)) = fcr.F[t](0,0);
                    Ct(static_cast<Eigen::Index>(t)) = fcr.C[t](0,0);
                    Rt(static_cast<Eigen::Index>(t)) = fcr.R[t](0,0);
                }
                result.F[i][j][k] = std::move(Ft);
                result.C[i][j][k] = std::move(Ct);
                result.R[i][j][k] = std::move(Rt);
            }
        }
    }

    return result;
}

} // namespace taco::tcl4
