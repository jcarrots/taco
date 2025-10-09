// bath_tabulated.hpp — Tabulated correlation functions with linear interpolation

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "taco/bath.hpp"
#include "taco/correlation_fft.hpp"

namespace taco::bath {

class TabulatedCorrelation final : public CorrelationFunction {
public:
    // C[alpha][beta][k] corresponds to time t[k]
    TabulatedCorrelation(std::vector<double> t,
                         std::vector<std::vector<std::vector<std::complex<double>>>> C)
        : t_(std::move(t)), C_(std::move(C))
    {
        if (t_.empty()) throw std::invalid_argument("TabulatedCorrelation: empty time grid");
        const std::size_t r = C_.size();
        if (r == 0) throw std::invalid_argument("TabulatedCorrelation: rank must be > 0");
        for (const auto& row : C_) {
            if (row.size() != r) throw std::invalid_argument("TabulatedCorrelation: C must be r x r blocks");
            for (const auto& vec : row) {
                if (vec.size() != t_.size()) throw std::invalid_argument("TabulatedCorrelation: block length must match t.size()");
            }
        }
        if (!std::is_sorted(t_.begin(), t_.end())) throw std::invalid_argument("TabulatedCorrelation: t must be sorted ascending");
        rank_ = r;
    }

    std::complex<double> operator()(double tau, std::size_t alpha, std::size_t beta) const override {
        if (alpha >= rank_ || beta >= rank_) return {0.0, 0.0};
        if (!(tau >= 0.0)) return {0.0, 0.0};
        if (tau > t_.back()) return {0.0, 0.0}; // zero outside support
        auto it = std::lower_bound(t_.begin(), t_.end(), tau);
        if (it == t_.begin()) return C_[alpha][beta][0];
        if (it == t_.end()) return {0.0, 0.0};
        std::size_t k = static_cast<std::size_t>(it - t_.begin());
        const double t1 = t_[k-1], t2 = t_[k];
        const double w = (tau - t1) / (t2 - t1);
        return (1.0 - w) * C_[alpha][beta][k-1] + w * C_[alpha][beta][k];
    }

    std::size_t rank() const noexcept override { return rank_; }

    const std::vector<double>& times() const noexcept { return t_; }
    const auto& data() const noexcept { return C_; }

    // Convenience: build diagonal r×r correlation from a single scalar C(t)
    static TabulatedCorrelation diagonal(std::size_t r,
                                         std::vector<double> t,
                                         std::vector<std::complex<double>> Cdiag) {
        if (r == 0) throw std::invalid_argument("TabulatedCorrelation::diagonal: rank must be > 0");
        if (t.size() != Cdiag.size()) throw std::invalid_argument("diagonal: t and C must have same length");
        std::vector<std::vector<std::vector<std::complex<double>>>> C(r,
            std::vector<std::vector<std::complex<double>>>(r, std::vector<std::complex<double>>(t.size(), {0.0,0.0})));
        for (std::size_t a = 0; a < r; ++a) {
            for (std::size_t k = 0; k < t.size(); ++k) C[a][a][k] = Cdiag[k];
        }
        return TabulatedCorrelation(std::move(t), std::move(C));
    }

    // Convenience: build diagonal correlation from spectral density J(w) and temperature beta
    template<class JCallable>
    static TabulatedCorrelation from_spectral(std::size_t r,
                                              std::size_t N,
                                              double dt,
                                              JCallable J,
                                              double beta) {
        std::vector<double> t;
        std::vector<std::complex<double>> C;
        bcf::bcf_fft_fun(N, dt, J, beta, t, C);
        return diagonal(r, std::move(t), std::move(C));
    }

private:
    std::size_t rank_{0};
    std::vector<double> t_;
    std::vector<std::vector<std::vector<std::complex<double>>>> C_;
};

inline TabulatedCorrelation make_ohmic_bath(std::size_t rank,
                                            double alpha,
                                            double omega_c,
                                            double beta,
                                            std::size_t N,
                                            double dt)
{
    if (!(omega_c > 0.0)) throw std::invalid_argument("make_ohmic_bath: omega_c must be > 0");
    auto J = [alpha, omega_c](double w) {
        if (!(w > 0.0)) return 0.0;
        return 2.0 * alpha * w * std::exp(-w / omega_c);
    };
    return TabulatedCorrelation::from_spectral(rank, N, dt, J, beta);
}

} // namespace taco::bath
