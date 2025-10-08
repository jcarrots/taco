#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <fstream>

#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"

using cd = std::complex<double>;

static inline double rel_err(double a, double b) {
    double denom = std::max(1.0, std::abs(b));
    return std::abs(a - b) / denom;
}

int main() {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);

    // Zero-temperature Ohmic with exponential cutoff (g=1):
    //   J(ω) = 2π ω e^{-ω/ωc} Θ(ω)
    //   C(t) = ωc^2 / (1 + i ωc t)^2
    const double wc = 1.0;
    auto C_exact = [&](double tt) -> cd {
        cd denom = cd{1.0, wc * tt};  // 1 + i wc t
        return (wc * wc) / (denom * denom);
    };

    // Build analytic C(t) samples on [0, T]
    const double dt = 0.0001;     // finer sampling
    const double T  = 10000.0;      // larger window to better approximate t -> ∞
    const std::size_t N = static_cast<std::size_t>(std::ceil(T / dt));
    std::vector<cd> Cnum(N+1);
    for (std::size_t k = 0; k <= N; ++k) Cnum[k] = C_exact(k * dt);

    // Test Γ(ω, t) at t ≈ T against γ(ω) for ω ∈ {0, -1, 1}
    const double PI = 3.141592653589793238462643383279502884;
    const std::vector<double> test_omegas = {0.0, -1.0, 1.0};
    double max_rel_gamma_pos = 0.0; // only ω>0
    double max_abs_im = 0.0;
    double im_zero_abs_err = 0.0;
    std::cout << "time step dt, final time T respectively: " << dt << ", " << T << "\n";
    for (double w : test_omegas) {
        auto Gw = taco::gamma::compute_trapz(Cnum, dt, w);
        cd Gfinal = Gw.back();
        // For Hermitian C: γ(ω) = 2 Re ∫_0^∞ e^{i ω t} C(t) dt ≈ 2 Re Γ(ω, T)
        const double gamma_est_re = 2.0 * std::real(Gfinal);
        const double gamma_est_im = 2.0 * std::imag(Gfinal);
        max_abs_im = std::max(max_abs_im, std::abs(gamma_est_im));

        const double J_exact = (w > 0.0) ? (2.0 * PI * w * std::exp(-w / wc)) : 0.0;
        if (J_exact > 0.0) {
            max_rel_gamma_pos = std::max(max_rel_gamma_pos, rel_err(gamma_est_re, J_exact));
        }
        // Imag part check: at ω=0, S(0) = 2 Im Γ(0,∞) = -2 ωc (analytic)
        if (std::abs(w) < 1e-12) {
            const double S0_exact = -2.0 * wc;
            im_zero_abs_err = std::abs(gamma_est_im - S0_exact);
        }
        std::cout << "omega=" << w
                  << " J_est=" << gamma_est_re
                  << " S_est=" << gamma_est_im
                  << " J_exact=" << J_exact
                  << "\n";
    }
    std::cout << "Gamma real-part rel error (ω>0): " << max_rel_gamma_pos << "\n";
    std::cout << "Gamma imag-part max |Im|: " << max_abs_im << "\n";
    std::cout << "Gamma imag at w=0 abs error vs -2*wc: " << im_zero_abs_err << "\n";

    // Write summary
    std::ofstream ofs("gamma_test_results.txt", std::ios::out | std::ios::trunc);
    ofs.setf(std::ios::fixed); ofs.precision(9);
    ofs << "Gamma real-part rel error (ω>0): " << max_rel_gamma_pos << "\n";
    ofs << "Gamma imag at w=0 abs error vs -2*wc: " << im_zero_abs_err << "\n";

    // Criteria: with dt=0.002, T=20, expect a few % on real part at ω=1, and imag ~ 1e-3
    const bool ok = (max_rel_gamma_pos < 2e-2) && (im_zero_abs_err < 5e-3);
    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
