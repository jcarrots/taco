#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "taco/bath.hpp"
#include "taco/tcl.hpp"

namespace demo {

class ExpBath final : public taco::bath::CorrelationFunction {
public:
    ExpBath(std::size_t r, double g2, double gamma)
        : r_(r), g2_(g2), gamma_(gamma) {}

    std::complex<double> operator()(double tau, std::size_t alpha, std::size_t beta) const override {
        if (alpha >= r_ || beta >= r_) return {0.0, 0.0};
        if (alpha != beta) return {0.0, 0.0};
        const double val = g2_ * std::exp(-gamma_ * std::max(0.0, tau));
        return {val, 0.0};
    }

    std::size_t rank() const noexcept override { return r_; }

private:
    std::size_t r_;
    double g2_;
    double gamma_;
};

}  // namespace demo

int main() {
    using namespace taco::tcl;

    // Two-level system: H = 0.5 * w0 * sigma_z in lab basis
    const double w0 = 1.0;
    Eigen::MatrixXcd H(2, 2);
    H << 0.5 * w0, 0.0,
         0.0,     -0.5 * w0;

    // One jump operator: sigma_minus
    Eigen::MatrixXcd sm(2, 2);
    sm << 0.0, 0.0,
          1.0, 0.0;

    std::vector<JumpOperator> Ls;
    Ls.push_back({"sigma_minus", sm});

    // Exponential bath correlation on each channel (diagonal in alpha,beta)
    demo::ExpBath bath(/*rank=*/Ls.size(), /*g2=*/1.0, /*gamma=*/0.5);

    GeneratorOptions opts;
    opts.integration_step_hint = 0.01;     // finer integration for the demo
    opts.integration_min_subdivisions = 10;

    TCL2Generator gen(H, Ls, bath, opts);
    gen.reset(0.0);

    // Time evolution setup
    const double dt = 0.01;   // step
    const double tf = 10.0;   // final time
    const int steps = static_cast<int>(std::ceil(tf / dt));
    const int sample_every = 1; // write every k steps

    // Output CSV setup
    const std::string out_csv = "rho_timeseries.csv";
    std::ofstream ofs(out_csv, std::ios::out | std::ios::trunc);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << out_csv << "\n";
        return 2;
    }
    ofs.setf(std::ios::fixed);
    ofs << std::setprecision(9);

    const std::size_t dim = gen.dimension();
    // Header: t, rho_ij_re, rho_ij_im in row-major order
    ofs << "t";
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            ofs << ",rho_" << i << "_" << j << "_re";
            ofs << ",rho_" << i << "_" << j << "_im";
        }
    }
    ofs << "\n";

    // Initial state: ground state |0><0|
    Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(2, 2);
    rho(0, 0) = 1.0;

    Eigen::MatrixXcd drho(2, 2);

    // Simple explicit Euler integrator for demonstration
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
        // Ensure generator kernels are at current time t
        if (gen.current_time() < t) {
            gen.advance(t);
        }
        // Compute derivative at time t
        gen.apply(rho, drho);
        // Step forward
        rho.noalias() += dt * drho;
        // Symmetrize and renormalize to mitigate drift
        rho = 0.5 * (rho + rho.adjoint());
        const double tr = rho.trace().real();
        if (std::abs(tr) > 0) {
            rho /= tr;
        }
        // Write sample
        if ((i % sample_every) == 0) {
            ofs << t;
            for (std::size_t r = 0; r < dim; ++r) {
                for (std::size_t c = 0; c < dim; ++c) {
                    const auto val = rho(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
                    ofs << "," << val.real() << "," << val.imag();
                }
            }
            ofs << "\n";
        }
        // Advance generator to t+dt for next step
        t = std::min(tf, t + dt);
        gen.advance(t);
    }

    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);
    std::cout << "Final time: " << t << "\n";
    std::cout << "rho(t_f):\n" << rho << "\n";
    std::cout << "Time series written to " << out_csv << "\n";
    return 0;
}
