#include <cmath>
#include <iostream>
#include <vector>

#include "taco/ops.hpp"
#include "taco/propagate.hpp"
#include "taco/spin_boson.hpp"
#include "taco/bath_models.hpp"
#include "taco/generator.hpp"

int main() {
    using namespace taco;

    spin_boson::Params params;
    params.Delta = 1.0;
    params.epsilon = 0.2;
    params.alpha = 0.05;
    params.omega_c = 5.0;
    params.beta = 5.0;
    params.rank = 1;

    const std::size_t Ncorr = 8192;
    const double dt_corr = 0.001;

    tcl::GeneratorOptions opts;
    opts.integration_step_hint = 0.01;
    opts.integration_min_subdivisions = 8;

    auto model = spin_boson::build_model(params, Ncorr, dt_corr, opts);
    auto& generator = model.generator;
    generator.reset(0.0);

    Eigen::MatrixXcd rho = ops::rho_qubit_1();

    const double t0 = 0.0;
    const double tf = 10.0;
    const double dt = 0.01;

    std::vector<double> times;
    std::vector<Eigen::MatrixXcd> states;
    const std::size_t sample_every = 1;
    auto on_sample = [&](double t, const Eigen::MatrixXcd& rho_sample) {
        times.push_back(t);
        states.push_back(rho_sample);
    };

    tcl::propagate_rk4(generator, rho, t0, tf, dt, on_sample, sample_every);

    const std::size_t dim = static_cast<std::size_t>(rho.rows());
    std::cout.setf(std::ios::fixed); std::cout.precision(6);

    // CSV header
    std::cout << "t";
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            std::cout << ",rho_" << r << c << "_re";
            std::cout << ",rho_" << r << c << "_im";
        }
    }
    std::cout << '\n';

    for (std::size_t k = 0; k < times.size(); ++k) {
        std::cout << times[k];
        const auto& rk = states[k];
        for (std::size_t r = 0; r < dim; ++r) {
            for (std::size_t c = 0; c < dim; ++c) {
                const auto val = rk(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
                std::cout << ',' << val.real() << ',' << val.imag();
            }
        }
        std::cout << '\n';
    }

    // Optional: report Liouvillian components at final time
    auto system = spin_boson::build_system(params);
    auto kernels = bath::build_spectral_kernels_from_correlation(system, model.bath, model.bath.times());
    auto components = tcl2::build_tcl2_components(system, kernels, opts.gamma_cutoff);

    std::cout << '\n' << "# L_total (row, col, real, imag)" << '\n';
    const Eigen::MatrixXcd Ltot = components.total();
    const std::size_t NN = static_cast<std::size_t>(Ltot.rows());
    for (std::size_t r = 0; r < NN; ++r) {
        for (std::size_t c = 0; c < NN; ++c) {
            const auto v = Ltot(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
            std::cout << 'L' << ',' << r << ',' << c << ',' << v.real() << ',' << v.imag() << '\n';
        }
    }

    return 0;
}
