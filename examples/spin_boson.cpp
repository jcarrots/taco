#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "taco/ops.hpp"
#include "taco/propagate.hpp"
#include "taco/spin_boson.hpp"
#include "taco/tcl.hpp"

int main(int argc, char* argv[]) {
    using namespace taco;
    using Matrix = Eigen::MatrixXcd;

    struct CliSettings {
        double Delta{1.0};
        double epsilon{0.2};
        double alpha{0.05};
        double omega_c{5.0};
        double beta{5.0};
        std::size_t rank{1};
        std::string coupling{"sz"};
        std::string generator_type{"tcl2"};
        double t0{0.0};
        double tf{10.0};
        double dt{0.01};
        std::size_t sample_every{10};
        std::size_t Ncorr{8192};
        double dt_corr{0.001};
        std::string observables_file{"spin_boson_observables.csv"};
        std::string density_file{"spin_boson_density.csv"};
    } settings;

    auto parse_size = [](const std::string& s) -> std::size_t {
        std::size_t pos = 0;
        std::size_t value = 0;
        try {
            value = static_cast<std::size_t>(std::stoull(s, &pos));
        } catch (...) {
            throw std::invalid_argument("Invalid integer: " + s);
        }
        if (pos != s.size()) throw std::invalid_argument("Invalid integer: " + s);
        return value;
    };

    auto parse_double = [](const std::string& s) -> double {
        std::size_t pos = 0;
        double value = 0.0;
        try {
            value = std::stod(s, &pos);
        } catch (...) {
            throw std::invalid_argument("Invalid double: " + s);
        }
        if (pos != s.size()) throw std::invalid_argument("Invalid double: " + s);
        return value;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--", 0) != 0) {
            std::cerr << "Ignoring argument (expected --key=value): " << arg << '\n';
            continue;
        }
        const auto eq = arg.find('=');
        std::string key = arg.substr(2, eq == std::string::npos ? std::string::npos : eq - 2);
        std::string value = (eq == std::string::npos) ? std::string{} : arg.substr(eq + 1);
        try {
            if (key == "delta") settings.Delta = parse_double(value);
            else if (key == "epsilon") settings.epsilon = parse_double(value);
            else if (key == "alpha") settings.alpha = parse_double(value);
            else if (key == "omega_c") settings.omega_c = parse_double(value);
            else if (key == "beta") settings.beta = parse_double(value);
            else if (key == "rank") settings.rank = parse_size(value);
            else if (key == "coupling") settings.coupling = value;
            else if (key == "generator") settings.generator_type = value;
            else if (key == "t0") settings.t0 = parse_double(value);
            else if (key == "tf") settings.tf = parse_double(value);
            else if (key == "dt") settings.dt = parse_double(value);
            else if (key == "sample_every") settings.sample_every = parse_size(value);
            else if (key == "ncorr") settings.Ncorr = parse_size(value);
            else if (key == "dt_corr") settings.dt_corr = parse_double(value);
            else if (key == "observables") settings.observables_file = value;
            else if (key == "density") settings.density_file = value;
            else {
                std::cerr << "Unknown option: --" << key << '\n';
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error parsing --" << key << ": " << ex.what() << '\n';
            return 1;
        }
    }

    // Map coupling keyword to operator
    Eigen::MatrixXcd coupling_matrix;
    if (settings.coupling == "sz") coupling_matrix = ops::sigma_z();
    else if (settings.coupling == "sx") coupling_matrix = ops::sigma_x();
    else if (settings.coupling == "sy") coupling_matrix = ops::sigma_y();
    else if (settings.coupling == "sm") coupling_matrix = ops::sigma_minus();
    else if (settings.coupling == "sp") coupling_matrix = ops::sigma_plus();
    else {
        std::cerr << "Unknown coupling operator label: " << settings.coupling << '\n';
        std::cerr << "Supported: sz, sx, sy, sm (sigma_minus), sp (sigma_plus)" << std::endl;
        return 1;
    }

    if (settings.generator_type != "tcl2") {
        std::cerr << "Only generator=tcl2 is currently supported (requested '"
                  << settings.generator_type << "')" << std::endl;
        return 1;
    }
    if (settings.sample_every == 0) {
        std::cerr << "sample_every must be >= 1" << std::endl;
        return 1;
    }

    spin_boson::Params params;
    params.Delta = settings.Delta;
    params.epsilon = settings.epsilon;
    params.alpha = settings.alpha;
    params.omega_c = settings.omega_c;
    params.beta = settings.beta;
    params.rank = settings.rank;
    params.coupling_operator = coupling_matrix;

    tcl::GeneratorOptions opts;
    opts.integration_step_hint = 0.01;
    opts.integration_min_subdivisions = 8;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Spin-boson configuration:\n"
              << "  Delta=" << settings.Delta
              << ", epsilon=" << settings.epsilon
              << ", alpha=" << settings.alpha
              << ", omega_c=" << settings.omega_c
              << ", beta=" << settings.beta
              << ", rank=" << settings.rank
              << "\n  coupling=" << settings.coupling
              << ", generator=" << settings.generator_type
              << "\n  t0=" << settings.t0
              << ", tf=" << settings.tf
              << ", dt=" << settings.dt
              << ", sample_every=" << settings.sample_every
              << "\n  ncorr=" << settings.Ncorr
              << ", dt_corr=" << settings.dt_corr
              << "\n  observables file=" << settings.observables_file
              << "\n  density file=" << settings.density_file
              << std::endl;

    spin_boson::Model model(params, settings.Ncorr, settings.dt_corr, opts);
    auto& generator = model.generator;
    generator.reset(settings.t0);

    Matrix rho = ops::rho_qubit_1(); // initial state (excited)

    const double t0 = settings.t0;
    const double tf = settings.tf;
    const double dt = settings.dt;
    const std::size_t sample_every = settings.sample_every;

    std::vector<double> times;
    std::vector<double> exp_sz;
    std::vector<double> excited_pop;
    std::vector<Matrix> density_samples;
    times.reserve(static_cast<std::size_t>((tf - t0) / dt) + 10);
    exp_sz.reserve(times.capacity());
    excited_pop.reserve(times.capacity());
    density_samples.reserve(times.capacity());

    const Matrix sigma_z = ops::sigma_z();

    auto on_sample = [&](double t, const Matrix& rho_sample) {
        const double sz = (rho_sample * sigma_z).trace().real();
        times.push_back(t);
        exp_sz.push_back(sz);
        excited_pop.push_back(rho_sample(1, 1).real());
        density_samples.push_back(rho_sample);
    };

    tcl::propagate_rk4(generator, rho, t0, tf, dt, on_sample, sample_every);

    std::ofstream ofs(settings.observables_file, std::ios::out | std::ios::trunc);
    if (!ofs) {
        std::cerr << "Failed to open output file\n";
        return 1;
    }
    ofs << std::fixed << std::setprecision(6);
    ofs << "t,sz,p_excited\n";
    for (std::size_t i = 0; i < times.size(); ++i) {
        ofs << times[i] << ',' << exp_sz[i] << ',' << excited_pop[i] << '\n';
    }

    // Write full density matrices
    std::ofstream ofs_rho(settings.density_file, std::ios::out | std::ios::trunc);
    if (!ofs_rho) {
        std::cerr << "Failed to open density output file\n";
        return 1;
    }
    ofs_rho << std::fixed << std::setprecision(6);
    const std::size_t dim = static_cast<std::size_t>(rho.rows());
    ofs_rho << "t";
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            ofs_rho << ",rho_" << r << '_' << c << "_re";
            ofs_rho << ",rho_" << r << '_' << c << "_im";
        }
    }
    ofs_rho << '\n';
    for (std::size_t i = 0; i < times.size(); ++i) {
        ofs_rho << times[i];
        const Matrix& rs = density_samples[i];
        for (std::size_t r = 0; r < dim; ++r) {
            for (std::size_t c = 0; c < dim; ++c) {
                const auto& val = rs(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
                ofs_rho << ',' << val.real() << ',' << val.imag();
            }
        }
        ofs_rho << '\n';
    }

    std::cout << "Final time: " << tf << "\n";
    std::cout << "Final expectation <sigma_z>: " << exp_sz.back() << "\n";
    std::cout << "Final excited population: " << excited_pop.back() << "\n";
    std::cout << "Observables saved to " << settings.observables_file << '\n';
    std::cout << "Density matrices saved to " << settings.density_file << std::endl;

    return 0;
}
