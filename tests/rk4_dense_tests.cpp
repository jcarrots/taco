#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "taco/rk4_dense.hpp"

static int fails = 0;
static std::vector<std::string> g_log;

static double rel_err_vec(const Eigen::VectorXcd& got, const Eigen::VectorXcd& expect) {
    const double denom = std::max(1.0, expect.norm());
    return (got - expect).norm() / denom;
}

static void check_vec_close(const char* name,
                            const Eigen::VectorXcd& got,
                            const Eigen::VectorXcd& expect,
                            double tol) {
    const double e = rel_err_vec(got, expect);
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line.precision(10);
    if (e > tol || !got.allFinite() || !expect.allFinite()) {
        line << "FAIL " << name << ": relerr=" << e;
        std::cerr << line.str() << "\n";
        g_log.push_back(line.str());
        ++fails;
    } else {
        line << "ok   " << name << " (err=" << e << ")";
        std::cout << line.str() << "\n";
        g_log.push_back(line.str());
    }
}

static void test_constant_L() {
    const Eigen::Index D = 4;
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(D, D);
    L(0, 0) = std::complex<double>(0.10, -0.03);
    L(1, 1) = std::complex<double>(-0.07, 0.05);
    L(2, 2) = std::complex<double>(0.02, 0.00);
    L(3, 3) = std::complex<double>(-0.04, -0.02);
    L(0, 2) = std::complex<double>(0.03, 0.01);
    L(2, 0) = std::complex<double>(-0.02, 0.02);
    L(1, 3) = std::complex<double>(0.01, -0.04);

    Eigen::VectorXcd r0(D);
    r0 << std::complex<double>(1.0, 0.0),
          std::complex<double>(0.5, -0.2),
          std::complex<double>(-0.7, 0.1),
          std::complex<double>(0.1, 0.3);

    const double t0 = 0.0;
    const double tf = 1.25;
    const double dt = 0.0025;

    Eigen::VectorXcd r = r0;
    taco::tcl::propagate_rk4_dense_serial(L, r, t0, tf, dt);

    const Eigen::MatrixXcd M = ((tf - t0) * L).exp();
    const Eigen::VectorXcd expect = M * r0;
    check_vec_close("rk4_dense constant L", r, expect, 1e-6);
}

static void test_time_dependent_L_endpoints_midpoints() {
    const Eigen::Index D = 3;
    const std::size_t steps = 100;
    const double t0 = 0.0;
    const double dt = 0.01;
    const double tf = t0 + static_cast<double>(steps) * dt;

    std::vector<Eigen::MatrixXcd> L_series(steps + 1);
    std::vector<Eigen::MatrixXcd> L_half_series(steps);

    for (std::size_t i = 0; i < steps; ++i) {
        const double t = t0 + static_cast<double>(i) * dt;
        const double t_half = t + 0.5 * dt;
        L_series[i] = t * Eigen::MatrixXcd::Identity(D, D);
        L_half_series[i] = t_half * Eigen::MatrixXcd::Identity(D, D);
    }
    L_series[steps] = tf * Eigen::MatrixXcd::Identity(D, D);

    Eigen::VectorXcd r0(D);
    r0 << std::complex<double>(0.3, 0.1),
          std::complex<double>(-0.2, 0.4),
          std::complex<double>(0.9, -0.3);

    Eigen::VectorXcd r = r0;
    taco::tcl::propagate_rk4_dense_serial(L_series, L_half_series, r, t0, dt);

    const double integral = 0.5 * (tf * tf - t0 * t0);
    const std::complex<double> scale = std::exp(integral);
    const Eigen::VectorXcd expect = scale * r0;
    check_vec_close("rk4_dense time-dependent L (midpoints)", r, expect, 1e-6);
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(10);

    test_constant_L();
    test_time_dependent_L_endpoints_midpoints();

    {
        std::filesystem::path outdir;
        if (argc > 0) {
            std::filesystem::path exe(argv[0]);
            outdir = exe.has_parent_path() ? exe.parent_path() : std::filesystem::current_path();
        } else {
            outdir = std::filesystem::current_path();
        }
        std::filesystem::path outfile = outdir / "rk4_dense_test_results.txt";
        std::ofstream ofs(outfile, std::ios::out | std::ios::trunc);
        if (ofs) {
            ofs.setf(std::ios::fixed);
            ofs.precision(10);
            ofs << "RK4 Dense Tests Results\n";
            for (const auto& s : g_log) ofs << s << "\n";
            if (fails) ofs << "FAILED: " << fails << " test(s)\n";
            else ofs << "All rk4_dense tests passed.\n";
        }
    }

    if (fails) {
        std::cerr << "\nFAILED: " << fails << " test(s)\n";
        return 1;
    }
    std::cout << "\nAll rk4_dense tests passed.\n";
    return 0;
}
