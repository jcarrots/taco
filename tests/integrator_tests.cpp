#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <sstream>

#include "core/integrator.hpp"

template <typename T>
static inline T rel_err(T a, T b) {
    T denom = std::max<T>(T(1), std::abs(b));
    return std::abs(a - b) / denom;
}

static int fails = 0;
static std::vector<std::string> g_log;
static constexpr double PI = 3.141592653589793238462643383279502884;

static void check_close(const char* name, double got, double expect, double tol) {
    double e = rel_err(got, expect);
    std::ostringstream line;
    line.setf(std::ios::fixed); line.precision(10);
    if (e > tol || std::isnan(got)) {
        line << "FAIL " << name << ": got=" << got << " expect=" << expect << " relerr=" << e;
        std::cerr << line.str() << "\n";
        g_log.push_back(line.str());
        ++fails;
    } else {
        line << "ok   " << name << " (err=" << e << ")";
        std::cout << line.str() << "\n";
        g_log.push_back(line.str());
    }
}

static void test_scalar_quadrature() {
    // ∫_0^1 x^2 dx = 1/3
    auto f_quad = [](double x){ return x*x; };
    double t1 = integrate_trapezoid(f_quad, 0.0, 1.0, 1000);
    double s1 = integrate_simpson(f_quad, 0.0, 1.0, 1000);
    check_close("trapz x^2 [0,1]", t1, 1.0/3.0, 1e-6);
    check_close("simpson x^2 [0,1]", s1, 1.0/3.0, 1e-10);

    // Orientation flip
    double t2 = integrate_trapezoid(f_quad, 1.0, 0.0, 1000);
    check_close("trapz flip sign", t2, -t1, 1e-12);

    // ∫_0^π sin(x) dx = 2
    auto f_sin = [](double x){ return std::sin(x); };
    double s2 = integrate_simpson(f_sin, 0.0, PI, 2000);
    check_close("simpson sin [0,pi]", s2, 2.0, 1e-10);


    // ∫_{-inf}^{inf} e^{-x^2} dx = sqrt(pi)
    auto f_gauss = [](double x){ return std::exp(-x*x); };
    double gi = integrate_infinite_R(f_gauss, 1e-9);
    check_close("gaussian over R", gi, std::sqrt(PI), 5e-6);
}

static void test_discrete_trapz_and_cum() {
    // Sample sin on [0,pi]
    const std::size_t N = 100000;
    const double a = 0.0, b = PI;
    const double dx = (b - a) / N;
    std::vector<double> y(N+1);
    for (std::size_t i = 0; i <= N; ++i) y[i] = std::sin(a + dx * i);
    double trap = integrate_discrete_trapz(y, dx);
    check_close("discrete trapz sin [0,pi]", trap, 2.0, 1e-6);

    // Cumulative of constant 1 -> t
    std::vector<double> ones(N+1, 1.0);
    auto I = cumulative_trapz(ones, dx);
    check_close("cumulative trapz 1", I.back(), (N*dx), 1e-12);
}

static void test_convolution() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    const std::size_t n = 2048, m = 2048;
    const double dx = 0.0025;
    std::vector<double> a(n), b(m);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    auto full_ref = convolve_trapz(a, b, dx, ConvMode::Full);
    auto full_fft = convolve_fft(a, b, dx, ConvMode::Full);
    // Compare Full
    double max_rel = 0.0;
    for (std::size_t i = 0; i < full_ref.size(); ++i) {
        double e = rel_err(full_fft[i], full_ref[i]);
        if (e > max_rel) max_rel = e;
    }
    check_close("convolve Full (fft vs trapz)", 1.0 - max_rel, 1.0, 3e-2);

    // Compare Same
    auto same_ref = convolve_trapz(a, b, dx, ConvMode::Same);
    auto same_fft = convolve_fft(a, b, dx, ConvMode::Same);
    max_rel = 0.0;
    for (std::size_t i = 0; i < same_ref.size(); ++i) max_rel = std::max(max_rel, rel_err(same_fft[i], same_ref[i]));
    check_close("convolve Same (fft vs trapz)", 1.0 - max_rel, 1.0, 3e-2);

    // Compare Valid (only if n>=m)
    auto valid_ref = convolve_trapz(a, b, dx, ConvMode::Valid);
    auto valid_fft = convolve_fft(a, b, dx, ConvMode::Valid);
    max_rel = 0.0;
    for (std::size_t i = 0; i < valid_ref.size(); ++i) max_rel = std::max(max_rel, rel_err(valid_fft[i], valid_ref[i]));
    check_close("convolve Valid (fft vs trapz)", 1.0 - max_rel, 1.0, 3e-2);
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(10);

    test_scalar_quadrature();
    test_discrete_trapz_and_cum();
    test_convolution();

    // Write results to a local file in the current working directory
    {
        // 1) Try writing next to the executable
        {
            std::filesystem::path outdir;
            if (argc > 0) {
                std::filesystem::path exe(argv[0]);
                outdir = exe.has_parent_path() ? exe.parent_path() : std::filesystem::current_path();
            } else {
                outdir = std::filesystem::current_path();
            }
            std::filesystem::path outfile = outdir / "integrator_test_results.txt";
            std::ofstream ofs(outfile, std::ios::out | std::ios::trunc);
            if (ofs) {
                ofs.setf(std::ios::fixed); ofs.precision(10);
                ofs << "Integration Tests Results\n";
                for (const auto& s : g_log) ofs << s << "\n";
                if (fails) ofs << "FAILED: " << fails << " test(s)\n"; else ofs << "All integration tests passed.\n";
            }
        }
        // 2) Also write to current working directory
        {
            std::filesystem::path outfile = std::filesystem::current_path() / "integrator_test_results.txt";
            std::ofstream ofs(outfile, std::ios::out | std::ios::trunc);
            if (ofs) {
                ofs.setf(std::ios::fixed); ofs.precision(10);
                ofs << "Integration Tests Results\n";
                for (const auto& s : g_log) ofs << s << "\n";
                if (fails) ofs << "FAILED: " << fails << " test(s)\n"; else ofs << "All integration tests passed.\n";
            }
        }
    }

    if (fails) {
        std::cerr << "\nFAILED: " << fails << " test(s)\n";
        return 1;
    }
    std::cout << "\nAll integration tests passed.\n";
    return 0;
}
