#include <Eigen/Dense>

#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "taco/tcl4_mikx.hpp"

static int fails = 0;
static std::vector<std::string> g_log;

static void check_close(const char* name,
                        const std::complex<double>& got,
                        const std::complex<double>& expect,
                        double tol) {
    const double err = std::abs(got - expect);
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line.precision(10);
    if (err > tol || !std::isfinite(got.real()) || !std::isfinite(got.imag())) {
        line << "FAIL " << name << ": err=" << err;
        std::cerr << line.str() << "\n";
        g_log.push_back(line.str());
        ++fails;
    } else {
        line << "ok   " << name << " (err=" << err << ")";
        std::cout << line.str() << "\n";
        g_log.push_back(line.str());
    }
}

static void check_size(const char* name, std::size_t got, std::size_t expect) {
    std::ostringstream line;
    if (got != expect) {
        line << "FAIL " << name << ": got=" << got << " expect=" << expect;
        std::cerr << line.str() << "\n";
        g_log.push_back(line.str());
        ++fails;
    } else {
        line << "ok   " << name;
        std::cout << line.str() << "\n";
        g_log.push_back(line.str());
    }
}

static void test_build_mikx_serial() {
    using taco::tcl4::Tcl4Map;
    using taco::tcl4::TripleKernelSeries;

    const int N = 2;
    const std::size_t nf = 4;

    Tcl4Map map;
    map.N = N;
    map.nf = static_cast<int>(nf);
    map.pair_to_freq = Eigen::MatrixXi::Constant(N, N, -1);
    map.pair_to_freq(0, 0) = 0;
    map.pair_to_freq(0, 1) = 1;
    map.pair_to_freq(1, 0) = 2;
    map.pair_to_freq(1, 1) = 3;

    TripleKernelSeries kernels;
    kernels.F.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    kernels.C.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));
    kernels.R.resize(nf, std::vector<std::vector<Eigen::VectorXcd>>(nf, std::vector<Eigen::VectorXcd>(nf)));

    auto make_val = [](double base, std::size_t a, std::size_t b, std::size_t c) {
        const double re = base + 100.0 * a + 10.0 * b + static_cast<double>(c);
        const double im = 0.1 * base + 0.01 * a - 0.001 * b + 0.0001 * c;
        return std::complex<double>(re, im);
    };

    for (std::size_t a = 0; a < nf; ++a) {
        for (std::size_t b = 0; b < nf; ++b) {
            for (std::size_t c = 0; c < nf; ++c) {
                kernels.F[a][b][c] = Eigen::VectorXcd(1);
                kernels.C[a][b][c] = Eigen::VectorXcd(1);
                kernels.R[a][b][c] = Eigen::VectorXcd(1);
                kernels.F[a][b][c](0) = make_val(1.0, a, b, c);
                kernels.C[a][b][c](0) = make_val(2.0, a, b, c);
                kernels.R[a][b][c](0) = make_val(3.0, a, b, c);
            }
        }
    }

    const auto mikx = taco::tcl4::build_mikx_serial(map, kernels, 0);

    const std::size_t N2 = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    check_size("M rows", static_cast<std::size_t>(mikx.M.rows()), N2);
    check_size("M cols", static_cast<std::size_t>(mikx.M.cols()), N2);
    check_size("I rows", static_cast<std::size_t>(mikx.I.rows()), N2);
    check_size("I cols", static_cast<std::size_t>(mikx.I.cols()), N2);
    check_size("K rows", static_cast<std::size_t>(mikx.K.rows()), N2);
    check_size("K cols", static_cast<std::size_t>(mikx.K.cols()), N2);
    check_size("X size", mikx.X.size(), N2 * N2 * N2);

    auto f = [&](int a, int b) {
        return static_cast<std::size_t>(map.pair_to_freq(a, b));
    };
    auto flat2 = [&](int row, int col) {
        return static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * static_cast<std::size_t>(N);
    };
    auto flat6 = [&](int j, int k, int p, int q, int r, int s) {
        const std::size_t NN = static_cast<std::size_t>(N);
        return static_cast<std::size_t>(j) +
               NN * (static_cast<std::size_t>(k) +
               NN * (static_cast<std::size_t>(p) +
               NN * (static_cast<std::size_t>(q) +
               NN * (static_cast<std::size_t>(r) +
               NN * static_cast<std::size_t>(s)))));
    };

    const double tol = 1e-12;
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            const auto f_jk = f(j, k);
            for (int p = 0; p < N; ++p) {
                for (int q = 0; q < N; ++q) {
                    const auto f_jq = f(j, q);
                    const auto f_pj = f(p, j);
                    const auto f_pq = f(p, q);
                    const auto f_qk = f(q, k);
                    const auto f_kq = f(k, q);
                    const auto f_qj = f(q, j);
                    const auto f_qp = f(q, p);

                    const std::size_t row = flat2(j, k);
                    const std::size_t col = flat2(p, q);

                    const std::complex<double> M_expect =
                        kernels.F[f_jk][f_jq][f_pj](0) - kernels.R[f_jq][f_pq][f_qk](0);
                    const std::complex<double> I_expect =
                        kernels.F[f_jk][f_qp][f_kq](0);
                    const std::complex<double> K_expect =
                        kernels.R[f_jk][f_pq][f_qj](0);

                    check_close("M entry", mikx.M(static_cast<Eigen::Index>(row),
                                                 static_cast<Eigen::Index>(col)), M_expect, tol);
                    check_close("I entry", mikx.I(static_cast<Eigen::Index>(row),
                                                 static_cast<Eigen::Index>(col)), I_expect, tol);
                    check_close("K entry", mikx.K(static_cast<Eigen::Index>(row),
                                                 static_cast<Eigen::Index>(col)), K_expect, tol);

                    for (int r = 0; r < N; ++r) {
                        for (int s = 0; s < N; ++s) {
                            const auto f_rs = f(r, s);
                            const std::complex<double> X_expect =
                                kernels.C[f_jk][f_pq][f_rs](0) + kernels.R[f_jk][f_pq][f_rs](0);
                            const std::size_t idx = flat6(j, k, p, q, r, s);
                            check_close("X entry", mikx.X[idx], X_expect, tol);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(10);

    test_build_mikx_serial();

    {
        std::filesystem::path outdir;
        if (argc > 0) {
            std::filesystem::path exe(argv[0]);
            outdir = exe.has_parent_path() ? exe.parent_path() : std::filesystem::current_path();
        } else {
            outdir = std::filesystem::current_path();
        }
        std::filesystem::path outfile = outdir / "tcl4_mikx_test_results.txt";
        std::ofstream ofs(outfile, std::ios::out | std::ios::trunc);
        if (ofs) {
            ofs.setf(std::ios::fixed);
            ofs.precision(10);
            ofs << "TCL4 MIKX Tests Results\n";
            for (const auto& s : g_log) ofs << s << "\n";
            if (fails) ofs << "FAILED: " << fails << " test(s)\n";
            else ofs << "All tcl4_mikx tests passed.\n";
        }
    }

    if (fails) {
        std::cerr << "\nFAILED: " << fails << " test(s)\n";
        return 1;
    }
    std::cout << "\nAll tcl4_mikx tests passed.\n";
    return 0;
}
