#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "taco/correlation_fft.hpp"
#include "taco/gamma.hpp"
#include "taco/generator.hpp"
#include "taco/io.hpp"
#include "taco/ops.hpp"
#include "taco/profile.hpp"
#include "taco/rk4_dense.hpp"
#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_kernels.hpp"
#include "taco/tcl4_mikx.hpp"

namespace {

struct Settings {
    // System (spin-boson)
    // H_lab = (Delta*sigma_x + epsilon*sigma_z)/2
    // A_lab = a_scale * sigma_z
    double Delta{1.0};
    double epsilon{0.0};
    double a_scale{0.5}; // MATLAB benchmark uses 1/2

    // Bath / grid
    double beta{0.5};
    double omega_c{10.0};
    double alpha{1.0}; // prefactor in J(w) = alpha * w * exp(-w/omega_c)
    double dt{0.001};
    std::size_t N{262144};      // N steps => Nt=N+1 samples (used for Gamma/GW window)
    std::size_t bcf_N{0};       // if 0, uses N; can be larger for a better C(t)

    // Integration rule for Gamma(omega,t)
    std::string gamma_rule{"trapz"}; // rect|trapz

    // TCL4 kernel method
    taco::tcl4::FCRMethod method{taco::tcl4::FCRMethod::Convolution};

    // Output time index and file path
    std::string tidx_str{"last"}; // integer or "last"
    std::string out_csv{};

    // Propagation (RK4 on vec(rho) with dense Liouvillian)
    bool propagate{false};
    int order{2};                // 0, 2, or 4 (0th/2nd/4th order generator)
    std::string rho0{"0"};       // 0|1|+x|-x|+y|-y (qubit only)
    bool print_series{false};    // print observables each sample
    std::size_t sample_every{1}; // print every N steps (only if print_series=1)
    std::string rho_out{};       // optional CSV for final rho (row,col,re,im)

    // Print stage timings
    bool profile{false};

    // OpenMP threads (if enabled at build time)
    int threads{0};
};

std::size_t parse_index(const std::string& s, std::size_t Nt) {
    if (s == "last") {
        if (Nt == 0) throw std::runtime_error("Nt is zero");
        return Nt - 1;
    }
    const long long v = std::stoll(s);
    if (v < 0) throw std::runtime_error("tidx must be >= 0");
    const auto idx = static_cast<std::size_t>(v);
    if (idx >= Nt) throw std::runtime_error("tidx out of range");
    return idx;
}

bool parse_bool_flag(const std::string& s) {
    if (s.empty()) return true;
    std::string v = s;
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw std::runtime_error("Invalid bool value: " + s);
}

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

Eigen::MatrixXcd parse_rho0_qubit(const std::string& s) {
    const std::string v = lower_copy(s);
    if (v.empty() || v == "0" || v == "g" || v == "ground") return taco::ops::rho_qubit_0();
    if (v == "1" || v == "e" || v == "excited") return taco::ops::rho_qubit_1();
    if (v == "+x") return taco::ops::rho_plus_x();
    if (v == "-x") return taco::ops::rho_minus_x();
    if (v == "+y") return taco::ops::rho_plus_y();
    if (v == "-y") return taco::ops::rho_minus_y();
    throw std::runtime_error("Unsupported rho0 (qubit): " + s);
}

} // namespace

int main(int argc, char** argv) {
    using Matrix = Eigen::MatrixXcd;
    using cd = std::complex<double>;

    try {
        Settings S;

        // Basic --key=value parsing
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg.rfind("--", 0) != 0) continue;
            const auto eq = arg.find('=');
            const std::string key = arg.substr(2, eq == std::string::npos ? std::string::npos : eq - 2);
            const std::string val = (eq == std::string::npos) ? std::string{} : arg.substr(eq + 1);

            if (key == "beta") S.beta = std::stod(val);
            else if (key == "omega_c") S.omega_c = std::stod(val);
            else if (key == "alpha") S.alpha = std::stod(val);
            else if (key == "dt") S.dt = std::stod(val);
            else if (key == "delta") S.Delta = std::stod(val);
            else if (key == "epsilon") S.epsilon = std::stod(val);
            else if (key == "a_scale") S.a_scale = std::stod(val);
            else if (key == "N") S.N = static_cast<std::size_t>(std::stoull(val));
            else if (key == "bcf_N") S.bcf_N = static_cast<std::size_t>(std::stoull(val));
            else if (key == "gamma_rule") S.gamma_rule = val;
            else if (key == "method") {
                if (val == "direct") S.method = taco::tcl4::FCRMethod::Direct;
                else S.method = taco::tcl4::FCRMethod::Convolution;
            } else if (key == "tidx") {
                S.tidx_str = val;
            } else if (key == "out") {
                S.out_csv = val;
            } else if (key == "propagate") {
                S.propagate = parse_bool_flag(val);
            } else if (key == "order") {
                S.order = std::stoi(val);
            } else if (key == "rho0") {
                S.rho0 = val;
            } else if (key == "print_series") {
                S.print_series = parse_bool_flag(val);
            } else if (key == "sample_every") {
                S.sample_every = static_cast<std::size_t>(std::stoull(val));
            } else if (key == "rho_out") {
                S.rho_out = val;
            } else if (key == "profile") {
                S.profile = parse_bool_flag(val);
            } else if (key == "threads") {
                S.threads = std::stoi(val);
            }
        }

#ifdef _OPENMP
        if (S.threads > 0) omp_set_num_threads(S.threads);
#endif

        taco::profile::Session prof(S.profile, std::cout);
        auto total_sec = prof.section("Total");

        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(9);

        const std::size_t bcf_N = (S.bcf_N == 0) ? S.N : S.bcf_N;
        if (bcf_N < S.N) {
            throw std::runtime_error("bcf_N must be >= N (need at least N+1 C(t) samples)");
        }

        std::cout << "TCL4 spin-boson example\n"
                  << "Delta=" << S.Delta << ", epsilon=" << S.epsilon << ", a_scale=" << S.a_scale << "\n"
                  << "beta=" << S.beta << ", omega_c=" << S.omega_c
                  << ", alpha=" << S.alpha
                  << ", dt=" << S.dt << ", N=" << S.N << ", bcf_N=" << bcf_N
                  << ", gamma_rule=" << S.gamma_rule
                  << ", method=" << (S.method == taco::tcl4::FCRMethod::Convolution ? "convolution" : "direct")
                  << ", propagate=" << (S.propagate ? "1" : "0")
                  << ", order=" << S.order
                  << "\n";
#ifdef _OPENMP
        std::cout << "OpenMP: max_threads=" << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP: disabled at build time\n";
#endif

        // System in lab basis
        Matrix H = 0.5 * S.Delta * taco::ops::sigma_x() + 0.5 * S.epsilon * taco::ops::sigma_z();
        Matrix A_lab = S.a_scale * taco::ops::sigma_z();

        taco::sys::System system;
        {
            auto sec = prof.section("Build system");
            system.build(H, {A_lab}, /*freq_tol=*/1e-9);
        }

        // Build C(t) from Ohmic J(w) = alpha*w * exp(-w/omega_c) via FFT.
        // We optionally build on a longer grid (bcf_N) to improve frequency resolution,
        // then slice the first Nt=N+1 samples for Gamma/GW.
        std::vector<double> tgrid;
        std::vector<cd> Ccorr;
        auto J = [&](double w) -> double {
            if (!(w > 0.0)) return 0.0;
            return S.alpha * w * std::exp(-w / S.omega_c);
        };
        {
            auto sec = prof.section("BCF FFT");
            bcf::bcf_fft_fun(bcf_N, S.dt, J, S.beta, tgrid, Ccorr);
        }
        const std::size_t Nt_use = S.N + 1;
        if (tgrid.size() < Nt_use || Ccorr.size() < Nt_use) {
            throw std::runtime_error("BCF output shorter than requested Nt=N+1");
        }
        tgrid.resize(Nt_use);
        Ccorr.resize(Nt_use);

        // Omega buckets for this system
        const std::size_t nf = system.fidx.buckets.size();
        std::vector<double> omegas(nf);
        for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;

        // Build Gamma(omega,t) series
        Eigen::MatrixXcd gamma_series;
        {
            auto sec = prof.section("Gamma series");
            if (S.gamma_rule == "trapz") {
                gamma_series = taco::gamma::compute_trapz_prefix_multi_matrix(Ccorr, S.dt, omegas);
            } else {
                gamma_series = taco::gamma::compute_rect_prefix_multi_matrix(Ccorr, S.dt, omegas);
            }
        }

        const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
        const std::size_t tidx = parse_index(S.tidx_str, Nt);
        const double tsel = (tidx < tgrid.size()) ? tgrid[tidx] : (static_cast<double>(tidx) * S.dt);

        taco::tcl4::TripleKernelSeries kernels;
        taco::tcl4::Tcl4Map map;
        {
            auto sec = prof.section("Kernels (F/C/R)");
            kernels = taco::tcl4::compute_triple_kernels(system, gamma_series, S.dt, /*nmax*/2, S.method);
        }
        {
            auto sec = prof.section("Build TCL4 map");
            map = taco::tcl4::build_map(system, tgrid);
        }

        // Build TCL4 correction tensor (N^2 x N^2)
        Eigen::MatrixXcd GW;
        Eigen::MatrixXcd L4;
        {
            auto sec = prof.section("Build GW");
            const taco::tcl4::MikxTensors mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
            GW = taco::tcl4::assemble_liouvillian(mikx, system.A_eig);
            L4 = taco::tcl4::gw_to_liouvillian(GW, system.eig.dim);
        }

        std::cout << "Selected tidx=" << tidx << " (t=" << tsel << ")\n";
        std::cout << "GW (NAKZWAN indexing: row=(n,i), col=(m,j)):\n" << GW << "\n";
        std::cout << "L4 (Liouvillian superop: row=(n,m), col=(i,j)):\n" << L4 << "\n";

        if (!S.out_csv.empty()) {
            taco::io::write_csv_matrix(S.out_csv, GW);
            std::cout << "Wrote GW to " << S.out_csv << "\n";
        }

        if (S.propagate) {
            if (S.sample_every == 0) throw std::runtime_error("sample_every must be > 0");
            if (!(S.order == 0 || S.order == 2 || S.order == 4)) {
                throw std::runtime_error("order must be 0, 2, or 4");
            }

            const std::size_t dim = system.eig.dim;
            const Eigen::Index D = static_cast<Eigen::Index>(dim * dim);

            // Initial state in eigen basis
            Eigen::MatrixXcd rho0 = (dim == 2) ? parse_rho0_qubit(S.rho0)
                                               : taco::ops::projector(dim, 0);
            Eigen::VectorXcd r = taco::ops::vec(rho0);

            // L0 = -i[H,·] with H diagonal in eigen basis
            const Eigen::MatrixXcd H0 = system.eig.eps.asDiagonal().toDenseMatrix().cast<std::complex<double>>();
            const Eigen::MatrixXcd L0 = taco::tcl2::build_unitary_superop(system, H0);

            // Prepare TCL2 spectral kernels container (diagonal-in-channel assumption)
            const std::size_t channels = system.A_eig_parts.size();
            taco::tcl2::SpectralKernels K2;
            K2.buckets.resize(nf);
            for (std::size_t b = 0; b < nf; ++b) {
                K2.buckets[b].omega = system.fidx.buckets[b].omega;
                K2.buckets[b].Gamma = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(channels),
                                                            static_cast<Eigen::Index>(channels));
            }

            auto fill_tcl2_kernels = [&](std::size_t time_index) {
                for (std::size_t b = 0; b < nf; ++b) {
                    const cd g = gamma_series(static_cast<Eigen::Index>(time_index), static_cast<Eigen::Index>(b));
                    if (channels == 1) {
                        K2.buckets[b].Gamma(0, 0) = g;
                    } else {
                        K2.buckets[b].Gamma.setZero();
                        for (std::size_t a = 0; a < channels; ++a) {
                            K2.buckets[b].Gamma(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(a)) = g;
                        }
                    }
                }
            };

            auto build_L_at = [&](std::size_t time_index) -> Eigen::MatrixXcd {
                if (S.order == 0) return L0;

                fill_tcl2_kernels(time_index);
                const taco::tcl2::TCL2Components comps2 = taco::tcl2::build_tcl2_components(system, K2, /*cutoff=*/0.0);
                Eigen::MatrixXcd L = comps2.total(); // includes L0 + 2nd-order Lamb shift + dissipator

                if (S.order == 4) {
                    const taco::tcl4::MikxTensors mikx = taco::tcl4::build_mikx_serial(map, kernels, time_index);
                    const Eigen::MatrixXcd GWt = taco::tcl4::assemble_liouvillian(mikx, system.A_eig); // (n,i;m,j)
                    L.noalias() += taco::tcl4::gw_to_liouvillian(GWt, dim);                             // (n,m;i,j)
                }
                return L;
            };

            const std::size_t steps = tidx;
            const double t0 = 0.0;
            const double tf = static_cast<double>(steps) * S.dt;

            std::cout << "RK4: steps=" << steps << ", t0=" << t0 << ", tf=" << tf << ", dt=" << S.dt
                      << " (order=" << S.order << ")\n";

            taco::tcl::Rk4DenseWorkspace ws;
            ws.resize(D);

            const Eigen::MatrixXcd sigma_z = taco::ops::sigma_z();
            auto sample = [&](std::size_t step, double t) {
                if (!S.print_series) return;
                if (step % S.sample_every != 0 && step != steps) return;
                Eigen::MatrixXcd rho = taco::ops::unvec(r, dim);
                rho = taco::ops::hermitize_and_normalize(rho);
                const double tr = rho.trace().real();
                const double purity = taco::ops::purity(rho);
                if (dim == 2) {
                    const Eigen::MatrixXcd rho_lab = system.eig.rho_to_lab(rho);
                    const double sz = (rho_lab * sigma_z).trace().real();
                    std::cout << t << "," << sz << "," << tr << "," << purity << "\n";
                } else {
                    std::cout << t << "," << tr << "," << purity << "\n";
                }
            };

            if (S.print_series) {
                if (dim == 2) std::cout << "t,sz,tr,purity\n";
                else std::cout << "t,tr,purity\n";
            }

            // On-the-fly RK4 integration using endpoint Liouvillians (midpoint via averaging)
            Eigen::MatrixXcd L_cur = build_L_at(0);
            Eigen::MatrixXcd L_next = (steps >= 1) ? build_L_at(1) : L_cur;

            double t = t0;
            sample(/*step=*/0, t);
            for (std::size_t step = 0; step < steps; ++step) {
                const Eigen::MatrixXcd Lhalf = 0.5 * (L_cur + L_next);
                taco::tcl::rk4_update_serial(L_cur, Lhalf, L_next, r, ws, S.dt);
                t += S.dt;

                sample(step + 1, t);

                if (step + 1 < steps) {
                    L_cur = std::move(L_next);
                    L_next = build_L_at(step + 2);
                }
            }

            Eigen::MatrixXcd rho_f = taco::ops::unvec(r, dim);
            rho_f = taco::ops::hermitize_and_normalize(rho_f);
            std::cout << "rho(tf) (eigen basis):\n" << rho_f << "\n";
            if (dim == 2) {
                std::cout << "rho(tf) (lab basis):\n" << system.eig.rho_to_lab(rho_f) << "\n";
            }

            if (!S.rho_out.empty()) {
                taco::io::write_csv_matrix(S.rho_out, rho_f);
                std::cout << "Wrote rho(tf) to " << S.rho_out << "\n";
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
