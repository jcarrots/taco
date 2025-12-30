#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "taco/correlation_fft.hpp"
#include "taco/expression.hpp"
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

using cd = std::complex<double>;
using Matrix = Eigen::MatrixXcd;

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

Eigen::MatrixXd parse_real_matrix(const YAML::Node& n, const std::string& name) {
    if (!n || !n.IsSequence()) {
        throw std::runtime_error(name + " must be a sequence of rows");
    }
    const std::size_t rows = n.size();
    if (rows == 0) return Eigen::MatrixXd();
    if (!n[0].IsSequence()) throw std::runtime_error(name + " must be a 2D sequence");
    const std::size_t cols = n[0].size();
    if (cols == 0) return Eigen::MatrixXd(static_cast<Eigen::Index>(rows), 0);

    Eigen::MatrixXd out(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    for (std::size_t r = 0; r < rows; ++r) {
        const YAML::Node row = n[r];
        if (!row.IsSequence()) throw std::runtime_error(name + " row is not a sequence");
        if (row.size() != cols) {
            throw std::runtime_error(name + " has inconsistent row sizes");
        }
        for (std::size_t c = 0; c < cols; ++c) {
            out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = row[c].as<double>();
        }
    }
    return out;
}

Matrix parse_complex_matrix(const YAML::Node& n, const std::string& name) {
    if (!n || !n.IsMap()) throw std::runtime_error(name + " must be a map with keys {re, im?}");
    if (!n["re"]) throw std::runtime_error("Missing key: " + name + ".re");
    const Eigen::MatrixXd re = parse_real_matrix(n["re"], name + ".re");
    Eigen::MatrixXd im;
    if (n["im"]) {
        im = parse_real_matrix(n["im"], name + ".im");
        if (im.rows() != re.rows() || im.cols() != re.cols()) {
            throw std::runtime_error(name + ": re/im dims mismatch");
        }
    } else {
        im = Eigen::MatrixXd::Zero(re.rows(), re.cols());
    }

    Matrix out(re.rows(), re.cols());
    for (Eigen::Index r = 0; r < re.rows(); ++r) {
        for (Eigen::Index c = 0; c < re.cols(); ++c) {
            out(r, c) = cd(re(r, c), im(r, c));
        }
    }
    return out;
}

std::size_t parse_index(const YAML::Node& n, std::size_t Nt) {
    if (!n) {
        if (Nt == 0) throw std::runtime_error("Nt is zero");
        return Nt - 1;
    }
    if (n.IsScalar()) {
        const std::string s = n.as<std::string>();
        if (lower_copy(s) == "last") {
            if (Nt == 0) throw std::runtime_error("Nt is zero");
            return Nt - 1;
        }
        const long long v = std::stoll(s);
        if (v < 0) throw std::runtime_error("tidx must be >= 0");
        const auto idx = static_cast<std::size_t>(v);
        if (idx >= Nt) throw std::runtime_error("tidx out of range");
        return idx;
    }
    throw std::runtime_error("tidx must be a scalar (integer or 'last')");
}

Matrix parse_rho0(const YAML::Node& n, std::size_t dim) {
    if (!n) return taco::ops::projector(dim, 0);

    if (n.IsMap() && n["re"]) {
        const Matrix rho = parse_complex_matrix(n, "simulation.rho0");
        if (rho.rows() != static_cast<Eigen::Index>(dim) || rho.cols() != static_cast<Eigen::Index>(dim)) {
            throw std::runtime_error("simulation.rho0: dimension mismatch");
        }
        return rho;
    }

    if (!n.IsScalar()) throw std::runtime_error("simulation.rho0 must be a matrix map or scalar");

    const std::string s = n.as<std::string>();
    if (dim == 2) {
        const std::string v = lower_copy(s);
        if (v.empty() || v == "0" || v == "g" || v == "ground") return taco::ops::rho_qubit_0();
        if (v == "1" || v == "e" || v == "excited") return taco::ops::rho_qubit_1();
        if (v == "+x") return taco::ops::rho_plus_x();
        if (v == "-x") return taco::ops::rho_minus_x();
        if (v == "+y") return taco::ops::rho_plus_y();
        if (v == "-y") return taco::ops::rho_minus_y();
    }

    // Fallback: treat as basis index
    const long long idx_ll = std::stoll(s);
    if (idx_ll < 0) throw std::runtime_error("simulation.rho0 index must be >= 0");
    const auto idx = static_cast<std::size_t>(idx_ll);
    return taco::ops::projector(dim, idx);
}

void print_usage(std::ostream& os) {
    os << "Usage: tcl_driver.exe [--config=PATH]\n"
          "Defaults: config=configs/tcl_driver.yaml\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path = "configs/tcl_driver.yaml";
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") {
                print_usage(std::cout);
                return 0;
            }
            if (arg.rfind("--config=", 0) == 0) {
                config_path = arg.substr(std::string("--config=").size());
            }
        }

        const YAML::Node cfg = YAML::LoadFile(config_path);

        // ------------------------------ Read config ------------------------------
        const YAML::Node sysn = cfg["system"];
        if (!sysn) throw std::runtime_error("Missing key: system");

        const double freq_tol = sysn["freq_tol"] ? sysn["freq_tol"].as<double>() : 1e-9;
        const Matrix H = parse_complex_matrix(sysn["H"], "system.H");
        if (H.rows() != H.cols()) throw std::runtime_error("system.H must be square");

        const YAML::Node A_list_node = sysn["A"];
        if (!A_list_node || !A_list_node.IsSequence() || A_list_node.size() == 0) {
            throw std::runtime_error("system.A must be a non-empty sequence");
        }
        std::vector<Matrix> A_lab;
        A_lab.reserve(A_list_node.size());
        for (std::size_t i = 0; i < A_list_node.size(); ++i) {
            A_lab.push_back(parse_complex_matrix(A_list_node[i], "system.A[" + std::to_string(i) + "]"));
        }
        for (const auto& A : A_lab) {
            if (A.rows() != H.rows() || A.cols() != H.cols()) throw std::runtime_error("system.A: dimension mismatch vs H");
        }

        const YAML::Node bath = cfg["bath"];
        if (!bath) throw std::runtime_error("Missing key: bath");
        const double beta = bath["beta"].as<double>();
        const std::string J_expr_src = bath["J_expr"].as<std::string>();
        std::vector<std::pair<std::string, double>> J_params;
        if (bath["params"]) {
            const YAML::Node pn = bath["params"];
            if (!pn.IsMap()) throw std::runtime_error("bath.params must be a map");
            for (const auto& kv : pn) {
                J_params.emplace_back(kv.first.as<std::string>(), kv.second.as<double>());
            }
        }
        const taco::expr::Expression J_expr = taco::expr::Expression::compile(J_expr_src, J_params);

        const YAML::Node num = cfg["numerics"];
        if (!num) throw std::runtime_error("Missing key: numerics");
        const double dt = num["dt"].as<double>();
        const std::size_t N = static_cast<std::size_t>(num["N"].as<unsigned long long>());
        const std::size_t bcf_N = num["bcf_N"] ? static_cast<std::size_t>(num["bcf_N"].as<unsigned long long>()) : N;
        if (bcf_N < N) throw std::runtime_error("numerics.bcf_N must be >= numerics.N");

        const std::string gamma_rule = num["gamma_rule"] ? lower_copy(num["gamma_rule"].as<std::string>()) : "rect";

        taco::tcl4::FCRMethod method = taco::tcl4::FCRMethod::Convolution;
        int nmax = 2;
        std::size_t fcr_fft_pad = 0;
        if (num["tcl4"]) {
            const YAML::Node t4 = num["tcl4"];
            if (t4["method"]) {
                const std::string m = lower_copy(t4["method"].as<std::string>());
                if (m == "direct") method = taco::tcl4::FCRMethod::Direct;
                else method = taco::tcl4::FCRMethod::Convolution;
            }
            if (t4["nmax"]) nmax = t4["nmax"].as<int>();
            if (t4["fcr_fft_pad"]) fcr_fft_pad = static_cast<std::size_t>(t4["fcr_fft_pad"].as<unsigned long long>());
        }

        const YAML::Node out = cfg["output"];
        const YAML::Node sim = cfg["simulation"];

        const bool profile = out && out["profile"] ? out["profile"].as<bool>() : false;

        int threads = 0;
        if (num["threads"]) threads = num["threads"].as<int>();
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#endif

        taco::profile::Session prof(profile, std::cout);
        auto total = prof.section("Total");

        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(9);
        std::cout << "tcl_driver config: " << config_path << "\n";
#ifdef _OPENMP
        std::cout << "OpenMP: max_threads=" << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP: disabled at build time\n";
#endif

        // ------------------------------- Build system ----------------------------
        taco::sys::System system;
        {
            auto sec = prof.section("Build system");
            system.build(H, A_lab, freq_tol);
        }
        const std::size_t dim = system.eig.dim;
        const std::size_t nf = system.fidx.buckets.size();

        std::cout << "dim=" << dim << ", channels=" << A_lab.size() << ", nf=" << nf
                  << ", dt=" << dt << ", N=" << N << ", bcf_N=" << bcf_N
                  << ", beta=" << beta
                  << ", gamma_rule=" << gamma_rule
                  << ", tcl4_method=" << (method == taco::tcl4::FCRMethod::Convolution ? "convolution" : "direct")
                  << ", nmax=" << nmax
                  << ", fcr_fft_pad=" << fcr_fft_pad
                  << "\n";

        if (fcr_fft_pad > 0) taco::tcl4::set_fcr_fft_pad_factor(fcr_fft_pad);

        // ------------------------------- BCF: C(t) -------------------------------
        std::vector<double> tgrid;
        std::vector<cd> Ccorr;
        auto J = [&](double w) -> double { return J_expr.eval(w); };
        {
            auto sec = prof.section("BCF FFT");
            bcf::bcf_fft_fun(bcf_N, dt, J, beta, tgrid, Ccorr);
        }
        const std::size_t Nt_use = N + 1;
        if (tgrid.size() < Nt_use || Ccorr.size() < Nt_use) {
            throw std::runtime_error("BCF output shorter than requested Nt=N+1");
        }
        tgrid.resize(Nt_use);
        Ccorr.resize(Nt_use);

        // Omega buckets for this system
        std::vector<double> omegas(nf);
        for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;

        // ------------------------------ Gamma series -----------------------------
        Eigen::MatrixXcd gamma_series;
        {
            auto sec = prof.section("Gamma series");
            if (gamma_rule == "trapz") {
                gamma_series = taco::gamma::compute_trapz_prefix_multi_matrix(Ccorr, dt, omegas);
            } else if (gamma_rule == "rect") {
                gamma_series = taco::gamma::compute_rect_prefix_multi_matrix(Ccorr, dt, omegas);
            } else {
                throw std::runtime_error("Unsupported numerics.gamma_rule (use rect or trapz)");
            }
        }
        const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());

        // ---------------------------- TCL4 kernels/map ---------------------------
        taco::tcl4::TripleKernelSeries kernels;
        taco::tcl4::Tcl4Map map;
        {
            auto sec = prof.section("Kernels (F/C/R)");
            kernels = taco::tcl4::compute_triple_kernels(system, gamma_series, dt, nmax, method);
        }
        {
            auto sec = prof.section("Build TCL4 map");
            map = taco::tcl4::build_map(system, tgrid);
        }

        const YAML::Node tidx_node = out && out["tidx"] ? out["tidx"] : (sim && sim["tidx"] ? sim["tidx"] : YAML::Node());
        const std::size_t tidx = parse_index(tidx_node, Nt);
        const double tsel = (tidx < tgrid.size()) ? tgrid[tidx] : (static_cast<double>(tidx) * dt);
        std::cout << "tidx=" << tidx << " (t=" << tsel << ")\n";

        // ------------------------------ Build GW/L4 ------------------------------
        Eigen::MatrixXcd GW_raw;
        Eigen::MatrixXcd L4;
        {
            auto sec = prof.section("Build GW/L4");
            const taco::tcl4::MikxTensors mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
            GW_raw = taco::tcl4::assemble_liouvillian(mikx, system.A_eig); // (n,i;m,j)
            L4 = taco::tcl4::gw_to_liouvillian(GW_raw, dim);               // (n,m;i,j)
        }

        std::cout << "GW_raw: " << GW_raw.rows() << "x" << GW_raw.cols()
                  << ", ||GW||_F=" << taco::ops::fro_norm(GW_raw) << "\n";
        std::cout << "L4: " << L4.rows() << "x" << L4.cols()
                  << ", ||L4||_F=" << taco::ops::fro_norm(L4) << "\n";

        if (out && out["gw_csv"]) {
            const std::string p = out["gw_csv"].as<std::string>();
            taco::io::write_csv_matrix(p, GW_raw);
            std::cout << "Wrote GW_raw to " << p << "\n";
        }
        if (out && out["l4_csv"]) {
            const std::string p = out["l4_csv"].as<std::string>();
            taco::io::write_csv_matrix(p, L4);
            std::cout << "Wrote L4 to " << p << "\n";
        }

        // ------------------------------- Propagation -----------------------------
        const bool propagate = sim && sim["propagate"] ? sim["propagate"].as<bool>() : false;
        if (propagate) {
            const int order = sim["order"] ? sim["order"].as<int>() : 4;
            if (!(order == 0 || order == 2 || order == 4)) throw std::runtime_error("simulation.order must be 0, 2, or 4");

            const Eigen::Index D = static_cast<Eigen::Index>(dim * dim);
            Matrix rho0 = parse_rho0(sim["rho0"], dim);
            Eigen::VectorXcd r = taco::ops::vec(rho0);

            // L0 = -i[H,·] with H diagonal in eigen basis
            const Matrix H0 = system.eig.eps.asDiagonal().toDenseMatrix().cast<cd>();
            const Matrix L0 = taco::tcl2::build_unitary_superop(system, H0);

            // TCL2 spectral kernels (diagonal-in-channel assumption)
            const std::size_t channels = system.A_eig_parts.size();
            taco::tcl2::SpectralKernels K2;
            K2.buckets.resize(nf);
            for (std::size_t b = 0; b < nf; ++b) {
                K2.buckets[b].omega = system.fidx.buckets[b].omega;
                K2.buckets[b].Gamma = Matrix::Zero(static_cast<Eigen::Index>(channels),
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

            auto build_L_at = [&](std::size_t time_index) -> Matrix {
                if (order == 0) return L0;
                fill_tcl2_kernels(time_index);
                const taco::tcl2::TCL2Components comps2 = taco::tcl2::build_tcl2_components(system, K2, /*cutoff=*/0.0);
                Matrix L = comps2.total();
                if (order == 4) {
                    const taco::tcl4::MikxTensors mikx = taco::tcl4::build_mikx_serial(map, kernels, time_index);
                    const Matrix GWt = taco::tcl4::assemble_liouvillian(mikx, system.A_eig); // (n,i;m,j)
                    L.noalias() += taco::tcl4::gw_to_liouvillian(GWt, dim);                   // (n,m;i,j)
                }
                return L;
            };

            const std::size_t steps = tidx;
            const double t0 = 0.0;
            const double tf = static_cast<double>(steps) * dt;
            std::cout << "RK4: steps=" << steps << ", t0=" << t0 << ", tf=" << tf << ", dt=" << dt << " (order=" << order << ")\n";

            taco::tcl::Rk4DenseWorkspace ws;
            ws.resize(D);

            const bool print_series = sim["print_series"] ? sim["print_series"].as<bool>() : false;
            const std::size_t sample_every =
                sim["sample_every"] ? static_cast<std::size_t>(sim["sample_every"].as<unsigned long long>()) : 1;
            if (print_series && sample_every == 0) throw std::runtime_error("simulation.sample_every must be >= 1");

            const Eigen::MatrixXcd sigma_z = (dim == 2) ? taco::ops::sigma_z() : Matrix();

            auto sample = [&](std::size_t step, double t) {
                if (!print_series) return;
                if (step % sample_every != 0 && step != steps) return;
                Matrix rho = taco::ops::unvec(r, dim);
                rho = taco::ops::hermitize_and_normalize(rho);
                const double tr = rho.trace().real();
                const double purity = taco::ops::purity(rho);
                if (dim == 2) {
                    const Matrix rho_lab = system.eig.rho_to_lab(rho);
                    const double sz = (rho_lab * sigma_z).trace().real();
                    std::cout << t << "," << sz << "," << tr << "," << purity << "\n";
                } else {
                    std::cout << t << "," << tr << "," << purity << "\n";
                }
            };

            if (print_series) {
                if (dim == 2) std::cout << "t,sz,tr,purity\n";
                else std::cout << "t,tr,purity\n";
            }

            Matrix L_cur = build_L_at(0);
            Matrix L_next = (steps >= 1) ? build_L_at(1) : L_cur;

            double t = t0;
            sample(0, t);
            for (std::size_t step = 0; step < steps; ++step) {
                const Matrix Lhalf = 0.5 * (L_cur + L_next);
                taco::tcl::rk4_update_serial(L_cur, Lhalf, L_next, r, ws, dt);
                t += dt;
                sample(step + 1, t);

                if (step + 1 < steps) {
                    L_cur = std::move(L_next);
                    L_next = build_L_at(step + 2);
                }
            }

            Matrix rho_f = taco::ops::unvec(r, dim);
            rho_f = taco::ops::hermitize_and_normalize(rho_f);
            std::cout << "rho(tf) (eigen basis):\n" << rho_f << "\n";
            if (dim == 2) std::cout << "rho(tf) (lab basis):\n" << system.eig.rho_to_lab(rho_f) << "\n";

            if (sim["rho_out"]) {
                const std::string p = sim["rho_out"].as<std::string>();
                taco::io::write_csv_matrix(p, rho_f);
                std::cout << "Wrote rho(tf) to " << p << "\n";
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
