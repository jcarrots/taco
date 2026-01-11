#include <complex>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "taco/backend/cpu/tcl4_mpi_omp.hpp"
#include "taco/ops.hpp"
#include "taco/system.hpp"
#include "taco/tcl4.hpp"

#ifdef TACO_HAS_MPI
#include <mpi.h>
#endif

static double max_abs_diff(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return std::numeric_limits<double>::infinity();
    return (a - b).cwiseAbs().maxCoeff();
}

int main(int argc, char** argv) {
#ifndef TACO_HAS_MPI
    (void)argc;
    (void)argv;
    std::cout << "tcl4_mpi_omp_tests: SKIP (built without MPI)\n";
    return 0;
#else
    int rank = 0;
    int size = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Eigen::MatrixXcd H = 0.5 * taco::ops::sigma_x();
    Eigen::MatrixXcd A = 0.5 * taco::ops::sigma_z();
    taco::sys::System system;
    system.build(H, {A}, 1e-9);

    const std::size_t nf = system.fidx.buckets.size();
    const std::size_t Nt = 16;
    const double dt = 0.01;

    Eigen::MatrixXcd gamma_series(static_cast<Eigen::Index>(Nt), static_cast<Eigen::Index>(nf));
    for (std::size_t t = 0; t < Nt; ++t) {
        for (std::size_t b = 0; b < nf; ++b) {
            gamma_series(static_cast<Eigen::Index>(t), static_cast<Eigen::Index>(b)) =
                std::complex<double>(0.01 * static_cast<double>((t + 1) * (b + 1)),
                                     0.005 * static_cast<double>(t) - 0.002 * static_cast<double>(b));
        }
    }

    const std::vector<std::size_t> tids = {0, Nt / 2, Nt - 1};

    taco::Exec exec;
    exec.backend = taco::Backend::Serial;
    const auto ref_full =
        taco::tcl4::build_correction_series(system, gamma_series, dt, taco::tcl4::FCRMethod::Convolution, exec);

    const auto out =
        taco::tcl4::build_TCL4_generator_cpu_mpi_omp_batch(system,
                                                           gamma_series,
                                                           dt,
                                                           tids,
                                                           taco::tcl4::FCRMethod::Convolution,
                                                           comm);

    bool ok = true;
    if (rank == 0) {
        if (out.size() != tids.size()) {
            std::cerr << "FAIL: output size mismatch (got " << out.size()
                      << ", expected " << tids.size() << ")\n";
            ok = false;
        } else {
            for (std::size_t i = 0; i < tids.size(); ++i) {
                const double err = max_abs_diff(out[i], ref_full[tids[i]]);
                if (err > 1e-10) {
                    std::cerr << "FAIL: mismatch at tids[" << i << "]=" << tids[i]
                              << " (max_abs_diff=" << err << ")\n";
                    ok = false;
                    break;
                }
            }
        }
        std::cout << "tcl4_mpi_omp_tests: " << (ok ? "PASS" : "FAIL")
                  << " (size=" << size << ")\n";
    } else {
        if (!out.empty()) {
            std::cerr << "FAIL: non-root rank returned non-empty output\n";
            ok = false;
        }
    }

    MPI_Finalize();

    return ok ? 0 : 1;
#endif
}

