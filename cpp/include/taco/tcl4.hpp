#pragma once

#include <vector>

#include <Eigen/Dense>

#include "taco/system.hpp"
#include "taco/tcl4_kernels.hpp"

namespace taco::tcl4 {

struct Tcl4Map {
    int N{0};
    int nf{0};
    std::vector<double> time_grid;
    std::vector<double> omegas;
    Eigen::MatrixXi pair_to_freq;
    std::vector<std::pair<int,int>> freq_to_pair;
    // For each frequency bucket index b, mirror_index[b] gives the index b' such that
    // omegas[b'] ≈ -omegas[b] (and b'==b when omegas[b]≈0). Useful for symmetric layouts
    // where w=0 is the center and +/-ω map to each other.
    std::vector<int> mirror_index;
    // Index of the zero-frequency bucket (|ω| ≤ tol). Must exist because ω_{mm}=0.
    int zero_index{-1};
};

Tcl4Map build_map(const sys::System& system, const std::vector<double>& time_grid);

struct TripleKernelSeries {
    std::vector<std::vector<std::vector<Eigen::VectorXcd>>> F;
    std::vector<std::vector<std::vector<Eigen::VectorXcd>>> C;
    std::vector<std::vector<std::vector<Eigen::VectorXcd>>> R;
};

// Default method = Convolution for performance
TripleKernelSeries compute_triple_kernels(const sys::System& system,
                                          const Eigen::MatrixXcd& gamma_series,
                                          double dt,
                                          int nmax,
                                          FCRMethod method = FCRMethod::Convolution);

// ---------------- Convenience rebuild helpers ----------------

// Rebuild an N×N Γ matrix at a given time index from the bucket‑major series.
Eigen::MatrixXcd build_gamma_matrix_at(const Tcl4Map& map,
                                       const Eigen::MatrixXcd& gamma_series,
                                       std::size_t time_index);

// Flattening helper consistent with MIKX (column‑major over j,k,p,q,r,s)
// flat6(N,j,k,p,q,r,s) = j + N*(k + N*(p + N*(q + N*(r + N*s))))

// Rebuild full 6‑index F/C/R tensors at a given time as flat N^6 vectors.
void build_FCR_6d_at(const Tcl4Map& map,
                     const TripleKernelSeries& kernels,
                     std::size_t time_index,
                     std::vector<std::complex<double>>& F_out,
                     std::vector<std::complex<double>>& C_out,
                     std::vector<std::complex<double>>& R_out);

// Convenience: rebuild at the final time sample (last index in series)
void build_FCR_6d_final(const Tcl4Map& map,
                        const TripleKernelSeries& kernels,
                        std::vector<std::complex<double>>& F_out,
                        std::vector<std::complex<double>>& C_out,
                        std::vector<std::complex<double>>& R_out);

// Rebuild full 6‑index F/C/R tensors for all time indices.
// Outputs are time-major: series[t] is a flat N^6 vector (column‑major over j,k,p,q,r,s).
void build_FCR_6d_series(const Tcl4Map& map,
                         const TripleKernelSeries& kernels,
                         std::vector<std::vector<std::complex<double>>>& F_series,
                         std::vector<std::vector<std::complex<double>>>& C_series,
                         std::vector<std::vector<std::complex<double>>>& R_series);

// Composition of TCL2 + TCL4 is intentionally left to application layer
// (e.g., examples/tcl_driver.cpp) to keep modules orthogonal.

// ---------------- High-level TCL4 wrappers ----------------

// Build the TCL4 Liouvillian superoperator L4 at a single time index (acts on vec(rho)).
Eigen::MatrixXcd build_TCL4_generator(const sys::System& system,
                                      const Eigen::MatrixXcd& gamma_series,
                                      double dt,
                                      std::size_t time_index,
                                      FCRMethod method = FCRMethod::Convolution);

// Build the TCL4 Liouvillian superoperator L4 for all time indices [0..Nt-1]. Returns a vector of length Nt.
std::vector<Eigen::MatrixXcd> build_correction_series(const sys::System& system,
                                                      const Eigen::MatrixXcd& gamma_series,
                                                      double dt,
                                                      FCRMethod method = FCRMethod::Convolution);

} // namespace taco::tcl4
