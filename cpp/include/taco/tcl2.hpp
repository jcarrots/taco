#pragma once

#include <cstddef>
#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>


#include "taco/bath.hpp"

namespace taco::tcl {

using complex = std::complex<double>;
using real = double;

struct JumpOperator {
    std::string label;
    // operator in the lab basis
    Eigen::MatrixXcd matrix;
};

struct GeneratorOptions {
    double frequency_tolerance = 1.0e-9;
    double transition_cutoff = 1.0e-12;
    double gamma_cutoff = 1.0e-14;
    double integration_step_hint = 0.02;
    std::size_t integration_min_subdivisions = 6;
    std::size_t integration_max_subdivisions = 2046;
};

class TCL2Generator {
public:
    TCL2Generator(const Eigen::MatrixXcd& hamiltonian,
                  std::vector<JumpOperator> jump_ops,
                  const bath::CorrelationFunction& correlation,
                  const GeneratorOptions& options = GeneratorOptions{});
    const Eigen::MatrixXcd& eigenvectors() const noexcept { return U_; }
    const Eigen::VectorXd& eigenvalues() const noexcept { return eps_; }

    void reset(double t0 = 0.0);

    void advance(double t1);

    void apply(const Eigen::MatrixXcd& rho_lab, Eigen::MatrixXcd& drho_lab) const;

    std::size_t dimension() const noexcept { return dim_; }
    std::size_t num_jump_operators() const noexcept { return jump_ops_; }
    double current_time() const noexcept { return current_time_; }

private:
    struct FrequencyBucket {
        double omega = 0.0;
        std::vector<Eigen::MatrixXcd> A;
        std::vector<Eigen::MatrixXcd> A_dag;
        // flattened alpha,beta -> matrix
        std::vector<Eigen::MatrixXcd> A_dag_A;
        Eigen::MatrixXcd G;
        Eigen::MatrixXd gamma;
        // Lamb-shift contribution in eigen basis
        Eigen::MatrixXcd H_ls;
    };

    complex weighted_integral(double omega, double a, double b, std::size_t alpha, std::size_t beta) const;
    std::size_t bucket_index_for(double omega) const;
    void rebuild_bucket_caches();
    void rebuild_lamb_shift();

    const bath::CorrelationFunction& bath_;
    GeneratorOptions opts_;
    std::size_t dim_ = 0;
    std::size_t jump_ops_ = 0;
    double current_time_ = 0.0;

    Eigen::MatrixXcd U_;
    Eigen::MatrixXcd U_dag_;
    Eigen::VectorXd eps_;
    Eigen::MatrixXcd H_ls_eig_;

    std::vector<JumpOperator> lab_jump_ops_;
    std::vector<FrequencyBucket> buckets_;
};

}  // namespace taco::tcl
