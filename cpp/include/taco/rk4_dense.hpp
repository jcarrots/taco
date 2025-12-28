#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

namespace taco::tcl {

// Workspace buffers for dense RK4 integration.
struct Rk4DenseWorkspace {
    Eigen::VectorXcd k1;
    Eigen::VectorXcd k2;
    Eigen::VectorXcd k3;
    Eigen::VectorXcd k4;
    Eigen::VectorXcd tmp;

    void resize(Eigen::Index n);
};

// Dense matrix-vector multiply: y = L * x (serial).
void matvec_serial(const Eigen::MatrixXcd& L,
                   const Eigen::VectorXcd& x,
                   Eigen::VectorXcd& y);

// Dense matrix-vector multiply: y = L * x (OpenMP if enabled, otherwise serial fallback).
void matvec_omp(const Eigen::MatrixXcd& L,
                const Eigen::VectorXcd& x,
                Eigen::VectorXcd& y);

// One RK4 update for r' = L r (constant L, serial).
void rk4_update_serial(const Eigen::MatrixXcd& L,
                       Eigen::VectorXcd& r,
                       Rk4DenseWorkspace& ws,
                       double dt);

// One RK4 update for r' = L(t) r with L at t, t+dt/2, t+dt (serial).
void rk4_update_serial(const Eigen::MatrixXcd& L0,
                       const Eigen::MatrixXcd& Lhalf,
                       const Eigen::MatrixXcd& L1,
                       Eigen::VectorXcd& r,
                       Rk4DenseWorkspace& ws,
                       double dt);

// One RK4 update for r' = L r (constant L, OpenMP if enabled, otherwise serial fallback).
void rk4_update_omp(const Eigen::MatrixXcd& L,
                    Eigen::VectorXcd& r,
                    Rk4DenseWorkspace& ws,
                    double dt);

// One RK4 update for r' = L(t) r with L at t, t+dt/2, t+dt (OpenMP if enabled).
void rk4_update_omp(const Eigen::MatrixXcd& L0,
                    const Eigen::MatrixXcd& Lhalf,
                    const Eigen::MatrixXcd& L1,
                    Eigen::VectorXcd& r,
                    Rk4DenseWorkspace& ws,
                    double dt);

// One RK4 step for r' = L r (serial).
void rk4_dense_step_serial(const Eigen::MatrixXcd& L,
                           Eigen::VectorXcd& r,
                           Rk4DenseWorkspace& ws,
                           double dt);

// One RK4 step for r' = L r (OpenMP if enabled, otherwise serial fallback).
void rk4_dense_step_omp(const Eigen::MatrixXcd& L,
                        Eigen::VectorXcd& r,
                        Rk4DenseWorkspace& ws,
                        double dt);

// Propagate r' = L r with a fixed step (serial).
void propagate_rk4_dense_serial(const Eigen::MatrixXcd& L,
                                Eigen::VectorXcd& r,
                                double t0,
                                double tf,
                                double dt,
                                const std::function<void(double,const Eigen::VectorXcd&)>& on_sample = {},
                                std::size_t sample_every = 1);

// Propagate r' = L(t) r with a fixed step using prebuilt endpoints (serial).
// L_series size = steps + 1. Midpoints are estimated as 0.5 * (L_i + L_{i+1}).
// For best RK4 accuracy, prefer the L_half_series overload.
void propagate_rk4_dense_serial(const std::vector<Eigen::MatrixXcd>& L_series,
                                Eigen::VectorXcd& r,
                                double t0,
                                double dt,
                                const std::function<void(double,const Eigen::VectorXcd&)>& on_sample = {},
                                std::size_t sample_every = 1);

// Propagate r' = L(t) r with prebuilt endpoints and midpoints (serial).
// L_series size = steps + 1, L_half_series size = steps.
void propagate_rk4_dense_serial(const std::vector<Eigen::MatrixXcd>& L_series,
                                const std::vector<Eigen::MatrixXcd>& L_half_series,
                                Eigen::VectorXcd& r,
                                double t0,
                                double dt,
                                const std::function<void(double,const Eigen::VectorXcd&)>& on_sample = {},
                                std::size_t sample_every = 1);

// Propagate r' = L r with a fixed step (OpenMP if enabled, otherwise serial fallback).
void propagate_rk4_dense_omp(const Eigen::MatrixXcd& L,
                             Eigen::VectorXcd& r,
                             double t0,
                             double tf,
                             double dt,
                             const std::function<void(double,const Eigen::VectorXcd&)>& on_sample = {},
                             std::size_t sample_every = 1);

// Propagate r' = L(t) r with a fixed step using prebuilt endpoints (OpenMP if enabled).
// L_series size = steps + 1. Midpoints are estimated as 0.5 * (L_i + L_{i+1}).
// For best RK4 accuracy, prefer the L_half_series overload.
void propagate_rk4_dense_omp(const std::vector<Eigen::MatrixXcd>& L_series,
                             Eigen::VectorXcd& r,
                             double t0,
                             double dt,
                             const std::function<void(double,const Eigen::VectorXcd&)>& on_sample = {},
                             std::size_t sample_every = 1);

// Propagate r' = L(t) r with prebuilt endpoints and midpoints (OpenMP if enabled).
// L_series size = steps + 1, L_half_series size = steps.
void propagate_rk4_dense_omp(const std::vector<Eigen::MatrixXcd>& L_series,
                             const std::vector<Eigen::MatrixXcd>& L_half_series,
                             Eigen::VectorXcd& r,
                             double t0,
                             double dt,
                             const std::function<void(double,const Eigen::VectorXcd&)>& on_sample = {},
                             std::size_t sample_every = 1);

} // namespace taco::tcl
