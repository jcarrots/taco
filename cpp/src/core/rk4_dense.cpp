#include "taco/rk4_dense.hpp"

#include <cmath>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Matrix = Eigen::MatrixXcd;
using Vector = Eigen::VectorXcd;

#if defined(_OPENMP)
#  if !defined(_MSC_VER) && (_OPENMP >= 201307)
#    define TACO_OMP_FOR_SIMD _Pragma("omp for simd schedule(static)")
#  else
#    define TACO_OMP_FOR_SIMD _Pragma("omp for schedule(static)")
#  endif
#endif

void ensure_dims(const Matrix& L, const Vector& r) {
    if (L.rows() != L.cols()) {
        throw std::invalid_argument("rk4_dense: L must be square");
    }
    if (L.rows() != r.size()) {
        throw std::invalid_argument("rk4_dense: L and r dimension mismatch");
    }
}

void ensure_square_and_size(const Matrix& L, Eigen::Index n, const char* label) {
    if (L.rows() != L.cols()) {
        throw std::invalid_argument(std::string("rk4_dense: ") + label + " must be square");
    }
    if (L.rows() != n) {
        throw std::invalid_argument(std::string("rk4_dense: ") + label + " dimension mismatch");
    }
}

#ifdef _OPENMP
void resize_vec_omp(Vector& v, Eigen::Index n) {
    if (v.size() == n) return;
    if (omp_in_parallel()) {
        #pragma omp single
        v.resize(n);
        #pragma omp barrier
    } else {
        v.resize(n);
    }
}

void vec_copy_omp(const Vector& x, Vector& y) {
    const Eigen::Index D = x.size();
    resize_vec_omp(y, D);
    TACO_OMP_FOR_SIMD
    for (Eigen::Index i = 0; i < D; ++i) {
        y(i) = x(i);
    }
}

void vec_axpy_omp(double alpha, const Vector& x, Vector& y) {
    const Eigen::Index D = x.size();
    TACO_OMP_FOR_SIMD
    for (Eigen::Index i = 0; i < D; ++i) {
        y(i) += alpha * x(i);
    }
}
#endif

} // namespace

namespace taco::tcl {

void Rk4DenseWorkspace::resize(Eigen::Index n) {
    if (k1.size() == n) return;
    k1.resize(n);
    k2.resize(n);
    k3.resize(n);
    k4.resize(n);
    tmp.resize(n);
}

void matvec_serial(const Matrix& L, const Vector& x, Vector& y) {
    ensure_dims(L, x);
    const Eigen::Index D = L.rows();
    y.resize(D);
    for (Eigen::Index i = 0; i < D; ++i) {
        std::complex<double> sum(0.0, 0.0);
        for (Eigen::Index j = 0; j < D; ++j) {
            sum += L(i, j) * x(j);
        }
        y(i) = sum;
    }
}

void matvec_omp(const Matrix& L, const Vector& x, Vector& y) {
#ifdef _OPENMP
    ensure_dims(L, x);
    const Eigen::Index D = L.rows();
    resize_vec_omp(y, D);
    if (omp_in_parallel()) {
        #pragma omp for schedule(static)
        for (Eigen::Index i = 0; i < D; ++i) {
            std::complex<double> sum(0.0, 0.0);
            for (Eigen::Index j = 0; j < D; ++j) {
                sum += L(i, j) * x(j);
            }
            y(i) = sum;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < D; ++i) {
            std::complex<double> sum(0.0, 0.0);
            for (Eigen::Index j = 0; j < D; ++j) {
                sum += L(i, j) * x(j);
            }
            y(i) = sum;
        }
    }
#else
    matvec_serial(L, x, y);
#endif
}

void rk4_update_serial(const Matrix& L,
                       Vector& r,
                       Rk4DenseWorkspace& ws,
                       double dt) {
    rk4_update_serial(L, L, L, r, ws, dt);
}

void rk4_update_serial(const Matrix& L0,
                       const Matrix& Lhalf,
                       const Matrix& L1,
                       Vector& r,
                       Rk4DenseWorkspace& ws,
                       double dt) {
    if (!(dt > 0.0)) throw std::invalid_argument("rk4_update_serial: dt must be > 0");
    ensure_dims(L0, r);
    ensure_square_and_size(Lhalf, L0.rows(), "Lhalf");
    ensure_square_and_size(L1, L0.rows(), "L1");
    ws.resize(r.size());

    matvec_serial(L0, r, ws.k1);

    ws.tmp = r;
    ws.tmp.noalias() += 0.5 * dt * ws.k1;
    matvec_serial(Lhalf, ws.tmp, ws.k2);

    ws.tmp = r;
    ws.tmp.noalias() += 0.5 * dt * ws.k2;
    matvec_serial(Lhalf, ws.tmp, ws.k3);

    ws.tmp = r;
    ws.tmp.noalias() += dt * ws.k3;
    matvec_serial(L1, ws.tmp, ws.k4);

    r.noalias() += (dt / 6.0) * (ws.k1 + 2.0 * ws.k2 + 2.0 * ws.k3 + ws.k4);
}

void rk4_update_omp(const Matrix& L,
                    Vector& r,
                    Rk4DenseWorkspace& ws,
                    double dt) {
    rk4_update_omp(L, L, L, r, ws, dt);
}

void rk4_update_omp(const Matrix& L0,
                    const Matrix& Lhalf,
                    const Matrix& L1,
                    Vector& r,
                    Rk4DenseWorkspace& ws,
                    double dt) {
#ifdef _OPENMP
    if (!(dt > 0.0)) throw std::invalid_argument("rk4_update_omp: dt must be > 0");
    ensure_dims(L0, r);
    ensure_square_and_size(Lhalf, L0.rows(), "Lhalf");
    ensure_square_and_size(L1, L0.rows(), "L1");
    ws.resize(r.size());

    #pragma omp parallel
    {
        matvec_omp(L0, r, ws.k1);

        vec_copy_omp(r, ws.tmp);
        vec_axpy_omp(0.5 * dt, ws.k1, ws.tmp);
        matvec_omp(Lhalf, ws.tmp, ws.k2);

        vec_copy_omp(r, ws.tmp);
        vec_axpy_omp(0.5 * dt, ws.k2, ws.tmp);
        matvec_omp(Lhalf, ws.tmp, ws.k3);

        vec_copy_omp(r, ws.tmp);
        vec_axpy_omp(dt, ws.k3, ws.tmp);
        matvec_omp(L1, ws.tmp, ws.k4);

        TACO_OMP_FOR_SIMD
        for (Eigen::Index i = 0; i < r.size(); ++i) {
            r(i) += (dt / 6.0) * (ws.k1(i) + 2.0 * ws.k2(i) + 2.0 * ws.k3(i) + ws.k4(i));
        }
    }
#else
    rk4_update_serial(L0, Lhalf, L1, r, ws, dt);
#endif
}

void rk4_dense_step_serial(const Matrix& L,
                           Vector& r,
                           Rk4DenseWorkspace& ws,
                           double dt) {
    rk4_update_serial(L, r, ws, dt);
}

void rk4_dense_step_omp(const Matrix& L,
                        Vector& r,
                        Rk4DenseWorkspace& ws,
                        double dt) {
    rk4_update_omp(L, r, ws, dt);
}

void propagate_rk4_dense_serial(const Matrix& L,
                                Vector& r,
                                double t0,
                                double tf,
                                double dt,
                                const std::function<void(double,const Vector&)>& on_sample,
                                std::size_t sample_every) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_rk4_dense_serial: dt must be > 0");
    if (sample_every == 0) throw std::invalid_argument("propagate_rk4_dense_serial: sample_every must be > 0");
    ensure_dims(L, r);
    Rk4DenseWorkspace ws;
    ws.resize(r.size());

    double t = t0;
    std::size_t step = 0;
    while (t + 0.5 * dt <= tf + 1e-15) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, r);
        rk4_update_serial(L, r, ws, dt);
        t += dt;
        ++step;
    }
    if (on_sample) on_sample(t, r);
}

void propagate_rk4_dense_serial(const std::vector<Matrix>& L_series,
                                Vector& r,
                                double t0,
                                double dt,
                                const std::function<void(double,const Vector&)>& on_sample,
                                std::size_t sample_every) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_rk4_dense_serial: dt must be > 0");
    if (sample_every == 0) throw std::invalid_argument("propagate_rk4_dense_serial: sample_every must be > 0");
    if (L_series.size() < 2) throw std::invalid_argument("propagate_rk4_dense_serial: L_series must have at least 2 matrices (endpoints)");

    const Eigen::Index n = L_series.front().rows();
    ensure_square_and_size(L_series.front(), n, "L_series");
    if (r.size() != n) {
        throw std::invalid_argument("propagate_rk4_dense_serial: L_series and r dimension mismatch");
    }
    for (std::size_t i = 1; i < L_series.size(); ++i) {
        ensure_square_and_size(L_series[i], n, "L_series");
    }

    Rk4DenseWorkspace ws;
    ws.resize(r.size());
    Matrix Lhalf;

    double t = t0;
    std::size_t step = 0;
    const std::size_t steps = L_series.size() - 1;
    for (std::size_t i = 0; i < steps; ++i) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, r);
        Lhalf.noalias() = 0.5 * (L_series[i] + L_series[i + 1]);
        rk4_update_serial(L_series[i], Lhalf, L_series[i + 1], r, ws, dt);
        t += dt;
        ++step;
    }
    if (on_sample) on_sample(t, r);
}

void propagate_rk4_dense_serial(const std::vector<Matrix>& L_series,
                                const std::vector<Matrix>& L_half_series,
                                Vector& r,
                                double t0,
                                double dt,
                                const std::function<void(double,const Vector&)>& on_sample,
                                std::size_t sample_every) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_rk4_dense_serial: dt must be > 0");
    if (sample_every == 0) throw std::invalid_argument("propagate_rk4_dense_serial: sample_every must be > 0");
    if (L_series.size() < 2) throw std::invalid_argument("propagate_rk4_dense_serial: L_series must have at least 2 matrices (endpoints)");
    if (L_half_series.size() != L_series.size() - 1) {
        throw std::invalid_argument("propagate_rk4_dense_serial: L_half_series must match L_series size - 1 (midpoints)");
    }

    const Eigen::Index n = L_series.front().rows();
    ensure_square_and_size(L_series.front(), n, "L_series");
    if (r.size() != n) {
        throw std::invalid_argument("propagate_rk4_dense_serial: L_series and r dimension mismatch");
    }
    for (std::size_t i = 1; i < L_series.size(); ++i) {
        ensure_square_and_size(L_series[i], n, "L_series");
    }
    for (std::size_t i = 0; i < L_half_series.size(); ++i) {
        ensure_square_and_size(L_half_series[i], n, "L_half_series");
    }

    Rk4DenseWorkspace ws;
    ws.resize(r.size());

    double t = t0;
    std::size_t step = 0;
    const std::size_t steps = L_series.size() - 1;
    for (std::size_t i = 0; i < steps; ++i) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, r);
        rk4_update_serial(L_series[i], L_half_series[i], L_series[i + 1], r, ws, dt);
        t += dt;
        ++step;
    }
    if (on_sample) on_sample(t, r);
}

void propagate_rk4_dense_omp(const Matrix& L,
                             Vector& r,
                             double t0,
                             double tf,
                             double dt,
                             const std::function<void(double,const Vector&)>& on_sample,
                             std::size_t sample_every) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_rk4_dense_omp: dt must be > 0");
    if (sample_every == 0) throw std::invalid_argument("propagate_rk4_dense_omp: sample_every must be > 0");
    ensure_dims(L, r);
    Rk4DenseWorkspace ws;
    ws.resize(r.size());

    double t = t0;
    std::size_t step = 0;
    while (t + 0.5 * dt <= tf + 1e-15) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, r);
        rk4_update_omp(L, r, ws, dt);
        t += dt;
        ++step;
    }
    if (on_sample) on_sample(t, r);
}

void propagate_rk4_dense_omp(const std::vector<Matrix>& L_series,
                             Vector& r,
                             double t0,
                             double dt,
                             const std::function<void(double,const Vector&)>& on_sample,
                             std::size_t sample_every) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_rk4_dense_omp: dt must be > 0");
    if (sample_every == 0) throw std::invalid_argument("propagate_rk4_dense_omp: sample_every must be > 0");
    if (L_series.size() < 2) throw std::invalid_argument("propagate_rk4_dense_omp: L_series must have at least 2 matrices (endpoints)");

    const Eigen::Index n = L_series.front().rows();
    ensure_square_and_size(L_series.front(), n, "L_series");
    if (r.size() != n) {
        throw std::invalid_argument("propagate_rk4_dense_omp: L_series and r dimension mismatch");
    }
    for (std::size_t i = 1; i < L_series.size(); ++i) {
        ensure_square_and_size(L_series[i], n, "L_series");
    }

    Rk4DenseWorkspace ws;
    ws.resize(r.size());
    Matrix Lhalf;

    double t = t0;
    std::size_t step = 0;
    const std::size_t steps = L_series.size() - 1;
    for (std::size_t i = 0; i < steps; ++i) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, r);
        Lhalf.noalias() = 0.5 * (L_series[i] + L_series[i + 1]);
        rk4_update_omp(L_series[i], Lhalf, L_series[i + 1], r, ws, dt);
        t += dt;
        ++step;
    }
    if (on_sample) on_sample(t, r);
}

void propagate_rk4_dense_omp(const std::vector<Matrix>& L_series,
                             const std::vector<Matrix>& L_half_series,
                             Vector& r,
                             double t0,
                             double dt,
                             const std::function<void(double,const Vector&)>& on_sample,
                             std::size_t sample_every) {
    if (!(dt > 0.0)) throw std::invalid_argument("propagate_rk4_dense_omp: dt must be > 0");
    if (sample_every == 0) throw std::invalid_argument("propagate_rk4_dense_omp: sample_every must be > 0");
    if (L_series.size() < 2) throw std::invalid_argument("propagate_rk4_dense_omp: L_series must have at least 2 matrices (endpoints)");
    if (L_half_series.size() != L_series.size() - 1) {
        throw std::invalid_argument("propagate_rk4_dense_omp: L_half_series must match L_series size - 1 (midpoints)");
    }

    const Eigen::Index n = L_series.front().rows();
    ensure_square_and_size(L_series.front(), n, "L_series");
    if (r.size() != n) {
        throw std::invalid_argument("propagate_rk4_dense_omp: L_series and r dimension mismatch");
    }
    for (std::size_t i = 1; i < L_series.size(); ++i) {
        ensure_square_and_size(L_series[i], n, "L_series");
    }
    for (std::size_t i = 0; i < L_half_series.size(); ++i) {
        ensure_square_and_size(L_half_series[i], n, "L_half_series");
    }

    Rk4DenseWorkspace ws;
    ws.resize(r.size());

    double t = t0;
    std::size_t step = 0;
    const std::size_t steps = L_series.size() - 1;
    for (std::size_t i = 0; i < steps; ++i) {
        if (on_sample && (step % sample_every == 0)) on_sample(t, r);
        rk4_update_omp(L_series[i], L_half_series[i], L_series[i + 1], r, ws, dt);
        t += dt;
        ++step;
    }
    if (on_sample) on_sample(t, r);
}

} // namespace taco::tcl
