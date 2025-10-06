// integrators.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>

// Composite trapezoid
template<class F, class Real>
Real integrate_trapezoid(F f, Real a, Real b, std::size_t n) {
    if (a == b) return Real(0);
    if (n == 0)  n = 1;
    if (b < a)   return -integrate_trapezoid(f, b, a, n);
    const Real h = (b - a) / Real(n);
    Real sum = Real(0.5) * (f(a) + f(b));
    for (std::size_t i = 1; i < n; ++i) sum += f(a + h * Real(i));
    return h * sum;
}

// Composite Simpson (n must be even)
template<class F, class Real>
Real integrate_simpson(F f, Real a, Real b, std::size_t n) {
    if (a == b) return Real(0);
    if (b < a)  return -integrate_simpson(f, b, a, n);
    if (n < 2)  n = 2;
    if (n % 2)  ++n;  // make even
    const Real h = (b - a) / Real(n);
    Real sum4 = Real(0), sum2 = Real(0);
    for (std::size_t i = 1; i < n; i += 2) sum4 += f(a + h * Real(i));
    for (std::size_t i = 2; i < n; i += 2) sum2 += f(a + h * Real(i));
    const Real fa = f(a), fb = f(b);
    return (h / Real(3)) * (fa + fb + Real(4) * sum4 + Real(2) * sum2);
}

// Adaptive Simpson with error control
namespace detail {
    template<class F, class Real>
    Real simpson_panel(F f, Real a, Real b, Real fa, Real fm, Real fb) {
        const Real h = b - a;
        return (h / Real(6)) * (fa + Real(4) * fm + fb);
    }

    template<class F, class Real>
    Real adaptive_simpson_recur(F f, Real a, Real b, Real eps, int depth,
                                Real fa, Real fm, Real fb, Real whole) {
        const Real m  = (a + b) / Real(2);
        const Real lm = (a + m) / Real(2);
        const Real rm = (m + b) / Real(2);
        const Real flm = f(lm);
        const Real frm = f(rm);
        const Real left  = simpson_panel(f, a, m,  fa, flm, fm);
        const Real right = simpson_panel(f, m, b,  fm, frm, fb);
        const Real diff = std::abs((left + right) - whole);
        if (diff <= Real(15) * eps || depth <= 0) {
            // Richardson correction
            return left + right + (left + right - whole) / Real(15);
        }
        return adaptive_simpson_recur(f, a, m, eps / Real(2), depth - 1, fa, flm, fm, left)
             + adaptive_simpson_recur(f, m, b, eps / Real(2), depth - 1, fm, frm, fb, right);
    }
}

template<class F, class Real>
Real integrate_adaptive_simpson(F f, Real a, Real b, Real tol = Real(1e-8), int maxDepth = 20) {
    if (a == b) return Real(0);
    if (b < a)  return -integrate_adaptive_simpson(f, b, a, tol, maxDepth);
    const Real fa = f(a);
    const Real fb = f(b);
    const Real m  = (a + b) / Real(2);
    const Real fm = f(m);
    const Real whole = detail::simpson_panel(f, a, b, fa, fm, fb);
    return detail::adaptive_simpson_recur(f, a, b, tol, maxDepth, fa, fm, fb, whole);
}

// Integrate over an infinite interval using a change of variables
// Example: (-inf, inf). Map t in (-pi/2, pi/2) via x = tan(t), dx = sec^2(t) dt
template<class F, class Real>
Real integrate_infinite_R(F f, Real tol = Real(1e-8)) {
    auto g = [&](Real t) {
        Real x = std::tan(t);
        Real dxdt = Real(1) / std::cos(t) / std::cos(t); // sec^2
        return f(x) * dxdt;
    };
    const Real a = -Real(M_PI) / Real(2) + Real(1e-6);
    const Real b =  Real(M_PI) / Real(2) - Real(1e-6);
    return integrate_adaptive_simpson(g, a, b, tol);
}

// Integrate discrete samples on a uniform grid using trapezoid
template<class Real, class Vec>
Real integrate_discrete_trapz(const Vec& y, Real dx) {
    if (y.size() < 2) return Real(0);
    Real sum = y.front() + y.back();
    for (std::size_t i = 1; i + 1 < y.size(); ++i) sum += Real(2) * y[i];
    return sum * dx / Real(2);
}
