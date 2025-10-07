// integrators.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

// FFT helpers (pocketfft preferred) for FFT-based convolution
#include "taco/correlation_fft.hpp"

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


// Integrate over an infinite interval using a change of variables
// Example: (-inf, inf). Map t in (-pi/2, pi/2) via x = tan(t), dx = sec^2(t) dt
template<class F, class Real>
Real integrate_infinite_R(F f, Real tol = Real(1e-8)) {
    auto g = [&](Real t) {
        Real x = std::tan(t);
        Real dxdt = Real(1) / std::cos(t) / std::cos(t); // sec^2
        return f(x) * dxdt;
    };
    const Real pi = Real(3.141592653589793238462643383279502884L);
    const Real a = -pi / Real(2) + Real(1e-6);
    const Real b =  pi / Real(2) - Real(1e-6);
    // Use composite Simpson with sufficiently fine partition
    std::size_t n = 4096;
    return integrate_simpson(g, a, b, n);
}

// Integrate discrete samples on a uniform grid using trapezoid
template<class Real, class Vec>
Real integrate_discrete_trapz(const Vec& y, Real dx) {
    if (y.size() < 2) return Real(0);
    Real sum = y.front() + y.back();
    for (std::size_t i = 1; i + 1 < y.size(); ++i) sum += Real(2) * y[i];
    return sum * dx / Real(2);
}

// Cumulative trapezoid integral on uniform grid: I[i] = ∫_0^{i*dx} y dt
template<class Vec, class Real>
auto cumulative_trapz(const Vec& y, Real dx)
    -> std::vector<decltype(y[0] * dx)>
{
    using T = decltype(y[0] * dx);
    const std::size_t n = y.size();
    std::vector<T> I(n);
    if (n == 0) return I;
    I[0] = T(0);
    for (std::size_t i = 1; i < n; ++i) I[i] = I[i-1] + (dx / Real(2)) * (y[i-1] + y[i]);
    return I;
}

// Convolution support
enum class ConvMode { Full, Same, Valid };

// Time-domain convolution with trapezoid end-weights
template<class VecA, class VecB, class Real>
auto convolve_trapz(const VecA& a, const VecB& b, Real dx, ConvMode mode = ConvMode::Full)
    -> std::vector<decltype(a[0] * b[0])>
{
    using T = decltype(a[0] * b[0]);
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    if (n == 0 || m == 0) return {};
    const std::size_t Lfull = n + m - 1;
    std::vector<T> full(Lfull, T(0));
    for (std::size_t k = 0; k < Lfull; ++k) {
        const std::size_t i0 = (k >= (m - 1)) ? (k - (m - 1)) : 0;
        const std::size_t i1 = (k < (n - 1)) ? k : (n - 1);
        if (i0 > i1) continue;
        T acc = T(0);
        for (std::size_t i = i0; i <= i1; ++i) {
            const std::size_t j = k - i;
            const bool end_i = (i == i0) || (i == i1);
            const Real w = end_i ? (dx / Real(2)) : dx;
            acc += w * a[i] * b[j];
        }
        full[k] = acc;
    }
    if (mode == ConvMode::Full) return full;
    if (mode == ConvMode::Valid) {
        if (n < m) return {};
        const std::size_t L = n - m + 1;
        std::vector<T> out(L);
        const std::size_t start = m - 1;
        for (std::size_t i = 0; i < L; ++i) out[i] = full[start + i];
        return out;
    }
    const std::size_t offset = (m - 1) / 2;
    std::vector<T> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = full[i + offset];
    return out;
}

// FFT-based convolution with endpoint trapezoid weighting applied once
template<class VecA, class VecB, class Real>
auto convolve_fft(const VecA& a, const VecB& b, Real dx, ConvMode mode = ConvMode::Full)
    -> std::vector<decltype(a[0] * b[0])>
{
    using Tout = decltype(a[0] * b[0]);
    using cd = std::complex<double>;
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    if (n == 0 || m == 0) return {};
    const std::size_t Lfull = n + m - 1;
    std::size_t Nfft = bcf::next_pow2(Lfull*16); 
    if (Nfft < 2) Nfft = 2;
    std::vector<cd> A(Nfft, cd(0.0, 0.0)), B(Nfft, cd(0.0, 0.0));
    auto to_cd = [](auto v) -> cd {
        using V = decltype(v);
        if constexpr (std::is_same_v<std::decay_t<V>, cd>) return v;
        else if constexpr (std::is_same_v<std::decay_t<V>, std::complex<float>>) return cd(v.real(), v.imag());
        else if constexpr (std::is_same_v<std::decay_t<V>, double>) return cd(v, 0.0);
        else if constexpr (std::is_same_v<std::decay_t<V>, float>) return cd(static_cast<double>(v), 0.0);
        else return cd(static_cast<double>(v), 0.0);
    };
    for (std::size_t i = 0; i < n; ++i) {
        const double wi = (i == 0 || i + 1 == n) ? 0.5 : 1.0;
        A[i] = to_cd(a[i]) * wi;
    }
    for (std::size_t j = 0; j < m; ++j) {
        const double wj = (j == 0 || j + 1 == m) ? 0.5 : 1.0;
        B[j] = to_cd(b[j]) * wj;
    }
    bcf::FFTPlan plan(Nfft);
    plan.exec_forward(A);
    plan.exec_forward(B);
    for (std::size_t k = 0; k < Nfft; ++k) A[k] *= B[k];
    plan.exec_inverse(A);
    for (auto& x : A) x *= static_cast<double>(dx);
    std::vector<Tout> full(Lfull);
    for (std::size_t k = 0; k < Lfull; ++k) {
        const cd v = A[k];
        if constexpr (std::is_same_v<Tout, cd>) full[k] = v;
        else if constexpr (std::is_same_v<Tout, std::complex<float>>) full[k] = std::complex<float>(static_cast<float>(v.real()), static_cast<float>(v.imag()));
        else full[k] = static_cast<Tout>(v.real());
    }
    if (mode == ConvMode::Full) return full;
    if (mode == ConvMode::Valid) {
        if (n < m) return {};
        const std::size_t L = n - m + 1;
        std::vector<Tout> out(L);
        const std::size_t start = m - 1;
        for (std::size_t i = 0; i < L; ++i) out[i] = full[start + i];
        return out;
    }
    const std::size_t offset = (m - 1) / 2;
    std::vector<Tout> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = full[i + offset];
    return out;
}
