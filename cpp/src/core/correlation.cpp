#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cstddef>

#if defined(TACO_USE_POCKETFFT)
#  if __has_include(<pocketfft/pocketfft.h>)
#    include <pocketfft/pocketfft.h>
#    define TACO_POCKETFFT_HEADER 1
#  elif __has_include("pocketfft_hdronly.h")
#    include "pocketfft_hdronly.h"
#    define TACO_POCKETFFT_HEADER 1
#  endif
#endif

namespace bcf {

// --- types & constants ---
using cd = std::complex<double>;
constexpr double PI = 3.141592653589793238462643383279502884;

// Next power of two (for FFT length)
inline std::size_t next_pow2(std::size_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;  n |= n >> 2;  n |= n >> 4;
    n |= n >> 8;  n |= n >> 16;
    if constexpr (sizeof(std::size_t) >= 8) n |= n >> 32;
    return ++n;
}

// In-house radix-2 Cooley–Tukey FFT (power-of-two lengths)
// Directional variant: forward=true uses e^{-i 2π kn/N}; inverse scales by 1/N.
inline void fft_inplace_builtin_dir(std::vector<cd>& a, bool forward) {
    const std::size_t n = a.size();
    if (n == 0) return;
    if ((n & (n - 1)) != 0) throw std::runtime_error("fft_inplace: length must be power of two");

    // bit-reversal permutation
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Danielson–Lanczos
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = (forward ? -1.0 : +1.0) * 2.0 * PI / static_cast<double>(len);
        const cd wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            const std::size_t half = len >> 1;
            for (std::size_t k = 0; k < half; ++k) {
                const cd u = a[i + k];
                const cd v = w * a[i + k + half];
                a[i + k]        = u + v;
                a[i + k + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (!forward) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto &x : a) x *= inv_n;
    }
}

// Unified FFT entry point: uses pocketfft if available, else built-in radix-2
inline void fft_inplace(std::vector<cd>& a) {
#if defined(TACO_POCKETFFT_HEADER)
    const std::size_t n = a.size();
    if (n == 0) return;
    std::vector<cd> out(n);
    // pocketfft expects shapes/strides in bytes
    pocketfft::shape_t shape{ static_cast<std::ptrdiff_t>(n) };
    pocketfft::stride_t stride_in{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::stride_t stride_out{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::axes_t axes{ 0 };
    pocketfft::c2c(shape, stride_in, stride_out, axes,
                   /*forward=*/true,
                   reinterpret_cast<const cd*>(a.data()),
                   out.data(),
                   /*fct=*/1.0);
    a.swap(out);
#else
    fft_inplace_builtin_dir(a, /*forward=*/true);
#endif
}

inline void ifft_inplace(std::vector<cd>& a) {
#if defined(TACO_POCKETFFT_HEADER)
    const std::size_t n = a.size();
    if (n == 0) return;
    std::vector<cd> out(n);
    pocketfft::shape_t shape{ static_cast<std::ptrdiff_t>(n) };
    pocketfft::stride_t stride_in{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::stride_t stride_out{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::axes_t axes{ 0 };
    // inverse: forward=false; scale by 1/n to match unitary-style normalization
    pocketfft::c2c(shape, stride_in, stride_out, axes,
                   /*forward=*/false,
                   reinterpret_cast<const cd*>(a.data()),
                   out.data(),
                   /*fct=*/1.0/static_cast<double>(n));
    a.swap(out);
#else
    fft_inplace_builtin_dir(a, /*forward=*/false);
#endif
}

// Tiny plan object to reuse scratch/state between repeated transforms of same length
struct FFTPlan {
    std::size_t n{0};
#if defined(TACO_POCKETFFT_HEADER)
    pocketfft::shape_t shape{0};
    pocketfft::stride_t stride_in{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::stride_t stride_out{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::axes_t axes{ 0 };
    std::vector<cd> scratch;
    explicit FFTPlan(std::size_t len)
        : n(len), shape{ static_cast<std::ptrdiff_t>(len) } {
        scratch.reserve(n);
    }
    void exec_forward(std::vector<cd>& a) {
        if (a.size() != n) throw std::invalid_argument("FFTPlan.exec_forward: size mismatch");
        scratch.resize(n);
        pocketfft::c2c(shape, stride_in, stride_out, axes,
                       /*forward=*/true,
                       reinterpret_cast<const cd*>(a.data()),
                       scratch.data(),
                       /*fct=*/1.0);
        a.swap(scratch);
    }
    void exec_inverse(std::vector<cd>& a) {
        if (a.size() != n) throw std::invalid_argument("FFTPlan.exec_inverse: size mismatch");
        scratch.resize(n);
        pocketfft::c2c(shape, stride_in, stride_out, axes,
                       /*forward=*/false,
                       reinterpret_cast<const cd*>(a.data()),
                       scratch.data(),
                       /*fct=*/1.0/static_cast<double>(n));
        a.swap(scratch);
    }
#else
    explicit FFTPlan(std::size_t len) : n(len) {
        if ((n & (n - 1)) != 0) throw std::runtime_error("FFTPlan: length must be power of two");
    }
    void exec_forward(std::vector<cd>& a) {
        if (a.size() != n) throw std::invalid_argument("FFTPlan.exec_forward: size mismatch");
        fft_inplace_builtin_dir(a, /*forward=*/true);
    }
    void exec_inverse(std::vector<cd>& a) {
        if (a.size() != n) throw std::invalid_argument("FFTPlan.exec_inverse: size mismatch");
        fft_inplace_builtin_dir(a, /*forward=*/false);
    }
#endif
};

/**
 * Compute BCF C(t) from a spectral density J(ω) given as a callable (lambda, func ptr, functor).
 *
 * Inputs:
 *  - N      : you want samples t = 0, dt, ..., N*dt  (returns N+1 points)
 *  - dt     : fixed time step
 *  - J      : callable with signature double(double) that returns J(ω) for ω>=0
 *  - beta   : inverse temperature (use std::numeric_limits<double>::infinity() for T=0)
 *
 * Outputs (by reference):
 *  - t      : size N+1, t[n] = n*dt
 *  - C      : size N+1, complex BCF at those times
 */
template<class JCallable>
void bcf_fft_fun(std::size_t N, double dt, JCallable J, double beta,
                 std::vector<double>& t, std::vector<cd>& C)
{
    if (!(dt > 0.0)) throw std::invalid_argument("dt must be > 0");

    const std::size_t Nt   = N + 1;          // number of returned time samples
    std::size_t Nfft       = 2 * Nt;         // symmetric spectrum length
    if ((Nfft & (Nfft - 1)) != 0) Nfft = next_pow2(Nfft);  // round up for FFT

    const double tf     = dt * static_cast<double>(Nfft);  // total period
    const double domega = 2.0 * PI / tf;                   // frequency spacing
    const double omegaNy= PI / dt;                         // Nyquist (fixed by dt)
    const std::size_t posCount = Nfft/2 - 1;               // # of strictly +ω bins

    // Build S(ωk) on [DC, +ω, Nyquist, -ω] with half-weights at DC/Nyquist.
    std::vector<cd> S(Nfft, cd(0.0, 0.0));

    if (std::isinf(beta)) {
        // T = 0: S(+ω) = (π/2) J(ω); negative side = 0
        for (std::size_t k = 1; k <= posCount; ++k) {
            const double omega = domega * static_cast<double>(k);
            S[k] = cd( (PI/2.0) * J(omega), 0.0 );
        }
        S[0]         = cd(0.0, 0.0);
        S[Nfft/2]    = cd( 0.25 * PI * J(omegaNy), 0.0 );
    } else {
        // finite T: detailed balance S(-ω) = e^{-βω} S(+ω)
        for (std::size_t k = 1; k <= posCount; ++k) {
            const double omega = domega * static_cast<double>(k);
            const double Jval  = J(omega);
            const double KMS   = std::exp(-beta * omega);  // e^{-βω}
            S[k]               = cd( (PI/2.0) * Jval / (1.0 - KMS), 0.0 );
            S[Nfft - k]        = S[k] * KMS;               // S(-ω)
        }
        // DC via discrete ω→0 limit using ω=dω
        const double ed = std::exp(-beta * domega);
        S[0] = cd( 0.5 * (PI/2.0) * J(domega) * (1.0 + ed) / (1.0 - ed), 0.0 );

        // Nyquist: coth(βω/2) = 1 + 2/(e^{βω} - 1) for stability
        const double eNy = std::exp(beta * omegaNy);
        S[Nfft/2] = cd( 0.25 * PI * J(omegaNy) * (1.0 + 2.0 / (eNy - 1.0)), 0.0 );
    }

    // Inverse FFT → time domain using a tiny plan to reuse scratch
    {
        FFTPlan plan(Nfft);
        plan.exec_inverse(S);
    }
    const double scale = domega / PI;  // matches continuum normalization

    // Fill outputs (keep t >= 0)
    t.resize(Nt);
    C.resize(Nt);
    for (std::size_t n = 0; n < Nt; ++n) {
        t[n] = dt * static_cast<double>(n);
        C[n] = S[n] * scale;
    }
}

} // namespace bcf
