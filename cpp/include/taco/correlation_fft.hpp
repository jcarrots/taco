#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

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

using cd = std::complex<double>;
inline constexpr double PI = 3.141592653589793238462643383279502884;

// ----- Small helpers (header-only; keep inline) -----
inline std::size_t next_pow2(std::size_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;  n |= n >> 2;  n |= n >> 4;
    n |= n >> 8;  n |= n >> 16;
    if constexpr (sizeof(std::size_t) >= 8) n |= n >> 32;
    return ++n;
}

// In-house radix-2 Cooley–Tukey FFT (power-of-two lengths)
// Directional variant: forward=true uses e^{-i 2*pi*k*n/N}; inverse scales by 1/N.
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

    // Danielson–Lanczos passes
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
                w *= wlen; // recurrence, no trig in inner loop
            }
        }
    }

    if (!forward) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto &x : a) x *= inv_n;
    }
}

inline void fft_inplace(std::vector<cd>& a) {
#if defined(TACO_POCKETFFT_HEADER)
    const std::size_t n = a.size();
    if (n == 0) return;
    std::vector<cd> out(n);
    pocketfft::shape_t shape{ static_cast<std::ptrdiff_t>(n) };
    pocketfft::stride_t stride{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::axes_t axes{ 0 };
    pocketfft::c2c(shape, stride, stride, axes,
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
    pocketfft::stride_t stride{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::axes_t axes{ 0 };
    pocketfft::c2c(shape, stride, stride, axes,
                   /*forward=*/false,
                   reinterpret_cast<const cd*>(a.data()),
                   out.data(),
                   /*fct=*/1.0/static_cast<double>(n));
    a.swap(out);
#else
    fft_inplace_builtin_dir(a, /*forward=*/false);
#endif
}

struct FFTPlan {
    std::size_t n{0};
#if defined(TACO_POCKETFFT_HEADER)
    pocketfft::shape_t shape{0};
    pocketfft::stride_t stride{ static_cast<std::ptrdiff_t>(sizeof(cd)) };
    pocketfft::axes_t axes{ 0 };
    std::vector<cd> scratch;
    explicit FFTPlan(std::size_t len)
        : n(len), shape{ static_cast<std::ptrdiff_t>(len) } { scratch.reserve(n); }
    inline void exec_forward(std::vector<cd>& a) {
        if (a.size() != n) throw std::invalid_argument("FFTPlan.exec_forward: size mismatch");
        scratch.resize(n);
        pocketfft::c2c(shape, stride, stride, axes,
                       /*forward=*/true,
                       reinterpret_cast<const cd*>(a.data()),
                       scratch.data(),
                       /*fct=*/1.0);
        a.swap(scratch);
    }
    inline void exec_inverse(std::vector<cd>& a) {
        if (a.size() != n) throw std::invalid_argument("FFTPlan.exec_inverse: size mismatch");
        scratch.resize(n);
        pocketfft::c2c(shape, stride, stride, axes,
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
    inline void exec_forward(std::vector<cd>& a) { fft_inplace_builtin_dir(a, /*forward=*/true); }
    inline void exec_inverse(std::vector<cd>& a) { fft_inplace_builtin_dir(a, /*forward=*/false); }
#endif
};

// Build C(t) from J(w) using discrete FFT at temperature beta.
// Returns t[0..N] and C[0..N] (non-negative times), with internal zero-padding to power-of-two.
template<class JCallable>
inline void bcf_fft_fun(std::size_t N, double dt, JCallable J, double beta,
                        std::vector<double>& t, std::vector<cd>& C)
{
    if (!(dt > 0.0)) throw std::invalid_argument("dt must be > 0");

    const std::size_t Nt   = N + 1;          // number of returned time samples
    std::size_t Nfft       = 2 * Nt;         // symmetric spectrum length
    if ((Nfft & (Nfft - 1)) != 0) Nfft = next_pow2(Nfft);  // round up for FFT

    const double tf     = dt * static_cast<double>(Nfft);  // total period
    const double domega = 2.0 * PI / tf;                   // frequency spacing
    const double omegaNy= PI / dt;                         // Nyquist (fixed by dt)
    const std::size_t posCount = Nfft/2 - 1;               // # of strictly +w bins

    // Build S(w_k) on [DC, +w, Nyquist, -w] with half-weights at DC/Nyquist.
    std::vector<cd> S(Nfft, cd(0.0, 0.0));

    if (std::isinf(beta)) {
        // T = 0: S(+w) = (pi/2) J(w); negative side = 0
        for (std::size_t k = 1; k <= posCount; ++k) {
            const double w = domega * static_cast<double>(k);
            S[k] = cd( (PI/2.0) * J(w), 0.0 );
        }
        S[0]         = cd(0.0, 0.0);
        S[Nfft/2]    = cd( 0.25 * PI * J(omegaNy), 0.0 );
    } else {
        // finite T: detailed balance S(-w) = exp(-beta*w) * S(+w)
        const double r = std::exp(-beta * domega);
        double KMS = r; // k = 1
        for (std::size_t k = 1; k <= posCount; ++k) {
            const double w = domega * static_cast<double>(k);
            const double Jval = J(w);
            const double Sp   = (PI/2.0) * Jval / (1.0 - KMS);
            S[k]        = cd(Sp, 0.0);
            S[Nfft - k] = cd(Sp * KMS, 0.0);
            KMS *= r;
        }
        // DC via discrete w->0 limit using w = domega
        const double ed = r;
        S[0] = cd( 0.5 * (PI/2.0) * J(domega) * (1.0 + ed) / (1.0 - ed), 0.0 );

        // Nyquist: coth(beta*w/2) = 1 + 2/(exp(beta*w) - 1) for stability
        const double eNy = std::exp(beta * omegaNy);
        S[Nfft/2] = cd( 0.25 * PI * J(omegaNy) * (1.0 + 2.0 / (eNy - 1.0)), 0.0 );
    }

    // Forward FFT -> time domain using a tiny plan (matches MATLAB: C = fft(S) * (dω/π))
    {
        FFTPlan plan(Nfft);
        plan.exec_forward(S);
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
