#include "taco/tcl4_assemble.hpp"

#include <cstddef>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace taco::tcl4 {

namespace {
// Column-major pair flatten: idx(row,col) = row + col*N
inline std::size_t flat2(std::size_t N, int row, int col) {
    return static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * N;
}

inline std::size_t flat6(std::size_t N, int j, int k, int p, int q, int r, int s) {
    const std::size_t NN = N;
    return static_cast<std::size_t>(j) +
           NN * (static_cast<std::size_t>(k) +
           NN * (static_cast<std::size_t>(p) +
           NN * (static_cast<std::size_t>(q) +
           NN * (static_cast<std::size_t>(r) +
           NN * static_cast<std::size_t>(s)))));
}
} // namespace

Eigen::MatrixXcd assemble_liouvillian(const MikxTensors& tensors,
                                      const std::vector<Eigen::MatrixXcd>& coupling_ops)
{
    if (tensors.N <= 0) {
        throw std::invalid_argument("assemble_liouvillian: tensors.N must be > 0");
    }
    const std::size_t N = static_cast<std::size_t>(tensors.N);
    const std::size_t N2 = N * N;

    // Validate tensor shapes
    const std::size_t X_expected = N * N * N * N * N * N;
    if (tensors.M.rows() != static_cast<Eigen::Index>(N2) || tensors.M.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.I.rows() != static_cast<Eigen::Index>(N2) || tensors.I.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.K.rows() != static_cast<Eigen::Index>(N2) || tensors.K.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.X.size() != X_expected) {
        throw std::invalid_argument("assemble_liouvillian: tensor dimensions do not match N");
    }

    if (coupling_ops.empty()) {
        throw std::invalid_argument("assemble_liouvillian: coupling_ops must be non-empty");
    }
    for (const auto& C : coupling_ops) {
        if (C.rows() != static_cast<Eigen::Index>(N) || C.cols() != static_cast<Eigen::Index>(N)) {
            throw std::invalid_argument("assemble_liouvillian: coupling operator has wrong shape");
        }
    }

    // GW holds the raw (possibly non-Hermitian) matrix first, then is symmetrized in-place.
    // This avoids a second N^2x N^2 allocation for the intermediate T.
    Eigen::MatrixXcd GW =
        Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));

    // Alias for readability
    const auto& M = tensors.M;
    const auto& Ia = tensors.I; // MATLAB uses variable name Ia
    const auto& K = tensors.K;
    const auto& X = tensors.X;

    // Outer indices follow MATLAB NAKZWAN_v9.m: T(n,i,m,j)
    // Parallelize over all Liouville indices (row=(n,i), col=(m,j)) => N^4 tasks.
    const std::size_t total = N2 * N2;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    #endif
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(total); ++idx) {
        // Column-major linearization: idx = row + col*N2
        const std::size_t row_u = static_cast<std::size_t>(idx) % N2; // (n,i) flattened
        const std::size_t col_u = static_cast<std::size_t>(idx) / N2; // (m,j) flattened

        const int n = static_cast<int>(row_u % N);
        const int i = static_cast<int>(row_u / N);
        const int m = static_cast<int>(col_u % N);
        const int j = static_cast<int>(col_u / N);

        std::complex<double> res(0.0, 0.0);

        // Sum over coupling channels A,B ∈ coupling_ops
        for (std::size_t iA = 0; iA < coupling_ops.size(); ++iA) {
            const auto& A = coupling_ops[iA];
            for (std::size_t iB = 0; iB < coupling_ops.size(); ++iB) {
                const auto& B = coupling_ops[iB];

                // Sum over internal indices a,b
                for (int a = 0; a < static_cast<int>(N); ++a) {
                    for (int b = 0; b < static_cast<int>(N); ++b) {
                        // Fetch M/I/K values with (row=(j,k), col=(p,q)) mapping
                        const auto idx_M_row_bi = static_cast<Eigen::Index>(flat2(N, b, i));
                        const auto idx_M_col_na = static_cast<Eigen::Index>(flat2(N, n, a));
                        const auto idx_I_row_an = static_cast<Eigen::Index>(flat2(N, a, n));
                        const auto idx_I_col_ib = static_cast<Eigen::Index>(flat2(N, i, b));
                        const auto idx_K_row_ib = static_cast<Eigen::Index>(flat2(N, i, b));
                        const auto idx_K_col_na = static_cast<Eigen::Index>(flat2(N, n, a));

                        const std::complex<double> M_bi_na = M(idx_M_row_bi, idx_M_col_na);
                        const std::complex<double> I_an_ib = Ia(idx_I_row_an, idx_I_col_ib);
                        const std::complex<double> K_ib_na = K(idx_K_row_ib, idx_K_col_na);

                        // X indices use order (j,k,p,q,r,s)
                        const std::complex<double> X_anjbni =
                            X[flat6(N, /*j=*/a, /*k=*/n, /*p=*/j, /*q=*/b, /*r=*/n, /*s=*/i)];
                        const std::complex<double> X_iajbni =
                            X[flat6(N, /*j=*/i, /*k=*/a, /*p=*/j, /*q=*/b, /*r=*/n, /*s=*/i)];
                        const std::complex<double> X_bajmai =
                            X[flat6(N, /*j=*/b, /*k=*/a, /*p=*/j, /*q=*/m, /*r=*/a, /*s=*/i)];
                        const std::complex<double> X_ibjmai =
                            X[flat6(N, /*j=*/i, /*k=*/b, /*p=*/j, /*q=*/m, /*r=*/a, /*s=*/i)];

                        // Terms inside the big parentheses of res -= (...)
                        const std::complex<double> t1 = B(n, a) * A(a, b) * B(b, i) * A(j, m) * M_bi_na;
                        const std::complex<double> t2 = A(n, a) * B(a, b) * B(b, i) * A(j, m) * I_an_ib;
                        const std::complex<double> t3 = B(n, a) * B(a, b) * A(b, i) * A(j, m) * K_ib_na;
                        const std::complex<double> t4 = A(n, a) * B(a, i) * B(j, b) * A(b, m) * X_anjbni;
                        const std::complex<double> t5 = B(n, a) * A(a, i) * B(j, b) * A(b, m) * X_iajbni;
                        const std::complex<double> t6 = A(n, a) * A(a, b) * B(b, i) * B(j, m) * X_bajmai;
                        const std::complex<double> t7 = A(n, a) * B(a, b) * A(b, i) * B(j, m) * X_ibjmai;

                        // res -= (t1 - t2 + t3 + t4 - t5 - t6 + t7)
                        res -= (t1 - t2 + t3 + t4 - t5 - t6 + t7);
                    }
                }

                // If (j == m) additional sum over c
                if (j == m) {
                    for (int a = 0; a < static_cast<int>(N); ++a) {
                        for (int b = 0; b < static_cast<int>(N); ++b) {
                            for (int c = 0; c < static_cast<int>(N); ++c) {
                                const auto idx_M_row_ci = static_cast<Eigen::Index>(flat2(N, c, i));
                                const auto idx_M_col_ab = static_cast<Eigen::Index>(flat2(N, a, b));
                                const auto idx_K_row_ic = static_cast<Eigen::Index>(flat2(N, i, c));
                                const auto idx_K_col_ab = static_cast<Eigen::Index>(flat2(N, a, b));
                                const auto idx_I_row_ba = static_cast<Eigen::Index>(flat2(N, b, a));
                                const auto idx_I_col_ic = static_cast<Eigen::Index>(flat2(N, i, c));

                                const std::complex<double> M_ci_ab = M(idx_M_row_ci, idx_M_col_ab);
                                const std::complex<double> K_ic_ab = K(idx_K_row_ic, idx_K_col_ab);
                                const std::complex<double> I_ba_ic = Ia(idx_I_row_ba, idx_I_col_ic);

                                const std::complex<double> add1 =
                                    A(n, a) * B(a, b) * A(b, c) * B(c, i) * M_ci_ab;
                                const std::complex<double> add2 =
                                    A(n, a) * B(a, b) * B(b, c) * A(c, i) * K_ic_ab;
                                const std::complex<double> add3 =
                                    -A(n, a) * A(a, b) * B(b, c) * B(c, i) * I_ba_ic;
                                res += (add1 + add2 + add3);
                            }
                        }
                    }
                }
            }
        }

        GW(static_cast<Eigen::Index>(row_u), static_cast<Eigen::Index>(col_u)) = res;
    }

    // Symmetrize: GW = T + T^H (MATLAB ' operator), done in-place.
    // We stored T into GW above, so: GW <- GW + GW^H.
    for (Eigen::Index col = 0; col < static_cast<Eigen::Index>(N2); ++col) {
        for (Eigen::Index row = 0; row <= col; ++row) {
            const auto t_rc = GW(row, col);
            const auto t_cr = GW(col, row);
            const auto g_rc = t_rc + std::conj(t_cr);
            GW(row, col) = g_rc;
            if (row != col) {
                GW(col, row) = std::conj(g_rc);
            }
        }
    }
    return GW;
}

Eigen::MatrixXcd gw_to_liouvillian(const Eigen::MatrixXcd& GW, std::size_t N)
{
    if (N == 0) throw std::invalid_argument("gw_to_liouvillian: N must be > 0");
    const std::size_t N2 = N * N;
    if (GW.rows() != GW.cols()) {
        throw std::invalid_argument("gw_to_liouvillian: GW must be square");
    }
    if (static_cast<std::size_t>(GW.rows()) != N2) {
        throw std::invalid_argument("gw_to_liouvillian: GW dims must be (N^2 x N^2)");
    }

    Eigen::MatrixXcd L(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
    for (int n = 0; n < static_cast<int>(N); ++n) {
        for (int m = 0; m < static_cast<int>(N); ++m) {
            for (int i = 0; i < static_cast<int>(N); ++i) {
                for (int j = 0; j < static_cast<int>(N); ++j) {
                    const auto row_L = static_cast<Eigen::Index>(flat2(N, n, m));
                    const auto col_L = static_cast<Eigen::Index>(flat2(N, i, j));
                    const auto row_G = static_cast<Eigen::Index>(flat2(N, n, i));
                    const auto col_G = static_cast<Eigen::Index>(flat2(N, m, j));
                    L(row_L, col_L) = GW(row_G, col_G);
                }
            }
        }
    }
    return L;
}

} // namespace taco::tcl4
