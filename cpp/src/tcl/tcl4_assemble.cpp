#include "taco/tcl4_assemble.hpp"

#include <stdexcept>

namespace taco::tcl4 {

namespace {
// Column-major pair flatten: idx(row,col) = row + col*N
inline std::size_t flat2(std::size_t N, int row, int col) {
    return static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * N;
}

inline std::size_t flat6(std::size_t N, int j,int k,int p,int q,int r,int s) {
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
    if (tensors.M.rows() != static_cast<Eigen::Index>(N2) || tensors.M.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.I.rows() != static_cast<Eigen::Index>(N2) || tensors.I.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.K.rows() != static_cast<Eigen::Index>(N2) || tensors.K.cols() != static_cast<Eigen::Index>(N2) ||
        tensors.X.size()  != static_cast<Eigen::Index>(N*N*N*N*N*N)) {
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

    // T is the raw (possibly non-Hermitian) matrix prior to symmetrization
    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));

    // Alias for readability
    const auto& M = tensors.M;
    const auto& Ia = tensors.I; // MATLAB uses variable name Ia
    const auto& K = tensors.K;
    const auto& X = tensors.X;

    // Outer indices follow MATLAB NAKZWAN_v9.m: T(n,i,m,j)
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int n = 0; n < static_cast<int>(N); ++n) {
        for (int i = 0; i < static_cast<int>(N); ++i) {
            for (int m = 0; m < static_cast<int>(N); ++m) {
                for (int j = 0; j < static_cast<int>(N); ++j) {
                    std::complex<double> res(0.0, 0.0);

                    // Sum over coupling channels A,B ∈ coupling_ops
                    for (std::size_t iA = 0; iA < coupling_ops.size(); ++iA) {
                        const auto& A = coupling_ops[iA];
                        for (std::size_t iB = 0; iB < coupling_ops.size(); ++iB) {
                            const auto& B = coupling_ops[iB];

                            // Sum over internal indices a,b
                            for (int a = 0; a < static_cast<int>(N); ++a) {
                                for (int b = 0; b < static_cast<int>(N); ++b) {
                                    // Fetch M/I/K/X values with (row=(j,k), col=(p,q)) mapping
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
                                    const std::complex<double> X_anjbni = X[flat6(N, /*j=*/a, /*k=*/n,
                                                                                     /*p=*/j, /*q=*/b,
                                                                                     /*r=*/n, /*s=*/i)];
                                    const std::complex<double> X_iajbni = X[flat6(N, /*j=*/i, /*k=*/a,
                                                                                     /*p=*/j, /*q=*/b,
                                                                                     /*r=*/n, /*s=*/i)];
                                    const std::complex<double> X_bajmai = X[flat6(N, /*j=*/b, /*k=*/a,
                                                                                     /*p=*/j, /*q=*/m,
                                                                                     /*r=*/a, /*s=*/i)];
                                    const std::complex<double> X_ibjmai = X[flat6(N, /*j=*/i, /*k=*/b,
                                                                                     /*p=*/j, /*q=*/m,
                                                                                     /*r=*/a, /*s=*/i)];

                                    // Terms inside the big parentheses of res -= (...)
                                    const std::complex<double> t1 = B(n,a) * A(a,b) * B(b,i) * A(j,m) * M_bi_na;
                                    const std::complex<double> t2 = A(n,a) * B(a,b) * B(b,i) * A(j,m) * I_an_ib;
                                    const std::complex<double> t3 = B(n,a) * B(a,b) * A(b,i) * A(j,m) * K_ib_na;
                                    const std::complex<double> t4 = A(n,a) * B(a,i) * B(j,b) * A(b,m) * X_anjbni;
                                    const std::complex<double> t5 = B(n,a) * A(a,i) * B(j,b) * A(b,m) * X_iajbni;
                                    const std::complex<double> t6 = A(n,a) * A(a,b) * B(b,i) * B(j,m) * X_bajmai;
                                    const std::complex<double> t7 = A(n,a) * B(a,b) * A(b,i) * B(j,m) * X_ibjmai;

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

                                            const std::complex<double> add1 = A(n,a) * B(a,b) * A(b,c) * B(c,i) * M_ci_ab;
                                            const std::complex<double> add2 = A(n,a) * B(a,b) * B(b,c) * A(c,i) * K_ic_ab;
                                            const std::complex<double> add3 = - A(n,a) * A(a,b) * B(b,c) * B(c,i) * I_ba_ic;
                                            res += (add1 + add2 + add3);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Map (n,i,m,j) -> flat row/col and store
                    const auto row = static_cast<Eigen::Index>(flat2(N, n, i));
                    const auto col = static_cast<Eigen::Index>(flat2(N, m, j));
                    T(row, col) = res;
                }
            }
        }
    }

    // Symmetrize: GW = T + T^† (MATLAB ' operator)
    Eigen::MatrixXcd GW = T + T.adjoint();
    return GW;
}

} // namespace taco::tcl4
