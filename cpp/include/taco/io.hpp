#pragma once

#include <Eigen/Dense>

#include <complex>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace taco::io {

inline void write_csv_matrix(std::ostream& os,
                             const Eigen::MatrixXcd& M,
                             int precision = 17)
{
    os.setf(std::ios::fixed);
    os << std::setprecision(precision);
    os << "row,col,re,im\n";
    for (Eigen::Index r = 0; r < M.rows(); ++r) {
        for (Eigen::Index c = 0; c < M.cols(); ++c) {
            const std::complex<double> v = M(r, c);
            os << r << "," << c << "," << v.real() << "," << v.imag() << "\n";
        }
    }
}

inline void write_csv_matrix(const std::string& path,
                             const Eigen::MatrixXcd& M,
                             int precision = 17)
{
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    write_csv_matrix(ofs, M, precision);
}

} // namespace taco::io
