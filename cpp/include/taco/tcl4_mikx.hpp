#pragma once

#include <vector>
#include <Eigen/Dense>

#include "taco/tcl4.hpp"

namespace taco::tcl4 {

struct MikxTensors {
    Eigen::MatrixXcd M; // size N^2 x N^2 (flattened)
    Eigen::MatrixXcd I;
    Eigen::MatrixXcd K;
    std::vector<std::complex<double>> X; // stored as length N^6 (row-major)
    int N{0};
};

// Serial reference implementation.
MikxTensors build_mikx_serial(const Tcl4Map& map,
                              const TripleKernelSeries& kernels,
                              std::size_t time_index);


//omp reference implementation 
#ifdef _OPENMP
MikxTensors build_mikx_omp(const Tcl4Map& map,
                              const TripleKernelSeries& kernels,
                              std::size_t time_index)
#endif
// Backward-compatible alias.
inline MikxTensors build_mikx(const Tcl4Map& map,
                              const TripleKernelSeries& kernels,
                              std::size_t time_index) {
    #ifdef _OPENMP
         return    build_mikx_omp(map,kernels,time_index)
    #else
        return build_mikx_serial(map, kernels, time_index);
    #endif
}

} // namespace taco::tcl4
