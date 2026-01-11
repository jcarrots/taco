#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "taco/system.hpp"
#include "taco/tcl4_kernels.hpp"

#ifdef TACO_HAS_MPI
#include <mpi.h>
#endif

namespace taco::tcl4 {

#ifdef TACO_HAS_MPI
// Hybrid MPI + OpenMP CPU builder for TCL4 L4(t) at many time indices.
//
// - MPI: partitions `time_indices` across ranks (coarse-grain parallelism).
// - OpenMP: parallelizes within each rank over its assigned time indices.
//
// If `time_indices` is empty, computes all time indices [0..Nt-1].
//
// Return value:
// - Rank 0 returns the full output vector in the same order as `time_indices` (or [0..Nt-1]).
// - Non-root ranks return an empty vector.
std::vector<Eigen::MatrixXcd>
build_TCL4_generator_cpu_mpi_omp_batch(const sys::System& system,
                                       const Eigen::MatrixXcd& gamma_series,
                                       double dt,
                                       const std::vector<std::size_t>& time_indices,
                                       FCRMethod method,
                                       MPI_Comm comm);
#endif

} // namespace taco::tcl4

