#pragma once

namespace taco {

// Backend execution model:
// - Serial: single-thread CPU
// - Omp: shared-memory CPU parallelism (OpenMP/TBB)
// - Cuda: single-node GPU
// - MpiOmp: distributed CPU (MPI + OpenMP)
// - MpiCuda: distributed GPU (MPI + CUDA)
enum class Backend { Serial, Omp, Cuda, MpiOmp, MpiCuda };

// CUDA numeric precision selection (used by CUDA backends that support it).
// Note: many algorithms are still defined in terms of double-precision host inputs; FP32
// paths typically cast inputs on upload and cast outputs on download.
enum class CudaPrecision { Fp64, Fp32 };

struct Exec {
    Backend backend{Backend::Omp};
    int threads{0};      // 0 => use hardware_concurrency or default
    int gpu_id{0};       // active GPU device id
    int streams{2};      // GPU streams for overlap
    bool pinned{true};   // use pinned host buffers for transfers
    CudaPrecision cuda_precision{CudaPrecision::Fp64};
};

} // namespace taco

