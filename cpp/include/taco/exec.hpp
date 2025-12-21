#pragma once

namespace taco {

// Backend execution model:
// - Serial: single-thread CPU
// - Omp: shared-memory CPU parallelism (OpenMP/TBB)
// - Cuda: single-node GPU
// - MpiOmp: distributed CPU (MPI + OpenMP)
// - MpiCuda: distributed GPU (MPI + CUDA)
enum class Backend { Serial, Omp, Cuda, MpiOmp, MpiCuda };

struct Exec {
    Backend backend{Backend::Omp};
    int threads{0};      // 0 => use hardware_concurrency or default
    int gpu_id{0};       // active GPU device id
    int streams{2};      // GPU streams for overlap
    bool pinned{true};   // use pinned host buffers for transfers
};

} // namespace taco

