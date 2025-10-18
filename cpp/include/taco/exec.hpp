#pragma once

namespace taco {

enum class Backend { CPU, GPU, Hybrid };

struct Exec {
    Backend backend{Backend::CPU};
    int threads{0};      // 0 => use hardware_concurrency or default
    int gpu_id{0};       // active GPU device id
    int streams{2};      // GPU streams for overlap
    bool pinned{true};   // use pinned host buffers for transfers
};

} // namespace taco

