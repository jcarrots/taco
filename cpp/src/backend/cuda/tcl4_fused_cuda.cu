#include "taco/backend/cuda/tcl4_fused_cuda.hpp"

#include "taco/backend/cuda/tcl4_assemble_cuda.hpp"
#include "taco/backend/cuda/tcl4_fcr_kernels_cuda.hpp"
#include "taco/backend/cuda/tcl4_mikx_cuda.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_kernels.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace taco::tcl4 {

namespace {

inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

inline int read_env_int(const char* name, int fallback) {
#ifdef _MSC_VER
    char* buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || !buf) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(buf, &end, 10);
    const bool ok = (end != buf);
    std::free(buf);
    if (!ok) return fallback;
#else
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) return fallback;
#endif
    if (parsed > std::numeric_limits<int>::max() || parsed < std::numeric_limits<int>::min()) return fallback;
    return static_cast<int>(parsed);
}

inline bool tcl4_cuda_graph_enabled() {
    // CUDA Graphs can reduce host launch overhead for the fixed MIKX -> GW -> L4 pipeline.
    // Disable with: TCL4_USE_CUDA_GRAPH=0
    // Diagnostics:  TCL4_CUDA_GRAPH_VERBOSE=1
    //
    // Fallback behavior: if capture/instantiate/update/launch fails, this file falls back to the
    // original non-graph kernel launch path.
    static const bool enabled = (read_env_int("TCL4_USE_CUDA_GRAPH", 1) != 0);
    return enabled;
}

inline bool tcl4_cuda_graph_verbose() {
    static const bool verbose = (read_env_int("TCL4_CUDA_GRAPH_VERBOSE", 0) != 0);
    return verbose;
}

// Forward decl (used by the CUDA graph capture path before the kernel definition below).
__global__ void kernel_gw_raw_sym_to_liouvillian(const cuDoubleComplex* __restrict__ GW_raw,
                                                 cuDoubleComplex* __restrict__ L4,
                                                 int N);

struct CudaEvent {
    cudaEvent_t ev{nullptr};

    CudaEvent() = default;
    explicit CudaEvent(unsigned flags) {
        cuda_check(cudaEventCreateWithFlags(&ev, flags), "cudaEventCreateWithFlags");
    }
    ~CudaEvent() {
        if (ev) cudaEventDestroy(ev);
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : ev(other.ev) { other.ev = nullptr; }
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this == &other) return *this;
        if (ev) cudaEventDestroy(ev);
        ev = other.ev;
        other.ev = nullptr;
        return *this;
    }
};

inline std::size_t next_pow2(std::size_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if constexpr (sizeof(std::size_t) >= 8) n |= n >> 32;
    return ++n;
}

struct DeviceBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
};

inline void release_buffer(DeviceBuffer& buf) {
    if (buf.ptr) cudaFree(buf.ptr);
    buf.ptr = nullptr;
    buf.bytes = 0;
}

template <typename T>
T* ensure_buffer(DeviceBuffer& buf, std::size_t count, const char* what) {
    if (count == 0) return nullptr;
    const std::size_t bytes = count * sizeof(T);
    if (bytes > buf.bytes) {
        release_buffer(buf);
        cuda_check(cudaMalloc(&buf.ptr, bytes), what);
        buf.bytes = bytes;
    }
    return static_cast<T*>(buf.ptr);
}

struct FusedCudaWorkspace {
    int gpu_id{-1};
    cudaStream_t stream{nullptr};
    cuda_fcr::FcrWorkspace fcr;

    DeviceBuffer gamma;
    DeviceBuffer omegas;
    DeviceBuffer mirror;
    DeviceBuffer pair_to_freq;
    DeviceBuffer ops;

    DeviceBuffer F;
    DeviceBuffer C;
    DeviceBuffer R;
    DeviceBuffer F_all;
    DeviceBuffer C_all;
    DeviceBuffer R_all;
    DeviceBuffer Ftmp;
    DeviceBuffer Ctmp;
    DeviceBuffer Rtmp;

    DeviceBuffer M;
    DeviceBuffer I;
    DeviceBuffer K;
    DeviceBuffer X;
    DeviceBuffer GW;
    DeviceBuffer L4;
    DeviceBuffer L4_all;

    DeviceBuffer time_indices;

    struct GraphKey {
        int gpu_id{0};
        int N{0};
        int nf{0};
        int num_ops{0};
        std::size_t num_times{0};

        friend bool operator==(const GraphKey& a, const GraphKey& b) {
            return a.gpu_id == b.gpu_id && a.N == b.N && a.nf == b.nf && a.num_ops == b.num_ops &&
                   a.num_times == b.num_times;
        }
    };

    struct GraphKeyHash {
        std::size_t operator()(const GraphKey& k) const noexcept {
            std::size_t h = 1469598103934665603ull;
            auto mix = [&](std::size_t v) {
                h ^= v;
                h *= 1099511628211ull;
            };
            mix(static_cast<std::size_t>(k.gpu_id));
            mix(static_cast<std::size_t>(k.N));
            mix(static_cast<std::size_t>(k.nf));
            mix(static_cast<std::size_t>(k.num_ops));
            mix(static_cast<std::size_t>(k.num_times));
            return h;
        }
    };

    struct PipelineArgs {
        const cuDoubleComplex* F_all{nullptr};
        const cuDoubleComplex* C_all{nullptr};
        const cuDoubleComplex* R_all{nullptr};
        const int* pair_to_freq{nullptr};
        const cuDoubleComplex* ops{nullptr};

        cuDoubleComplex* M{nullptr};
        cuDoubleComplex* I{nullptr};
        cuDoubleComplex* K{nullptr};
        cuDoubleComplex* X{nullptr};
        cuDoubleComplex* GW_raw{nullptr};
        cuDoubleComplex* L4_all{nullptr};

        int N{0};
        int nf{0};
        int num_ops{0};
        std::size_t num_times{0};
        std::size_t nf3{0};
        std::size_t l4_stride{0}; // elements per L4 output (N2*N2)
    };

    struct KernelParamPackMikx {
        const cuDoubleComplex* F{nullptr};
        const cuDoubleComplex* C{nullptr};
        const cuDoubleComplex* R{nullptr};
        const int* pair_to_freq{nullptr};
        int N{0};
        int nf{0};
        cuDoubleComplex* M{nullptr};
        cuDoubleComplex* I{nullptr};
        cuDoubleComplex* K{nullptr};
        cuDoubleComplex* X{nullptr};

        void* kernel_params[10]{};
        cudaKernelNodeParams params{};

        KernelParamPackMikx() {
            kernel_params[0] = &F;
            kernel_params[1] = &C;
            kernel_params[2] = &R;
            kernel_params[3] = &pair_to_freq;
            kernel_params[4] = &N;
            kernel_params[5] = &nf;
            kernel_params[6] = &M;
            kernel_params[7] = &I;
            kernel_params[8] = &K;
            kernel_params[9] = &X;
            params.kernelParams = kernel_params;
            params.extra = nullptr;
        }
    };

    struct KernelParamPackGw {
        const cuDoubleComplex* M{nullptr};
        const cuDoubleComplex* I{nullptr};
        const cuDoubleComplex* K{nullptr};
        const cuDoubleComplex* X{nullptr};
        const cuDoubleComplex* ops{nullptr};
        int N{0};
        int num_ops{0};
        cuDoubleComplex* GW_raw{nullptr};

        void* kernel_params[8]{};
        cudaKernelNodeParams params{};

        KernelParamPackGw() {
            kernel_params[0] = &M;
            kernel_params[1] = &I;
            kernel_params[2] = &K;
            kernel_params[3] = &X;
            kernel_params[4] = &ops;
            kernel_params[5] = &N;
            kernel_params[6] = &num_ops;
            kernel_params[7] = &GW_raw;
            params.kernelParams = kernel_params;
            params.extra = nullptr;
        }
    };

    struct KernelParamPackL4 {
        const cuDoubleComplex* GW_raw{nullptr};
        cuDoubleComplex* L4{nullptr};
        int N{0};

        void* kernel_params[3]{};
        cudaKernelNodeParams params{};

        KernelParamPackL4() {
            kernel_params[0] = &GW_raw;
            kernel_params[1] = &L4;
            kernel_params[2] = &N;
            params.kernelParams = kernel_params;
            params.extra = nullptr;
        }
    };

    struct Tcl4CudaGraph {
        GraphKey key{};
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
        std::vector<cudaGraphNode_t> mikx_nodes;
        std::vector<cudaGraphNode_t> gw_nodes;
        std::vector<cudaGraphNode_t> l4_nodes;
        KernelParamPackMikx mikx_pack;
        KernelParamPackGw gw_pack;
        KernelParamPackL4 l4_pack;
        PipelineArgs last_args{};
        bool logged_launch{false};

        ~Tcl4CudaGraph() { reset(); }

        void reset() {
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
            exec = nullptr;
            graph = nullptr;
            mikx_nodes.clear();
            gw_nodes.clear();
            l4_nodes.clear();
            last_args = {};
            logged_launch = false;
        }

        static void launch_pipeline(cudaStream_t stream, const PipelineArgs& a) {
            cuda_mikx::MikxDeviceInputs in;
            in.pair_to_freq = a.pair_to_freq;
            in.N = a.N;
            in.nf = a.nf;

            cuda_mikx::MikxDeviceOutputs out;
            out.M = a.M;
            out.I = a.I;
            out.K = a.K;
            out.X = a.X;

            constexpr unsigned block = 16;
            const dim3 block_l4(block, block);
            const std::size_t N_u = static_cast<std::size_t>(a.N);
            const std::size_t N2 = N_u * N_u;
            const dim3 grid_l4(static_cast<unsigned>((N2 + block_l4.x - 1) / block_l4.x),
                               static_cast<unsigned>((N2 + block_l4.y - 1) / block_l4.y));

            for (std::size_t out_idx = 0; out_idx < a.num_times; ++out_idx) {
                in.F = a.F_all + out_idx * a.nf3;
                in.C = a.C_all + out_idx * a.nf3;
                in.R = a.R_all + out_idx * a.nf3;

                cuda_mikx::build_mikx_device(in, out, stream);
                assemble_liouvillian_cuda_device_raw(a.M,
                                                     a.I,
                                                     a.K,
                                                     a.X,
                                                     a.ops,
                                                     a.N,
                                                     a.num_ops,
                                                     a.GW_raw,
                                                     stream);

                cuDoubleComplex* dL4_out = a.L4_all + out_idx * a.l4_stride;
                kernel_gw_raw_sym_to_liouvillian<<<grid_l4, block_l4, 0, stream>>>(a.GW_raw, dL4_out, a.N);
                cuda_check(cudaGetLastError(), "kernel_gw_raw_sym_to_liouvillian launch");
            }
        }

        static bool topo_order(cudaGraph_t g, std::vector<cudaGraphNode_t>& out) {
            out.clear();
            std::size_t num_nodes = 0;
            cudaError_t st = cudaGraphGetNodes(g, nullptr, &num_nodes);
            if (st != cudaSuccess) return false;
            std::vector<cudaGraphNode_t> nodes(num_nodes);
            st = cudaGraphGetNodes(g, nodes.data(), &num_nodes);
            if (st != cudaSuccess) return false;

            std::size_t num_edges = 0;
            st = cudaGraphGetEdges(g, nullptr, nullptr, &num_edges);
            if (st != cudaSuccess) return false;
            std::vector<cudaGraphNode_t> from(num_edges);
            std::vector<cudaGraphNode_t> to(num_edges);
            st = cudaGraphGetEdges(g, from.data(), to.data(), &num_edges);
            if (st != cudaSuccess) return false;

            std::unordered_map<cudaGraphNode_t, std::size_t> idx;
            idx.reserve(nodes.size());
            for (std::size_t i = 0; i < nodes.size(); ++i) idx[nodes[i]] = i;

            std::vector<std::vector<std::size_t>> adj(nodes.size());
            std::vector<int> indeg(nodes.size(), 0);
            for (std::size_t e = 0; e < num_edges; ++e) {
                const auto it_f = idx.find(from[e]);
                const auto it_t = idx.find(to[e]);
                if (it_f == idx.end() || it_t == idx.end()) continue;
                adj[it_f->second].push_back(it_t->second);
                indeg[it_t->second] += 1;
            }

            std::vector<std::size_t> q;
            q.reserve(nodes.size());
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                if (indeg[i] == 0) q.push_back(i);
            }

            out.reserve(nodes.size());
            for (std::size_t qi = 0; qi < q.size(); ++qi) {
                const std::size_t v = q[qi];
                out.push_back(nodes[v]);
                for (std::size_t w : adj[v]) {
                    indeg[w] -= 1;
                    if (indeg[w] == 0) q.push_back(w);
                }
            }

            return out.size() == nodes.size();
        }

        bool capture_once(cudaStream_t stream, const PipelineArgs& a) {
            reset();

            cuda_check(cudaSetDevice(key.gpu_id), "cudaSetDevice(graph)");

            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                       "cudaStreamBeginCapture");
            launch_pipeline(stream, a);
            cudaError_t st = cudaStreamEndCapture(stream, &graph);
            if (st != cudaSuccess) {
                if (tcl4_cuda_graph_verbose()) {
                    std::fprintf(stderr, "tcl4_cuda_graph: end capture failed: %s\n", cudaGetErrorString(st));
                }
                reset();
                return false;
            }

            st = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            if (st != cudaSuccess) {
                if (tcl4_cuda_graph_verbose()) {
                    std::fprintf(stderr, "tcl4_cuda_graph: instantiate failed: %s\n", cudaGetErrorString(st));
                }
                reset();
                return false;
            }

            std::vector<cudaGraphNode_t> ordered;
            if (!topo_order(graph, ordered)) {
                if (tcl4_cuda_graph_verbose()) {
                    std::fprintf(stderr, "tcl4_cuda_graph: topo sort failed\n");
                }
                reset();
                return false;
            }

            std::vector<cudaGraphNode_t> kernels;
            kernels.reserve(ordered.size());
            for (cudaGraphNode_t n : ordered) {
                cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
                st = cudaGraphNodeGetType(n, &type);
                if (st != cudaSuccess) {
                    reset();
                    return false;
                }
                if (type == cudaGraphNodeTypeKernel) kernels.push_back(n);
            }

            const std::size_t expected = key.num_times * 3;
            if (kernels.size() != expected) {
                if (tcl4_cuda_graph_verbose()) {
                    std::fprintf(stderr,
                                 "tcl4_cuda_graph: unexpected kernel node count (%zu != %zu)\n",
                                 kernels.size(),
                                 expected);
                }
                reset();
                return false;
            }

            mikx_nodes.resize(key.num_times);
            gw_nodes.resize(key.num_times);
            l4_nodes.resize(key.num_times);
            for (std::size_t t = 0; t < key.num_times; ++t) {
                mikx_nodes[t] = kernels[3 * t + 0];
                gw_nodes[t] = kernels[3 * t + 1];
                l4_nodes[t] = kernels[3 * t + 2];
            }

            auto init_pack = [&](cudaGraphNode_t node,
                                 cudaKernelNodeParams& dst,
                                 unsigned expect_x,
                                 unsigned expect_y,
                                 const char* name) -> bool {
                cudaKernelNodeParams p{};
                st = cudaGraphKernelNodeGetParams(node, &p);
                if (st != cudaSuccess) return false;
                if (p.blockDim.x != expect_x || p.blockDim.y != expect_y) {
                    if (tcl4_cuda_graph_verbose()) {
                        std::fprintf(stderr,
                                     "tcl4_cuda_graph: %s blockDim mismatch (%u,%u) != (%u,%u)\n",
                                     name,
                                     p.blockDim.x,
                                     p.blockDim.y,
                                     expect_x,
                                     expect_y);
                    }
                    return false;
                }
                dst.func = p.func;
                dst.gridDim = p.gridDim;
                dst.blockDim = p.blockDim;
                dst.sharedMemBytes = p.sharedMemBytes;
                return true;
            };

            if (!init_pack(mikx_nodes.front(), mikx_pack.params, 256, 1, "MIKX")) {
                reset();
                return false;
            }
            if (!init_pack(gw_nodes.front(), gw_pack.params, 128, 1, "GW")) {
                reset();
                return false;
            }
            if (!init_pack(l4_nodes.front(), l4_pack.params, 16, 16, "L4")) {
                reset();
                return false;
            }

            if (tcl4_cuda_graph_verbose()) {
                std::fprintf(stderr,
                             "tcl4_cuda_graph: captured key=(gpu=%d N=%d nf=%d ops=%d times=%zu)\n",
                             key.gpu_id,
                             key.N,
                             key.nf,
                             key.num_ops,
                             key.num_times);
            }

            last_args = a;
            return true;
        }

        static bool args_match_for_exec(const PipelineArgs& a, const PipelineArgs& b) {
            return a.F_all == b.F_all && a.C_all == b.C_all && a.R_all == b.R_all &&
                   a.pair_to_freq == b.pair_to_freq && a.ops == b.ops &&
                   a.M == b.M && a.I == b.I && a.K == b.K && a.X == b.X &&
                   a.GW_raw == b.GW_raw && a.L4_all == b.L4_all &&
                   a.N == b.N && a.nf == b.nf && a.num_ops == b.num_ops &&
                   a.num_times == b.num_times && a.nf3 == b.nf3 && a.l4_stride == b.l4_stride;
        }

        bool update_nodes(const PipelineArgs& a) {
            // Update MIKX nodes (vary by out_idx due to F/C/R pointer offset).
            mikx_pack.pair_to_freq = a.pair_to_freq;
            mikx_pack.N = a.N;
            mikx_pack.nf = a.nf;
            mikx_pack.M = a.M;
            mikx_pack.I = a.I;
            mikx_pack.K = a.K;
            mikx_pack.X = a.X;
            for (std::size_t t = 0; t < key.num_times; ++t) {
                mikx_pack.F = a.F_all + t * a.nf3;
                mikx_pack.C = a.C_all + t * a.nf3;
                mikx_pack.R = a.R_all + t * a.nf3;
                const cudaError_t st = cudaGraphExecKernelNodeSetParams(exec, mikx_nodes[t], &mikx_pack.params);
                if (st != cudaSuccess) return false;
            }

            // Update GW nodes (same args each time; intermediate buffers are reused sequentially).
            gw_pack.M = a.M;
            gw_pack.I = a.I;
            gw_pack.K = a.K;
            gw_pack.X = a.X;
            gw_pack.ops = a.ops;
            gw_pack.N = a.N;
            gw_pack.num_ops = a.num_ops;
            gw_pack.GW_raw = a.GW_raw;
            for (std::size_t t = 0; t < key.num_times; ++t) {
                const cudaError_t st = cudaGraphExecKernelNodeSetParams(exec, gw_nodes[t], &gw_pack.params);
                if (st != cudaSuccess) return false;
            }

            // Update L4 nodes (vary by out_idx due to output pointer offset).
            l4_pack.GW_raw = a.GW_raw;
            l4_pack.N = a.N;
            for (std::size_t t = 0; t < key.num_times; ++t) {
                l4_pack.L4 = a.L4_all + t * a.l4_stride;
                const cudaError_t st = cudaGraphExecKernelNodeSetParams(exec, l4_nodes[t], &l4_pack.params);
                if (st != cudaSuccess) return false;
            }

            return true;
        }

        bool launch(cudaStream_t stream, const PipelineArgs& a) {
            if (!exec) {
                // The capture path runs the pipeline as part of capture; we do not replay the graph
                // in the same call to avoid duplicate computation.
                return capture_once(stream, a);
            }

            if (!args_match_for_exec(a, last_args) && !update_nodes(a)) {
                if (tcl4_cuda_graph_verbose()) {
                    std::fprintf(stderr, "tcl4_cuda_graph: update failed; recapturing\n");
                }
                // Recapture on this stream; the capture path executes the work for this call.
                return capture_once(stream, a);
            }
            last_args = a;

            if (tcl4_cuda_graph_verbose() && !logged_launch) {
                std::fprintf(stderr,
                             "tcl4_cuda_graph: launching exec key=(gpu=%d N=%d nf=%d ops=%d times=%zu)\n",
                             key.gpu_id,
                             key.N,
                             key.nf,
                             key.num_ops,
                             key.num_times);
                logged_launch = true;
            }

            const cudaError_t st = cudaGraphLaunch(exec, stream);
            if (st != cudaSuccess) {
                if (tcl4_cuda_graph_verbose()) {
                    std::fprintf(stderr, "tcl4_cuda_graph: launch failed: %s\n", cudaGetErrorString(st));
                }
                reset();
                return false;
            }
            return true;
        }
    };

    std::unordered_map<GraphKey, std::unique_ptr<Tcl4CudaGraph>, GraphKeyHash> graphs;

    ~FusedCudaWorkspace() { reset(); }

    void reset() {
        if (gpu_id >= 0) {
            cudaSetDevice(gpu_id);
        }
        if (fcr.plan) {
            cufftDestroy(fcr.plan);
            fcr.plan = 0;
        }
        if (fcr.A) cudaFree(fcr.A);
        if (fcr.B) cudaFree(fcr.B);
        if (fcr.B_conj) cudaFree(fcr.B_conj);
        if (fcr.scan_tmp) cudaFree(fcr.scan_tmp);
        fcr.A = nullptr;
        fcr.B = nullptr;
        fcr.B_conj = nullptr;
        fcr.scan_tmp = nullptr;
        fcr.scan_tmp_bytes = 0;
        fcr.plan_batch = 0;
        fcr.plan_Nfft = 0;

        release_buffer(gamma);
        release_buffer(omegas);
        release_buffer(mirror);
        release_buffer(pair_to_freq);
        release_buffer(ops);
        release_buffer(F);
        release_buffer(C);
        release_buffer(R);
        release_buffer(F_all);
        release_buffer(C_all);
        release_buffer(R_all);
        release_buffer(Ftmp);
        release_buffer(Ctmp);
        release_buffer(Rtmp);
        release_buffer(M);
        release_buffer(I);
        release_buffer(K);
        release_buffer(X);
        release_buffer(GW);
        release_buffer(L4);
        release_buffer(L4_all);
        release_buffer(time_indices);

        graphs.clear();

        if (stream) cudaStreamDestroy(stream);
        stream = nullptr;
        gpu_id = -1;
    }
};

inline FusedCudaWorkspace& get_fused_workspace(int gpu_id) {
    static thread_local FusedCudaWorkspace ws;
    if (ws.gpu_id != gpu_id) {
        ws.reset();
        cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice");
        cuda_check(cudaStreamCreate(&ws.stream), "cudaStreamCreate");
        ws.gpu_id = gpu_id;
    }
    return ws;
}

inline void copy_time_column(cuDoubleComplex* dst,
                             const cuDoubleComplex* src,
                             std::size_t Nt,
                             std::size_t lane_count,
                             std::size_t time_index,
                             cudaStream_t stream,
                             const char* what)
{
    const std::size_t pitch = Nt * sizeof(cuDoubleComplex);
    cuda_check(cudaMemcpy2DAsync(dst,
                                 sizeof(cuDoubleComplex),
                                 src + time_index,
                                 pitch,
                                 sizeof(cuDoubleComplex),
                                 lane_count,
                                 cudaMemcpyDeviceToDevice,
                                 stream),
               what);
}

constexpr int kTransposeTile = 16;

__global__ void kernel_extract_time_slices_transpose(const cuDoubleComplex* __restrict__ F_batch,
                                                     const cuDoubleComplex* __restrict__ C_batch,
                                                     const cuDoubleComplex* __restrict__ R_batch,
                                                     cuDoubleComplex* __restrict__ F_out,
                                                     cuDoubleComplex* __restrict__ C_out,
                                                     cuDoubleComplex* __restrict__ R_out,
                                                     std::size_t Nt,
                                                     std::size_t base_idx,
                                                     std::size_t lane_count,
                                                     std::size_t nf3)
{
    __shared__ cuDoubleComplex tileF[kTransposeTile][kTransposeTile + 1];
    __shared__ cuDoubleComplex tileC[kTransposeTile][kTransposeTile + 1];
    __shared__ cuDoubleComplex tileR[kTransposeTile][kTransposeTile + 1];

    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * kTransposeTile + threadIdx.x; // time
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * kTransposeTile + threadIdx.y; // lane

    if (x < Nt && y < lane_count) {
        const std::size_t src = y * Nt + x;
        tileF[threadIdx.y][threadIdx.x] = F_batch[src];
        tileC[threadIdx.y][threadIdx.x] = C_batch[src];
        tileR[threadIdx.y][threadIdx.x] = R_batch[src];
    }
    __syncthreads();

    const std::size_t x2 = static_cast<std::size_t>(blockIdx.y) * kTransposeTile + threadIdx.x; // lane
    const std::size_t y2 = static_cast<std::size_t>(blockIdx.x) * kTransposeTile + threadIdx.y; // time
    if (x2 < lane_count && y2 < Nt) {
        const std::size_t dst = y2 * nf3 + base_idx + x2;
        F_out[dst] = tileF[threadIdx.x][threadIdx.y];
        C_out[dst] = tileC[threadIdx.x][threadIdx.y];
        R_out[dst] = tileR[threadIdx.x][threadIdx.y];
    }
}

__global__ void kernel_extract_time_slices_offset(const cuDoubleComplex* __restrict__ F_batch,
                                                  const cuDoubleComplex* __restrict__ C_batch,
                                                  const cuDoubleComplex* __restrict__ R_batch,
                                                  cuDoubleComplex* __restrict__ F_out,
                                                  cuDoubleComplex* __restrict__ C_out,
                                                  cuDoubleComplex* __restrict__ R_out,
                                                  const unsigned int* __restrict__ time_indices,
                                                  std::size_t num_times,
                                                  std::size_t Nt,
                                                  std::size_t base_idx,
                                                  std::size_t lane_count,
                                                  std::size_t nf3)
{
    const std::size_t lane = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t tpos = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (lane >= lane_count || tpos >= num_times) return;
    const std::size_t time_index = static_cast<std::size_t>(time_indices[tpos]);
    const std::size_t dst = tpos * nf3 + base_idx + lane;
    const std::size_t src = time_index + Nt * lane;
    F_out[dst] = F_batch[src];
    C_out[dst] = C_batch[src];
    R_out[dst] = R_batch[src];
}

__device__ __forceinline__ cuDoubleComplex cd_add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cuDoubleComplex cd_conj(cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x, -a.y);
}

// Fused symmetrize + GW->L4 permutation.
//
// Index conventions:
// - GW is stored as column-major over (row=(n,i), col=(m,j)).
// - L4 is stored as column-major over (row=(m,n), col=(j,i)).
// - Symmetrization is applied in GW-space: GW_sym = GW_raw + GW_raw^H.
__global__ void kernel_gw_raw_sym_to_liouvillian(const cuDoubleComplex* __restrict__ GW_raw,
                                                 cuDoubleComplex* __restrict__ L4,
                                                 int N)
{
    const std::size_t N_u = static_cast<std::size_t>(N);
    const std::size_t N2 = N_u * N_u;
    const std::size_t row_L = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t col_L = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row_L >= N2 || col_L >= N2) return;

    // row_L <-> (m,n)
    const std::size_t n = row_L % N_u;
    const std::size_t m = row_L / N_u;

    // col_L <-> (j,i)
    const std::size_t i = col_L % N_u;
    const std::size_t j = col_L / N_u;

    const std::size_t row_G = n + i * N_u;
    const std::size_t col_G = m + j * N_u;

    // Symmetrization in GW-space:
    //   GW_sym(row_G,col_G) = GW_raw(row_G,col_G) + conj(GW_raw(col_G,row_G))
    const cuDoubleComplex t_rc = GW_raw[row_G + col_G * N2];
    const cuDoubleComplex t_cr = GW_raw[col_G + row_G * N2];
    L4[row_L + col_L * N2] = cd_add(t_rc, cd_conj(t_cr));
}

} // namespace

Eigen::MatrixXcd build_TCL4_generator_cuda_fused(const sys::System& system,
                                                 const Eigen::MatrixXcd& gamma_series,
                                                 double dt,
                                                 std::size_t time_index,
                                                 FCRMethod method,
                                                 const Exec& exec)
{
    if (method != FCRMethod::Convolution) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: only FCRMethod::Convolution is supported");
    }
    if (time_index >= static_cast<std::size_t>(gamma_series.rows())) {
        throw std::out_of_range("build_TCL4_generator_cuda_fused: time_index out of range");
    }

    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    const std::size_t nf = static_cast<std::size_t>(gamma_series.cols());
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: gamma_series column count does not match frequency buckets");
    }
    if (nf == 0 || Nt == 0) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: empty gamma_series");
    }
    if (nf > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: nf too large for CUDA kernels");
    }

    Tcl4Map map = build_map(system, /*time_grid*/{});
    if (map.N <= 0) throw std::invalid_argument("build_TCL4_generator_cuda_fused: map.N must be > 0");
    if (map.nf <= 0) throw std::invalid_argument("build_TCL4_generator_cuda_fused: map.nf must be > 0");
    if (map.pair_to_freq.rows() != static_cast<Eigen::Index>(map.N) ||
        map.pair_to_freq.cols() != static_cast<Eigen::Index>(map.N)) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: map.pair_to_freq has wrong shape");
    }
    if (map.pair_to_freq.minCoeff() < 0) {
        throw std::runtime_error("build_TCL4_generator_cuda_fused: map.pair_to_freq contains -1 (missing frequency buckets)");
    }
    if (system.A_eig.empty()) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: coupling_ops must be non-empty");
    }

    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> must match cuDoubleComplex storage (2 doubles)");

    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t N2 = N * N;
    const std::size_t N6 = N * N * N * N * N * N;
    const std::size_t nf3 = nf * nf * nf;

    std::vector<double> h_omegas = map.omegas;
    std::vector<int> h_mirror = map.mirror_index;
    if (h_omegas.size() != nf || h_mirror.size() != nf) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: map frequency metadata has wrong size");
    }

    // Match CPU padding rule: Nfft = next_pow2(max(2*Nt-1, pad_factor*Nt))
    std::size_t L = 2 * Nt - 1;
    std::size_t target = L;
    const std::size_t pad_factor = get_fcr_fft_pad_factor();
    if (pad_factor > 0) {
        const std::size_t pf = pad_factor * Nt;
        if (pf > target) target = pf;
    }
    std::size_t Nfft = next_pow2(target);
    if (Nfft < 2) Nfft = 2;

    constexpr std::size_t kDefaultBatch = 64;
    const std::size_t Bplan = std::min(nf, kDefaultBatch);

    const std::size_t num_ops = system.A_eig.size();
    if (num_ops > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused: coupling_ops too large for CUDA kernel");
    }

    std::vector<std::complex<double>> h_ops(num_ops * N2);
    for (std::size_t op = 0; op < num_ops; ++op) {
        const auto& A = system.A_eig[op];
        if (A.rows() != static_cast<Eigen::Index>(N) || A.cols() != static_cast<Eigen::Index>(N)) {
            throw std::invalid_argument("build_TCL4_generator_cuda_fused: coupling operator has wrong shape");
        }
        std::copy(A.data(), A.data() + N2, h_ops.data() + op * N2);
    }

    cuda_check(cudaSetDevice(exec.gpu_id), "cudaSetDevice");
    FusedCudaWorkspace& ws = get_fused_workspace(exec.gpu_id);
    cudaStream_t stream = ws.stream;
    auto& fcr_ws = ws.fcr;

    cuDoubleComplex* d_gamma = nullptr;
    double* d_omegas = nullptr;
    int* d_mirror = nullptr;
    int* d_pair_to_freq = nullptr;
    cuDoubleComplex* d_ops = nullptr;

    cuDoubleComplex* d_F = nullptr;
    cuDoubleComplex* d_C = nullptr;
    cuDoubleComplex* d_R = nullptr;
    cuDoubleComplex* d_Ftmp = nullptr;
    cuDoubleComplex* d_Ctmp = nullptr;
    cuDoubleComplex* d_Rtmp = nullptr;

    cuDoubleComplex* dM = nullptr;
    cuDoubleComplex* dI = nullptr;
    cuDoubleComplex* dK = nullptr;
    cuDoubleComplex* dX = nullptr;
    cuDoubleComplex* dGW = nullptr;
    cuDoubleComplex* dL4 = nullptr;

    d_gamma = ensure_buffer<cuDoubleComplex>(ws.gamma, Nt * nf, "cudaMalloc(d_gamma)");
    d_omegas = ensure_buffer<double>(ws.omegas, nf, "cudaMalloc(d_omegas)");
    d_mirror = ensure_buffer<int>(ws.mirror, nf, "cudaMalloc(d_mirror)");
    d_pair_to_freq = ensure_buffer<int>(ws.pair_to_freq, N2, "cudaMalloc(d_pair_to_freq)");
    d_ops = ensure_buffer<cuDoubleComplex>(ws.ops, num_ops * N2, "cudaMalloc(d_ops)");

        cuda_check(cudaMemcpyAsync(d_gamma,
                                   gamma_series.data(),
                                   Nt * nf * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync(gamma)");
        cuda_check(cudaMemcpyAsync(d_omegas, h_omegas.data(), nf * sizeof(double),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(omegas)");
        cuda_check(cudaMemcpyAsync(d_mirror, h_mirror.data(), nf * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(mirror)");
        cuda_check(cudaMemcpyAsync(d_pair_to_freq, map.pair_to_freq.data(), N2 * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(pair_to_freq)");
        cuda_check(cudaMemcpyAsync(d_ops, h_ops.data(), num_ops * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(coupling_ops)");

    d_F = ensure_buffer<cuDoubleComplex>(ws.F, nf3, "cudaMalloc(d_F)");
    d_C = ensure_buffer<cuDoubleComplex>(ws.C, nf3, "cudaMalloc(d_C)");
    d_R = ensure_buffer<cuDoubleComplex>(ws.R, nf3, "cudaMalloc(d_R)");

    const std::size_t out_elems = Nt * Bplan;
    d_Ftmp = ensure_buffer<cuDoubleComplex>(ws.Ftmp, out_elems, "cudaMalloc(d_Ftmp)");
    d_Ctmp = ensure_buffer<cuDoubleComplex>(ws.Ctmp, out_elems, "cudaMalloc(d_Ctmp)");
    d_Rtmp = ensure_buffer<cuDoubleComplex>(ws.Rtmp, out_elems, "cudaMalloc(d_Rtmp)");

        cuda_fcr::FcrDeviceInputs inputs;
        inputs.gamma = d_gamma;
        inputs.omegas = d_omegas;
        inputs.mirror = d_mirror;
        inputs.Nt = Nt;
        inputs.nf = nf;
        inputs.dt = dt;

        for (std::size_t i = 0; i < nf; ++i) {
            for (std::size_t j = 0; j < nf; ++j) {
                std::size_t k0 = 0;
                for (; k0 + Bplan <= nf; k0 += Bplan) {
                    cuda_fcr::FcrBatch b;
                    b.batch = Bplan;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0);
                    b.F = d_Ftmp;
                    b.C = d_Ctmp;
                    b.R = d_Rtmp;

        cuda_fcr::compute_fcr_convolution_batched(inputs, b, fcr_ws, stream);

                    const std::size_t base_idx = (i * nf + j) * nf + k0;
                    copy_time_column(d_F + base_idx, d_Ftmp, Nt, Bplan, time_index, stream, "cudaMemcpy2DAsync(F)");
                    copy_time_column(d_C + base_idx, d_Ctmp, Nt, Bplan, time_index, stream, "cudaMemcpy2DAsync(C)");
                    copy_time_column(d_R + base_idx, d_Rtmp, Nt, Bplan, time_index, stream, "cudaMemcpy2DAsync(R)");
                }

                if (k0 < nf) {
                    const std::size_t rem = nf - k0;

                    cuda_fcr::FcrBatch b;
                    b.batch = rem;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0);
                    b.F = d_Ftmp;
                    b.C = d_Ctmp;
                    b.R = d_Rtmp;

        cuda_fcr::compute_fcr_convolution_batched(inputs, b, fcr_ws, stream);

                    const std::size_t base_idx = (i * nf + j) * nf + k0;
                    copy_time_column(d_F + base_idx, d_Ftmp, Nt, rem, time_index, stream, "cudaMemcpy2DAsync(F tail)");
                    copy_time_column(d_C + base_idx, d_Ctmp, Nt, rem, time_index, stream, "cudaMemcpy2DAsync(C tail)");
                    copy_time_column(d_R + base_idx, d_Rtmp, Nt, rem, time_index, stream, "cudaMemcpy2DAsync(R tail)");
                }
            }
        }

        dM = ensure_buffer<cuDoubleComplex>(ws.M, N2 * N2, "cudaMalloc(dM)");
        dI = ensure_buffer<cuDoubleComplex>(ws.I, N2 * N2, "cudaMalloc(dI)");
        dK = ensure_buffer<cuDoubleComplex>(ws.K, N2 * N2, "cudaMalloc(dK)");
        dX = ensure_buffer<cuDoubleComplex>(ws.X, N6, "cudaMalloc(dX)");

        cuda_mikx::MikxDeviceInputs in;
        in.F = d_F;
        in.C = d_C;
        in.R = d_R;
        in.pair_to_freq = d_pair_to_freq;
        in.N = map.N;
        in.nf = static_cast<int>(nf);

        cuda_mikx::MikxDeviceOutputs out;
        out.M = dM;
        out.I = dI;
        out.K = dK;
        out.X = dX;

        dGW = ensure_buffer<cuDoubleComplex>(ws.GW, N2 * N2, "cudaMalloc(dGW)");
        dL4 = ensure_buffer<cuDoubleComplex>(ws.L4, N2 * N2, "cudaMalloc(dL4)");

        bool used_graph = false;
        if (tcl4_cuda_graph_enabled()) {
            const std::size_t total = N2 * N2;
            FusedCudaWorkspace::GraphKey key{};
            key.gpu_id = exec.gpu_id;
            key.N = map.N;
            key.nf = static_cast<int>(nf);
            key.num_ops = static_cast<int>(num_ops);
            key.num_times = 1;

            FusedCudaWorkspace::PipelineArgs args{};
            args.F_all = d_F;
            args.C_all = d_C;
            args.R_all = d_R;
            args.pair_to_freq = d_pair_to_freq;
            args.ops = d_ops;
            args.M = dM;
            args.I = dI;
            args.K = dK;
            args.X = dX;
            args.GW_raw = dGW;
            args.L4_all = dL4;
            args.N = map.N;
            args.nf = static_cast<int>(nf);
            args.num_ops = static_cast<int>(num_ops);
            args.num_times = 1;
            args.nf3 = nf3;
            args.l4_stride = total;

            auto& g = ws.graphs[key];
            if (!g) {
                g = std::make_unique<FusedCudaWorkspace::Tcl4CudaGraph>();
                g->key = key;
            }
            used_graph = g->launch(stream, args);
            if (!used_graph && tcl4_cuda_graph_verbose()) {
                std::fprintf(stderr, "tcl4_cuda_graph: falling back to non-graph path\n");
            }
        }

        if (!used_graph) {
            cuda_mikx::build_mikx_device(in, out, stream);

            assemble_liouvillian_cuda_device_raw(dM, dI, dK, dX, d_ops,
                                                 map.N, static_cast<int>(num_ops), dGW, stream);

            constexpr unsigned block = 16;
            const dim3 block_l4(block, block);
            const dim3 grid_l4(static_cast<unsigned>((N2 + block_l4.x - 1) / block_l4.x),
                               static_cast<unsigned>((N2 + block_l4.y - 1) / block_l4.y));
            kernel_gw_raw_sym_to_liouvillian<<<grid_l4, block_l4, 0, stream>>>(dGW, dL4, map.N);
            cuda_check(cudaGetLastError(), "kernel_gw_raw_sym_to_liouvillian launch");
        }

        Eigen::MatrixXcd L4(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
        cuda_check(cudaMemcpyAsync(L4.data(), dL4, N2 * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(L4)");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        return L4;
}

std::vector<Eigen::MatrixXcd> build_TCL4_generator_cuda_fused_batch(const sys::System& system,
                                                                    const Eigen::MatrixXcd& gamma_series,
                                                                    double dt,
                                                                    const std::vector<std::size_t>& time_indices,
                                                                    FCRMethod method,
                                                                    const Exec& exec)
{
    return build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, time_indices, method, exec, nullptr);
}

std::vector<Eigen::MatrixXcd> build_TCL4_generator_cuda_fused_batch(const sys::System& system,
                                                                    const Eigen::MatrixXcd& gamma_series,
                                                                    double dt,
                                                                    const std::vector<std::size_t>& time_indices,
                                                                    FCRMethod method,
                                                                    const Exec& exec,
                                                                    double* cuda_fcr_ms)
{
    if (cuda_fcr_ms) *cuda_fcr_ms = 0.0;

    if (method != FCRMethod::Convolution) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: only FCRMethod::Convolution is supported");
    }

    const std::size_t Nt = static_cast<std::size_t>(gamma_series.rows());
    const std::size_t nf = static_cast<std::size_t>(gamma_series.cols());
    if (nf != system.fidx.buckets.size()) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: gamma_series column count does not match frequency buckets");
    }
    if (nf == 0 || Nt == 0) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: empty gamma_series");
    }
    if (nf > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: nf too large for CUDA kernels");
    }

    const bool use_all_times = time_indices.empty();
    std::vector<std::size_t> tids = time_indices;
    if (use_all_times) {
        tids.resize(Nt);
        for (std::size_t t = 0; t < Nt; ++t) tids[t] = t;
    }
    for (std::size_t tidx : tids) {
        if (tidx >= Nt) {
            throw std::out_of_range("build_TCL4_generator_cuda_fused_batch: time_index out of range");
        }
    }
    const std::size_t num_times = tids.size();
    if (num_times == 0) {
        return {};
    }
    std::vector<unsigned int> h_time_indices;
    if (!use_all_times) {
        if (Nt > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: Nt too large for time index list");
        }
        h_time_indices.resize(num_times);
        for (std::size_t i = 0; i < num_times; ++i) {
            if (tids[i] > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
                throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: time index too large");
            }
            h_time_indices[i] = static_cast<unsigned int>(tids[i]);
        }
    }

    Tcl4Map map = build_map(system, /*time_grid*/{});
    if (map.N <= 0) throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: map.N must be > 0");
    if (map.nf <= 0) throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: map.nf must be > 0");
    if (map.pair_to_freq.rows() != static_cast<Eigen::Index>(map.N) ||
        map.pair_to_freq.cols() != static_cast<Eigen::Index>(map.N)) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: map.pair_to_freq has wrong shape");
    }
    if (map.pair_to_freq.minCoeff() < 0) {
        throw std::runtime_error("build_TCL4_generator_cuda_fused_batch: map.pair_to_freq contains -1 (missing frequency buckets)");
    }
    if (system.A_eig.empty()) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: coupling_ops must be non-empty");
    }

    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> must match cuDoubleComplex storage (2 doubles)");

    const std::size_t N = static_cast<std::size_t>(map.N);
    const std::size_t N2 = N * N;
    const std::size_t N6 = N * N * N * N * N * N;
    const std::size_t nf3 = nf * nf * nf;

    std::vector<double> h_omegas = map.omegas;
    std::vector<int> h_mirror = map.mirror_index;
    if (h_omegas.size() != nf || h_mirror.size() != nf) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: map frequency metadata has wrong size");
    }

    // Match CPU padding rule: Nfft = next_pow2(max(2*Nt-1, pad_factor*Nt))
    std::size_t L = 2 * Nt - 1;
    std::size_t target = L;
    const std::size_t pad_factor = get_fcr_fft_pad_factor();
    if (pad_factor > 0) {
        const std::size_t pf = pad_factor * Nt;
        if (pf > target) target = pf;
    }
    std::size_t Nfft = next_pow2(target);
    if (Nfft < 2) Nfft = 2;

    constexpr std::size_t kDefaultBatch = 64;
    const std::size_t Bplan = std::min(nf, kDefaultBatch);

    const std::size_t num_ops = system.A_eig.size();
    if (num_ops > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: coupling_ops too large for CUDA kernel");
    }

    std::vector<std::complex<double>> h_ops(num_ops * N2);
    for (std::size_t op = 0; op < num_ops; ++op) {
        const auto& A = system.A_eig[op];
        if (A.rows() != static_cast<Eigen::Index>(N) || A.cols() != static_cast<Eigen::Index>(N)) {
            throw std::invalid_argument("build_TCL4_generator_cuda_fused_batch: coupling operator has wrong shape");
        }
        std::copy(A.data(), A.data() + N2, h_ops.data() + op * N2);
    }

    const std::size_t elems_per = N2 * N2;
    if (elems_per > 0 && tids.size() > (std::numeric_limits<std::size_t>::max() / elems_per)) {
        throw std::overflow_error("build_TCL4_generator_cuda_fused_batch: output too large");
    }
    const std::size_t total_out_elems = elems_per * tids.size();
    if (nf3 > 0 && num_times > (std::numeric_limits<std::size_t>::max() / nf3)) {
        throw std::overflow_error("build_TCL4_generator_cuda_fused_batch: FCR buffer too large");
    }
    const std::size_t total_fcr_elems = nf3 * num_times;

    cuda_check(cudaSetDevice(exec.gpu_id), "cudaSetDevice");
    FusedCudaWorkspace& ws = get_fused_workspace(exec.gpu_id);
    cudaStream_t stream = ws.stream;
    auto& fcr_ws = ws.fcr;

    CudaEvent ev_fcr_start;
    CudaEvent ev_fcr_stop;
    if (cuda_fcr_ms) {
        ev_fcr_start = CudaEvent(cudaEventDefault);
        ev_fcr_stop = CudaEvent(cudaEventDefault);
    }

    cuDoubleComplex* d_gamma = nullptr;
    double* d_omegas = nullptr;
    int* d_mirror = nullptr;
    int* d_pair_to_freq = nullptr;
    cuDoubleComplex* d_ops = nullptr;
    unsigned int* d_time_indices = nullptr;

    cuDoubleComplex* d_F_all = nullptr;
    cuDoubleComplex* d_C_all = nullptr;
    cuDoubleComplex* d_R_all = nullptr;
    cuDoubleComplex* d_Ftmp = nullptr;
    cuDoubleComplex* d_Ctmp = nullptr;
    cuDoubleComplex* d_Rtmp = nullptr;

    cuDoubleComplex* dM = nullptr;
    cuDoubleComplex* dI = nullptr;
    cuDoubleComplex* dK = nullptr;
    cuDoubleComplex* dX = nullptr;
    cuDoubleComplex* dGW = nullptr;
    cuDoubleComplex* dL4_all = nullptr;

    d_gamma = ensure_buffer<cuDoubleComplex>(ws.gamma, Nt * nf, "cudaMalloc(d_gamma)");
    d_omegas = ensure_buffer<double>(ws.omegas, nf, "cudaMalloc(d_omegas)");
    d_mirror = ensure_buffer<int>(ws.mirror, nf, "cudaMalloc(d_mirror)");
    d_pair_to_freq = ensure_buffer<int>(ws.pair_to_freq, N2, "cudaMalloc(d_pair_to_freq)");
    d_ops = ensure_buffer<cuDoubleComplex>(ws.ops, num_ops * N2, "cudaMalloc(d_ops)");
    if (!use_all_times) {
        d_time_indices = ensure_buffer<unsigned int>(ws.time_indices, num_times, "cudaMalloc(d_time_indices)");
    }

        cuda_check(cudaMemcpyAsync(d_gamma,
                                   gamma_series.data(),
                                   Nt * nf * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync(gamma)");
        cuda_check(cudaMemcpyAsync(d_omegas, h_omegas.data(), nf * sizeof(double),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(omegas)");
        cuda_check(cudaMemcpyAsync(d_mirror, h_mirror.data(), nf * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(mirror)");
        cuda_check(cudaMemcpyAsync(d_pair_to_freq, map.pair_to_freq.data(), N2 * sizeof(int),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(pair_to_freq)");
        cuda_check(cudaMemcpyAsync(d_ops, h_ops.data(), num_ops * N2 * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(coupling_ops)");
        if (!use_all_times) {
            cuda_check(cudaMemcpyAsync(d_time_indices, h_time_indices.data(), num_times * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(time_indices)");
        }

    d_F_all = ensure_buffer<cuDoubleComplex>(ws.F_all, total_fcr_elems, "cudaMalloc(d_F_all)");
    d_C_all = ensure_buffer<cuDoubleComplex>(ws.C_all, total_fcr_elems, "cudaMalloc(d_C_all)");
    d_R_all = ensure_buffer<cuDoubleComplex>(ws.R_all, total_fcr_elems, "cudaMalloc(d_R_all)");

    const std::size_t out_elems = Nt * Bplan;
    d_Ftmp = ensure_buffer<cuDoubleComplex>(ws.Ftmp, out_elems, "cudaMalloc(d_Ftmp)");
    d_Ctmp = ensure_buffer<cuDoubleComplex>(ws.Ctmp, out_elems, "cudaMalloc(d_Ctmp)");
    d_Rtmp = ensure_buffer<cuDoubleComplex>(ws.Rtmp, out_elems, "cudaMalloc(d_Rtmp)");

    dM = ensure_buffer<cuDoubleComplex>(ws.M, N2 * N2, "cudaMalloc(dM)");
    dI = ensure_buffer<cuDoubleComplex>(ws.I, N2 * N2, "cudaMalloc(dI)");
    dK = ensure_buffer<cuDoubleComplex>(ws.K, N2 * N2, "cudaMalloc(dK)");
    dX = ensure_buffer<cuDoubleComplex>(ws.X, N6, "cudaMalloc(dX)");
    dGW = ensure_buffer<cuDoubleComplex>(ws.GW, N2 * N2, "cudaMalloc(dGW)");
    dL4_all = ensure_buffer<cuDoubleComplex>(ws.L4_all, total_out_elems, "cudaMalloc(dL4_all)");

        cuda_fcr::FcrDeviceInputs inputs;
        inputs.gamma = d_gamma;
        inputs.omegas = d_omegas;
        inputs.mirror = d_mirror;
        inputs.Nt = Nt;
        inputs.nf = nf;
        inputs.dt = dt;

        cuda_mikx::MikxDeviceInputs in;
        in.pair_to_freq = d_pair_to_freq;
        in.N = map.N;
        in.nf = static_cast<int>(nf);

        cuda_mikx::MikxDeviceOutputs out;
        out.M = dM;
        out.I = dI;
        out.K = dK;
        out.X = dX;

        const std::size_t total = N2 * N2;
        constexpr int block_extract_x = 256;
        constexpr int block_extract_y = 4;
        const dim3 block_extract(block_extract_x, block_extract_y);
        const dim3 block_transpose(kTransposeTile, kTransposeTile);
        constexpr std::size_t kSmallTimeMemcpyThreshold = 8;
        const bool use_small_time_memcpy = (!use_all_times && num_times <= kSmallTimeMemcpyThreshold);

        if (cuda_fcr_ms) {
            cuda_check(cudaEventRecord(ev_fcr_start.ev, stream), "cudaEventRecord(fcr_start)");
        }

        for (std::size_t i = 0; i < nf; ++i) {
            for (std::size_t j = 0; j < nf; ++j) {
                std::size_t k0 = 0;
                for (; k0 + Bplan <= nf; k0 += Bplan) {
                    cuda_fcr::FcrBatch b;
                    b.batch = Bplan;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0);
                    b.F = d_Ftmp;
                    b.C = d_Ctmp;
                    b.R = d_Rtmp;

                    cuda_fcr::compute_fcr_convolution_batched(inputs, b, fcr_ws, stream);

                    const std::size_t base_idx = (i * nf + j) * nf + k0;
                    if (use_all_times) {
                        const dim3 grid_transpose(static_cast<unsigned>((Nt + kTransposeTile - 1) / kTransposeTile),
                                                  static_cast<unsigned>((Bplan + kTransposeTile - 1) / kTransposeTile));
                        kernel_extract_time_slices_transpose<<<grid_transpose, block_transpose, 0, stream>>>(
                            d_Ftmp, d_Ctmp, d_Rtmp,
                            d_F_all, d_C_all, d_R_all,
                            Nt, base_idx, Bplan, nf3);
                        cuda_check(cudaGetLastError(), "kernel_extract_time_slices_transpose launch");
                    } else if (use_small_time_memcpy) {
                        for (std::size_t tpos = 0; tpos < num_times; ++tpos) {
                            const std::size_t time_index = tids[tpos];
                            copy_time_column(d_F_all + tpos * nf3 + base_idx,
                                             d_Ftmp, Nt, Bplan, time_index, stream,
                                             "cudaMemcpy2DAsync(F_all)");
                            copy_time_column(d_C_all + tpos * nf3 + base_idx,
                                             d_Ctmp, Nt, Bplan, time_index, stream,
                                             "cudaMemcpy2DAsync(C_all)");
                            copy_time_column(d_R_all + tpos * nf3 + base_idx,
                                             d_Rtmp, Nt, Bplan, time_index, stream,
                                             "cudaMemcpy2DAsync(R_all)");
                        }
                    } else {
                        const dim3 grid_extract(static_cast<unsigned>((Bplan + block_extract_x - 1) / block_extract_x),
                                                static_cast<unsigned>((num_times + block_extract_y - 1) / block_extract_y));
                        kernel_extract_time_slices_offset<<<grid_extract, block_extract, 0, stream>>>(
                            d_Ftmp, d_Ctmp, d_Rtmp,
                            d_F_all, d_C_all, d_R_all,
                            d_time_indices, num_times, Nt, base_idx, Bplan, nf3);
                        cuda_check(cudaGetLastError(), "kernel_extract_time_slices_offset launch");
                    }
                }

                if (k0 < nf) {
                    const std::size_t rem = nf - k0;

                    cuda_fcr::FcrBatch b;
                    b.batch = rem;
                    b.Nfft = Nfft;
                    b.i = static_cast<int>(i);
                    b.j = static_cast<int>(j);
                    b.k0 = static_cast<int>(k0);
                    b.F = d_Ftmp;
                    b.C = d_Ctmp;
                    b.R = d_Rtmp;

                    cuda_fcr::compute_fcr_convolution_batched(inputs, b, fcr_ws, stream);

                    const std::size_t base_idx = (i * nf + j) * nf + k0;
                    if (use_all_times) {
                        const dim3 grid_transpose(static_cast<unsigned>((Nt + kTransposeTile - 1) / kTransposeTile),
                                                  static_cast<unsigned>((rem + kTransposeTile - 1) / kTransposeTile));
                        kernel_extract_time_slices_transpose<<<grid_transpose, block_transpose, 0, stream>>>(
                            d_Ftmp, d_Ctmp, d_Rtmp,
                            d_F_all, d_C_all, d_R_all,
                            Nt, base_idx, rem, nf3);
                        cuda_check(cudaGetLastError(), "kernel_extract_time_slices_transpose tail launch");
                    } else if (use_small_time_memcpy) {
                        for (std::size_t tpos = 0; tpos < num_times; ++tpos) {
                            const std::size_t time_index = tids[tpos];
                            copy_time_column(d_F_all + tpos * nf3 + base_idx,
                                             d_Ftmp, Nt, rem, time_index, stream,
                                             "cudaMemcpy2DAsync(F_all tail)");
                            copy_time_column(d_C_all + tpos * nf3 + base_idx,
                                             d_Ctmp, Nt, rem, time_index, stream,
                                             "cudaMemcpy2DAsync(C_all tail)");
                            copy_time_column(d_R_all + tpos * nf3 + base_idx,
                                             d_Rtmp, Nt, rem, time_index, stream,
                                             "cudaMemcpy2DAsync(R_all tail)");
                        }
                    } else {
                        const dim3 grid_extract(static_cast<unsigned>((rem + block_extract_x - 1) / block_extract_x),
                                                static_cast<unsigned>((num_times + block_extract_y - 1) / block_extract_y));
                        kernel_extract_time_slices_offset<<<grid_extract, block_extract, 0, stream>>>(
                            d_Ftmp, d_Ctmp, d_Rtmp,
                            d_F_all, d_C_all, d_R_all,
                            d_time_indices, num_times, Nt, base_idx, rem, nf3);
                        cuda_check(cudaGetLastError(), "kernel_extract_time_slices_offset tail launch");
                    }
                }
            }
        }

        if (cuda_fcr_ms) {
            cuda_check(cudaEventRecord(ev_fcr_stop.ev, stream), "cudaEventRecord(fcr_stop)");
        }

        constexpr unsigned block = 16;
        const dim3 block_l4(block, block);
        const dim3 grid_l4(static_cast<unsigned>((N2 + block_l4.x - 1) / block_l4.x),
                           static_cast<unsigned>((N2 + block_l4.y - 1) / block_l4.y));

        bool used_graph = false;
        if (tcl4_cuda_graph_enabled()) {
            FusedCudaWorkspace::GraphKey key{};
            key.gpu_id = exec.gpu_id;
            key.N = map.N;
            key.nf = static_cast<int>(nf);
            key.num_ops = static_cast<int>(num_ops);
            key.num_times = num_times;

            FusedCudaWorkspace::PipelineArgs args{};
            args.F_all = d_F_all;
            args.C_all = d_C_all;
            args.R_all = d_R_all;
            args.pair_to_freq = d_pair_to_freq;
            args.ops = d_ops;
            args.M = dM;
            args.I = dI;
            args.K = dK;
            args.X = dX;
            args.GW_raw = dGW;
            args.L4_all = dL4_all;
            args.N = map.N;
            args.nf = static_cast<int>(nf);
            args.num_ops = static_cast<int>(num_ops);
            args.num_times = num_times;
            args.nf3 = nf3;
            args.l4_stride = total;

            auto& g = ws.graphs[key];
            if (!g) {
                g = std::make_unique<FusedCudaWorkspace::Tcl4CudaGraph>();
                g->key = key;
            }
            used_graph = g->launch(stream, args);
            if (!used_graph && tcl4_cuda_graph_verbose()) {
                std::fprintf(stderr, "tcl4_cuda_graph: falling back to non-graph path\n");
            }
        }

        if (!used_graph) {
            for (std::size_t out_idx = 0; out_idx < num_times; ++out_idx) {
                in.F = d_F_all + out_idx * nf3;
                in.C = d_C_all + out_idx * nf3;
                in.R = d_R_all + out_idx * nf3;

                cuda_mikx::build_mikx_device(in, out, stream);

                cuDoubleComplex* dL4_out = dL4_all + out_idx * total;

                assemble_liouvillian_cuda_device_raw(dM, dI, dK, dX, d_ops,
                                                     map.N, static_cast<int>(num_ops), dGW, stream);

                kernel_gw_raw_sym_to_liouvillian<<<grid_l4, block_l4, 0, stream>>>(dGW, dL4_out, map.N);
                cuda_check(cudaGetLastError(), "kernel_gw_raw_sym_to_liouvillian launch");
            }
        }

        std::vector<std::complex<double>> hL4(total_out_elems);
        cuda_check(cudaMemcpyAsync(hL4.data(), dL4_all, total_out_elems * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(L4_all)");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        if (cuda_fcr_ms) {
            float ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&ms, ev_fcr_start.ev, ev_fcr_stop.ev),
                       "cudaEventElapsedTime(fcr)");
            *cuda_fcr_ms = static_cast<double>(ms);
        }

        std::vector<Eigen::MatrixXcd> out_series;
        out_series.resize(tids.size());
        for (std::size_t idx = 0; idx < tids.size(); ++idx) {
            Eigen::MatrixXcd L4(static_cast<Eigen::Index>(N2), static_cast<Eigen::Index>(N2));
            const std::complex<double>* src = hL4.data() + idx * total;
            std::copy(src, src + total, L4.data());
            out_series[idx] = std::move(L4);
        }

        return out_series;
}

std::vector<Eigen::MatrixXcd> build_correction_series_cuda_fused(const sys::System& system,
                                                                 const Eigen::MatrixXcd& gamma_series,
                                                                 double dt,
                                                                 FCRMethod method,
                                                                 const Exec& exec)
{
    return build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, {}, method, exec);
}

} // namespace taco::tcl4
