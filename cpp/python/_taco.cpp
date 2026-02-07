#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "taco/correlation_fft.hpp"
#include "taco/exec.hpp"
#include "taco/gamma.hpp"
#include "taco/generator.hpp"
#include "taco/ops.hpp"
#include "taco/rk4_dense.hpp"
#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_mikx.hpp"
#include "taco/version.hpp"

#ifdef TACO_HAS_CUDA
#include <cuda_runtime_api.h>
#include "taco/backend/cuda/tcl4_fused_cuda.hpp"
#include "taco/backend/cuda/rk4_dense_cuda.hpp"
#endif

namespace py = pybind11;

namespace taco::python {

using cd = std::complex<double>;
using Matrix = Eigen::MatrixXcd;

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

double max_abs_diff(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return std::numeric_limits<double>::infinity();
    return (a - b).cwiseAbs().maxCoeff();
}

double max_abs_diff(const Eigen::VectorXcd& a, const Eigen::VectorXcd& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    return (a - b).cwiseAbs().maxCoeff();
}

std::size_t clamp_tidx(std::size_t tidx, std::size_t Nt) {
    if (Nt == 0) return 0;
    return std::min(tidx, Nt - 1);
}

std::vector<std::size_t> parse_tidx_spec(const std::string& spec, std::size_t Nt) {
    if (spec.empty()) return {};

    std::vector<std::size_t> parts;
    parts.reserve(3);
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t next = spec.find(':', pos);
        const std::size_t len = (next == std::string::npos) ? (spec.size() - pos) : (next - pos);
        const std::string token = spec.substr(pos, len);
        if (token.empty()) {
            throw std::invalid_argument("invalid tidx spec (empty token)");
        }
        parts.push_back(static_cast<std::size_t>(std::stoull(token)));
        if (next == std::string::npos) break;
        pos = next + 1;
    }

    if (parts.size() == 1) {
        return {clamp_tidx(parts[0], Nt)};
    }

    std::size_t start = 0;
    std::size_t step = 1;
    std::size_t end = 0;
    if (parts.size() == 2) {
        start = parts[0];
        end = parts[1];
    } else if (parts.size() == 3) {
        start = parts[0];
        step = parts[1];
        end = parts[2];
    } else {
        throw std::invalid_argument("invalid tidx spec (expected k or a:b or a:step:b)");
    }

    if (step == 0) {
        throw std::invalid_argument("invalid tidx spec (step must be > 0)");
    }

    start = clamp_tidx(start, Nt);
    end = clamp_tidx(end, Nt);
    if (start > end) {
        throw std::invalid_argument("invalid tidx spec (start must be <= end)");
    }

    std::vector<std::size_t> out;
    out.reserve((end - start) / step + 1);
    for (std::size_t t = start; t <= end; t += step) {
        out.push_back(t);
    }
    return out;
}

std::vector<std::size_t> parse_tidx_from_python(const py::object& tidx, std::size_t Nt) {
    if (!tidx || tidx.is_none()) {
        if (Nt == 0) return {};
        std::vector<std::size_t> out = {0, Nt / 2, Nt - 1};
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    if (py::isinstance<py::str>(tidx)) {
        std::string spec = py::cast<std::string>(tidx);
        spec = to_lower(std::move(spec));
        if (spec.empty()) {
            return parse_tidx_from_python(py::none(), Nt);
        }
        if (spec == "series" || spec == "all") {
            std::vector<std::size_t> out;
            out.resize(Nt);
            for (std::size_t i = 0; i < Nt; ++i) out[i] = i;
            return out;
        }
        return parse_tidx_spec(spec, Nt);
    }

    std::vector<std::size_t> out;
    for (py::handle h : py::iterable(tidx)) {
        out.push_back(clamp_tidx(py::cast<std::size_t>(h), Nt));
    }
    if (out.empty()) return out;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct TabulatedSpectrum {
    std::vector<double> omega;
    std::vector<double> J;

    double operator()(double w) const {
        if (!(w > 0.0)) return 0.0;
        if (omega.empty()) return 0.0;
        if (w <= omega.front()) return J.front();
        if (w >= omega.back()) return 0.0;

        auto it = std::lower_bound(omega.begin(), omega.end(), w);
        if (it == omega.begin()) return J.front();
        if (it == omega.end()) return 0.0;

        const std::size_t i1 = static_cast<std::size_t>(it - omega.begin());
        const std::size_t i0 = i1 - 1;
        const double w0 = omega[i0];
        const double w1 = omega[i1];
        const double J0 = J[i0];
        const double J1 = J[i1];
        if (!(w1 > w0)) return J0;
        const double t = (w - w0) / (w1 - w0);
        return J0 + t * (J1 - J0);
    }
};

py::array ensure_numpy_array(const py::handle& obj, const char* name) {
    py::array arr = py::array::ensure(obj);
    if (!arr) throw py::type_error(std::string(name) + " must be a NumPy array");
    return arr;
}

py::array_t<cd, py::array::c_style> require_c_contig_complex128_2d(const py::handle& obj,
                                                                   const char* name,
                                                                   std::size_t& N_out) {
    py::array arr = ensure_numpy_array(obj, name);
    if (arr.ndim() != 2) throw py::value_error(std::string(name) + " must be a 2D array");
    const auto n0 = static_cast<std::size_t>(arr.shape(0));
    const auto n1 = static_cast<std::size_t>(arr.shape(1));
    if (n0 == 0 || n1 == 0) throw py::value_error(std::string(name) + " must be non-empty");
    if (n0 != n1) throw py::value_error(std::string(name) + " must be square (N,N)");
    if (!arr.dtype().is(py::dtype::of<cd>())) {
        throw py::type_error(std::string(name) + " must have dtype complex128");
    }
    N_out = n0;
    return py::array_t<cd, py::array::c_style>(arr);
}

py::array_t<double, py::array::c_style> require_c_contig_float64_1d(const py::handle& obj,
                                                                    const char* name) {
    py::array arr = ensure_numpy_array(obj, name);
    if (arr.ndim() != 1) throw py::value_error(std::string(name) + " must be a 1D array");
    if (arr.size() == 0) throw py::value_error(std::string(name) + " must be non-empty");
    if (!arr.dtype().is(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " must have dtype float64");
    }
    return py::array_t<double, py::array::c_style>(arr);
}

py::array_t<cd, py::array::c_style> require_c_contig_complex128_1d(const py::handle& obj,
                                                                   const char* name) {
    py::array arr = ensure_numpy_array(obj, name);
    if (arr.ndim() != 1) throw py::value_error(std::string(name) + " must be a 1D array");
    if (arr.size() == 0) throw py::value_error(std::string(name) + " must be non-empty");
    if (!arr.dtype().is(py::dtype::of<cd>())) {
        throw py::type_error(std::string(name) + " must have dtype complex128");
    }
    return py::array_t<cd, py::array::c_style>(arr);
}

Eigen::MatrixXcd eigen_from_c_rowmajor_complex128(const py::array_t<cd, py::array::c_style>& a, std::size_t N) {
    using MatrixRM = Eigen::Matrix<cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const auto* ptr = static_cast<const cd*>(a.data());
    Eigen::Map<const MatrixRM> m(ptr, static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(N));
    return Eigen::MatrixXcd(m);
}

TabulatedSpectrum spectrum_from_numpy(const py::array_t<double, py::array::c_style>& omega_arr,
                                      const py::array_t<double, py::array::c_style>& J_arr,
                                      const char* omega_name,
                                      const char* J_name) {
    const std::size_t n_omega = static_cast<std::size_t>(omega_arr.size());
    const std::size_t n_J = static_cast<std::size_t>(J_arr.size());
    if (n_omega != n_J) {
        throw py::value_error(std::string(omega_name) + " and " + std::string(J_name) + " must have the same length");
    }
    TabulatedSpectrum spec;
    spec.omega.assign(omega_arr.data(), omega_arr.data() + n_omega);
    spec.J.assign(J_arr.data(), J_arr.data() + n_J);
    for (std::size_t i = 1; i < spec.omega.size(); ++i) {
        if (!(spec.omega[i] > spec.omega[i - 1])) {
            throw py::value_error(std::string(omega_name) + " must be strictly increasing");
        }
    }
    return spec;
}

double beta_from_temperature(double temperature) {
    if (!std::isfinite(temperature)) throw py::value_error("bath.temperature must be finite");
    if (temperature < 0.0) throw py::value_error("bath.temperature must be >= 0");
    if (temperature == 0.0) return std::numeric_limits<double>::infinity();
    return 1.0 / temperature;
}

std::size_t steps_from_end_time(double dt, double t_end) {
    if (!std::isfinite(t_end)) throw py::value_error("t_end must be finite");
    if (t_end < 0.0) throw py::value_error("t_end must be >= 0");
    if (t_end == 0.0) return 0;
    return static_cast<std::size_t>(std::llround(std::ceil(t_end / dt)));
}

std::size_t compute_saved_count(std::size_t n_steps, std::size_t save_stride) {
    const std::size_t base = n_steps / save_stride + 1;  // includes step 0
    if (n_steps % save_stride == 0) return base;
    return base + 1;  // include final step
}

bool should_save_step(std::size_t step, std::size_t n_steps, std::size_t save_stride) {
    if (step == n_steps) return true;
    return (step % save_stride) == 0;
}

std::vector<cd> pad_or_truncate_bcf(const py::array_t<cd, py::array::c_style>& bcf, std::size_t Nt_sim) {
    std::vector<cd> Ccorr(Nt_sim, cd{0.0, 0.0});
    const std::size_t n_in = static_cast<std::size_t>(bcf.size());
    const std::size_t copy_len = std::min(n_in, Nt_sim);
    const auto* src = static_cast<const cd*>(bcf.data());
    std::copy(src, src + static_cast<std::ptrdiff_t>(copy_len), Ccorr.begin());
    return Ccorr;
}

#ifdef TACO_HAS_CUDA
inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template <typename BuildLAt, typename WriteState>
void propagate_rk4_dense_cuda_fp32(const Eigen::MatrixXcd& L0,
                                   BuildLAt&& build_L_at,
                                   Eigen::VectorXcd& r,
                                   std::size_t dim,
                                   std::size_t n_steps,
                                   std::size_t save_stride,
                                   double dt,
                                   int order,
                                   std::size_t& out_index,
                                   WriteState&& write_state) {
    const std::size_t D_u = dim * dim;
    if (D_u == 0 || D_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("CUDA RK4 (fp32): state dimension too large for int indexing");
    }
    if (D_u > std::numeric_limits<std::size_t>::max() / D_u) {
        throw std::overflow_error("CUDA RK4 (fp32): dense matrix size overflow");
    }

    const int D = static_cast<int>(D_u);
    const std::size_t L_elems = D_u * D_u;
    const float dt_f32 = static_cast<float>(dt);

    const std::size_t vbytes = D_u * sizeof(cuFloatComplex);
    const std::size_t Lbytes = L_elems * sizeof(cuFloatComplex);
    const cudaStream_t stream = 0;

    auto pack_vec = [&](const Eigen::VectorXcd& src, std::vector<cuFloatComplex>& dst) {
        dst.resize(D_u);
        for (std::size_t i = 0; i < D_u; ++i) {
            const cd z = src(static_cast<Eigen::Index>(i));
            dst[i] = make_cuFloatComplex(static_cast<float>(z.real()), static_cast<float>(z.imag()));
        }
    };
    auto unpack_vec = [&](const std::vector<cuFloatComplex>& src, Eigen::VectorXcd& dst) {
        for (std::size_t i = 0; i < D_u; ++i) {
            const auto z = src[i];
            dst(static_cast<Eigen::Index>(i)) = cd(static_cast<double>(z.x), static_cast<double>(z.y));
        }
    };
    auto pack_mat = [&](const Eigen::MatrixXcd& src, std::vector<cuFloatComplex>& dst) {
        dst.resize(L_elems);
        const auto* p = src.data(); // column-major
        for (std::size_t i = 0; i < L_elems; ++i) {
            dst[i] = make_cuFloatComplex(static_cast<float>(p[i].real()), static_cast<float>(p[i].imag()));
        }
    };

    std::vector<cuFloatComplex> h_r;
    std::vector<cuFloatComplex> h_L;

    cuFloatComplex* d_r = nullptr;
    cuda_check(cudaMalloc(&d_r, vbytes), "cudaMalloc(r)");
    auto free_r = [&] { if (d_r) cudaFree(d_r); };

    taco::tcl::Rk4DenseCudaWorkspaceF32 ws_cuda;

    try {
        pack_vec(r, h_r);
        cuda_check(cudaMemcpy(d_r, h_r.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r H2D)");

        if (order == 0) {
            cuFloatComplex* d_L = nullptr;
            cuda_check(cudaMalloc(&d_L, Lbytes), "cudaMalloc(L)");
            auto free_L = [&] {
                if (d_L) cudaFree(d_L);
                d_L = nullptr;
            };

            try {
                pack_mat(L0, h_L);
                cuda_check(cudaMemcpy(d_L, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L H2D)");

                for (std::size_t step = 0; step < n_steps; ++step) {
                    taco::tcl::rk4_update_cuda_f32(d_L, d_r, D, ws_cuda, dt_f32, stream);

                    const std::size_t step1 = step + 1;
                    if (should_save_step(step1, n_steps, save_stride)) {
                        cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r D2H)");
                        unpack_vec(h_r, r);
                        write_state(out_index++, step1);
                    }
                }
            } catch (...) {
                free_L();
                throw;
            }

            free_L();
        } else {
            cuFloatComplex* d_L0 = nullptr;
            cuFloatComplex* d_L1 = nullptr;
            cuFloatComplex* d_Lhalf = nullptr;
            cuda_check(cudaMalloc(&d_L0, Lbytes), "cudaMalloc(L0)");
            cuda_check(cudaMalloc(&d_L1, Lbytes), "cudaMalloc(L1)");
            cuda_check(cudaMalloc(&d_Lhalf, Lbytes), "cudaMalloc(Lhalf)");

            auto free_mats = [&] {
                if (d_L0) cudaFree(d_L0);
                if (d_L1) cudaFree(d_L1);
                if (d_Lhalf) cudaFree(d_Lhalf);
            };

            try {
                Eigen::MatrixXcd L_cur = build_L_at(0);
                Eigen::MatrixXcd L_next = build_L_at(1);

                pack_mat(L_cur, h_L);
                cuda_check(cudaMemcpy(d_L0, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L0 H2D)");
                pack_mat(L_next, h_L);
                cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L1 H2D)");

                for (std::size_t step = 0; step < n_steps; ++step) {
                    taco::tcl::half_sum_cuda_f32(d_L0, d_L1, d_Lhalf, L_elems, stream);
                    taco::tcl::rk4_update_cuda_f32(d_L0, d_Lhalf, d_L1, d_r, D, ws_cuda, dt_f32, stream);

                    const std::size_t step1 = step + 1;
                    if (should_save_step(step1, n_steps, save_stride)) {
                        cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r D2H)");
                        unpack_vec(h_r, r);
                        write_state(out_index++, step1);
                    }

                    if (step1 < n_steps) {
                        L_cur = std::move(L_next);
                        L_next = build_L_at(step + 2);

                        std::swap(d_L0, d_L1);
                        pack_mat(L_next, h_L);
                        cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(Lnext H2D)");
                    }
                }
            } catch (...) {
                free_mats();
                throw;
            }

            free_mats();
        }
    } catch (...) {
        free_r();
        throw;
    }

    free_r();
}
#endif // TACO_HAS_CUDA

} // namespace

py::dict build_info() {
    py::dict out;

#ifdef TACO_HAS_CUDA
    out["cuda_enabled"] = true;
#else
    out["cuda_enabled"] = false;
#endif

#ifdef _OPENMP
    out["openmp_enabled"] = true;
#else
    out["openmp_enabled"] = false;
#endif

    py::dict compiler;
#if defined(__clang__)
    compiler["id"] = "clang";
    compiler["version"] = __clang_version__;
#elif defined(_MSC_VER)
    compiler["id"] = "msvc";
    compiler["version"] = std::to_string(_MSC_VER);
#elif defined(__GNUC__)
    compiler["id"] = "gcc";
    compiler["version"] = __VERSION__;
#else
    compiler["id"] = "unknown";
    compiler["version"] = "";
#endif
    out["compiler"] = compiler;

#ifdef TACO_HAS_CUDA
    int runtime_ver = 0;
    if (cudaRuntimeGetVersion(&runtime_ver) == cudaSuccess) {
        out["cuda_runtime_version"] = runtime_ver;
    } else {
        out["cuda_runtime_version"] = py::none();
    }
#else
    out["cuda_runtime_version"] = py::none();
#endif

    return out;
}

int cuda_device_count() {
#ifdef TACO_HAS_CUDA
    int count = 0;
    const auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
#else
    return 0;
#endif
}

bool cuda_is_available() {
    return cuda_device_count() > 0;
}

py::array tcl_precompute_bcf(double temperature,
                             double dt,
                             double bcf_end_time,
                             py::handle omega,
                             py::handle J) {
    if (!(dt > 0.0)) throw py::value_error("cfg.dt must be > 0");
    if (!std::isfinite(bcf_end_time)) throw py::value_error("bath.bcf_end_time must be finite");
    if (bcf_end_time < 0.0) throw py::value_error("bath.bcf_end_time must be >= 0");

    const double beta = beta_from_temperature(temperature);

    auto omega_arr = require_c_contig_float64_1d(omega, "bath.omega");
    auto J_arr = require_c_contig_float64_1d(J, "bath.J");
    const auto spec = spectrum_from_numpy(omega_arr, J_arr, "bath.omega", "bath.J");

    const std::size_t N_bcf = steps_from_end_time(dt, bcf_end_time);

    std::vector<double> tgrid;
    std::vector<cd> Ccorr;
    bcf::bcf_fft_fun(N_bcf, dt, spec, beta, tgrid, Ccorr);

    py::array_t<cd> out(static_cast<py::ssize_t>(Ccorr.size()));
    auto* dst = static_cast<cd*>(out.mutable_data());
    std::copy(Ccorr.begin(), Ccorr.end(), dst);
    return out;
}

std::tuple<py::array, py::array> tcl_simulate(py::handle H,
                                              py::handle A,
                                              py::handle rho0,
                                              double temperature,
                                              double dt,
                                              std::size_t n_steps,
                                              std::size_t save_stride,
                                              double bcf_end_time,
                                              py::handle omega,
                                              py::handle J,
                                              std::string device,
                                              std::string precision,
                                              int order,
                                              int gpu_id) {
    if (!(dt > 0.0)) throw py::value_error("cfg.dt must be > 0");
    if (save_stride == 0) throw py::value_error("cfg.save_stride must be >= 1");
    if (!(order == 0 || order == 2 || order == 4)) throw py::value_error("cfg.order must be 0, 2, or 4");
    if (!std::isfinite(bcf_end_time)) throw py::value_error("bath.bcf_end_time must be finite");
    if (bcf_end_time < 0.0) throw py::value_error("bath.bcf_end_time must be >= 0");

    std::size_t N_H = 0;
    std::size_t N_A = 0;
    std::size_t N_rho0 = 0;
    const auto H_arr = require_c_contig_complex128_2d(H, "H", N_H);
    const auto A_arr = require_c_contig_complex128_2d(A, "A", N_A);
    const auto rho0_arr = require_c_contig_complex128_2d(rho0, "rho0", N_rho0);
    if (N_H != N_A || N_H != N_rho0) throw py::value_error("H, A, and rho0 must have the same (N,N) shape");

    const auto omega_arr = require_c_contig_float64_1d(omega, "bath.omega");
    const auto J_arr = require_c_contig_float64_1d(J, "bath.J");
    const auto spec = spectrum_from_numpy(omega_arr, J_arr, "bath.omega", "bath.J");

    device = to_lower(std::move(device));
    precision = to_lower(std::move(precision));
    const bool want_cuda = (device == "cuda");
    if (!(device == "cpu" || device == "cuda")) {
        throw py::value_error("device must be 'cpu' or 'cuda'");
    }
    if (!(precision == "fp64" || precision == "fp32")) {
        throw py::value_error("precision must be 'fp64' or 'fp32'");
    }
    if (!want_cuda && precision != "fp64") {
        throw py::value_error("precision='fp32' requires device='cuda'");
    }
    if (want_cuda && gpu_id < 0) throw py::value_error("gpu_id must be >= 0");
    const bool want_fp32 = want_cuda && (precision == "fp32");

    const std::size_t N = N_H;

    const std::size_t n_saved = compute_saved_count(n_steps, save_stride);

    py::array_t<double> t_out(static_cast<py::ssize_t>(n_saved));
    py::array_t<cd> rho_out({static_cast<py::ssize_t>(n_saved), static_cast<py::ssize_t>(N),
                             static_cast<py::ssize_t>(N)});
    auto* t_ptr = static_cast<double*>(t_out.mutable_data());
    auto* rho_ptr = static_cast<cd*>(rho_out.mutable_data());

    const Matrix H_mat = eigen_from_c_rowmajor_complex128(H_arr, N);
    const Matrix A_mat = eigen_from_c_rowmajor_complex128(A_arr, N);
    const Matrix rho0_mat = eigen_from_c_rowmajor_complex128(rho0_arr, N);

    const double beta = beta_from_temperature(temperature);

    {
        py::gil_scoped_release release;

        if (want_cuda) {
#ifdef TACO_HAS_CUDA
            if (cudaSetDevice(gpu_id) != cudaSuccess) {
                throw std::runtime_error("Failed to set CUDA device (gpu_id)");
            }
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
                throw std::runtime_error("CUDA backend requested but no CUDA device is available");
            }
#else
            throw std::runtime_error("CUDA backend requested but taco was built without CUDA (TACO_WITH_CUDA=OFF)");
#endif
        }

        // ------------------------------- Build system ----------------------------
        taco::sys::System system;
        system.build(H_mat, {A_mat}, /*freq_tol=*/1e-9);
        const std::size_t dim = system.eig.dim;
        const std::size_t nf = system.fidx.buckets.size();
        if (dim != N) throw std::runtime_error("Internal error: system dimension mismatch");

        // ------------------------------- BCF: C(t) -------------------------------
        const std::size_t N_bcf = steps_from_end_time(dt, bcf_end_time);
        std::vector<double> tgrid_bcf;
        std::vector<cd> C_bcf;
        bcf::bcf_fft_fun(N_bcf, dt, spec, beta, tgrid_bcf, C_bcf);

        // Use a simulation-length C(t) (truncate/pad with zeros after bcf_end_time).
        const std::size_t Nt_sim = n_steps + 1;
        std::vector<cd> Ccorr(Nt_sim, cd{0.0, 0.0});
        const std::size_t copy_len = std::min<std::size_t>(C_bcf.size(), Ccorr.size());
        std::copy(C_bcf.begin(), C_bcf.begin() + static_cast<std::ptrdiff_t>(copy_len), Ccorr.begin());

        // Omega buckets for this system
        std::vector<double> omegas(nf);
        for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;

        // ------------------------------ Gamma series -----------------------------
        Eigen::MatrixXcd gamma_series = taco::gamma::compute_trapz_prefix_multi_matrix(Ccorr, dt, omegas);
        if (gamma_series.rows() != static_cast<Eigen::Index>(Nt_sim)) {
            throw std::runtime_error("Failed to build gamma_series (unexpected length)");
        }

        // ------------------------------- L4 series --------------------------------
        std::vector<Eigen::MatrixXcd> L4_series;
        if (order == 4) {
            taco::Exec exec;
            if (want_cuda) {
                exec.backend = taco::Backend::Cuda;
                exec.gpu_id = gpu_id;
                exec.streams = 1;
                exec.pinned = true;
                exec.cuda_precision = want_fp32 ? CudaPrecision::Fp32 : CudaPrecision::Fp64;
#ifdef TACO_HAS_CUDA
                L4_series =
                    taco::tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, {},
                                                                      taco::tcl4::FCRMethod::Convolution, exec);
#endif
            } else {
                exec.backend = taco::Backend::Omp;
                L4_series = taco::tcl4::build_correction_series(system, gamma_series, dt,
                                                                taco::tcl4::FCRMethod::Convolution, exec);
            }
            if (L4_series.size() != Nt_sim) throw std::runtime_error("Failed to build L4 series");
        }

        // ------------------------------- L(t) builder -----------------------------
        const Eigen::MatrixXcd H0 = system.eig.eps.asDiagonal().toDenseMatrix().cast<cd>();
        const Eigen::MatrixXcd L0 = taco::tcl2::build_unitary_superop(system, H0);

        // Prepare TCL2 spectral kernels container (diagonal-in-channel assumption).
        taco::tcl2::SpectralKernels K2;
        K2.buckets.resize(nf);
        for (std::size_t b = 0; b < nf; ++b) {
            K2.buckets[b].omega = system.fidx.buckets[b].omega;
            K2.buckets[b].Gamma = Eigen::MatrixXcd::Zero(1, 1);
        }

        auto fill_tcl2_kernels = [&](std::size_t time_index) {
            for (std::size_t b = 0; b < nf; ++b) {
                K2.buckets[b].Gamma(0, 0) = gamma_series(static_cast<Eigen::Index>(time_index), static_cast<Eigen::Index>(b));
            }
        };

        auto build_L_at = [&](std::size_t time_index) -> Eigen::MatrixXcd {
            if (order == 0) return L0;
            fill_tcl2_kernels(time_index);
            const taco::tcl2::TCL2Components comps2 = taco::tcl2::build_tcl2_components(system, K2, /*cutoff=*/0.0);
            Eigen::MatrixXcd L = comps2.total();
            if (order == 4) {
                L.noalias() += L4_series[time_index];
            }
            return L;
        };

        // ------------------------------- Propagate -------------------------------
        const Eigen::MatrixXcd rho0_eig = system.eig.rho_to_eigen(rho0_mat);
        Eigen::VectorXcd r = taco::ops::vec(rho0_eig);

        auto write_state = [&](std::size_t out_index, std::size_t step) {
            const double t = static_cast<double>(step) * dt;
            t_ptr[out_index] = t;

            Eigen::MatrixXcd rho_eig = taco::ops::unvec(r, dim);
            rho_eig = taco::ops::hermitize_and_normalize(rho_eig);
            const Eigen::MatrixXcd rho_lab = system.eig.rho_to_lab(rho_eig);

            const std::size_t base = out_index * dim * dim;
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < dim; ++j) {
                    rho_ptr[base + i * dim + j] = rho_lab(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
                }
            }
        };

        std::size_t out_index = 0;
        write_state(out_index++, /*step=*/0);

        if (n_steps > 0) {
            if (want_cuda) {
                #ifdef TACO_HAS_CUDA
                auto cuda_check = [](cudaError_t status, const char* what) {
                    if (status == cudaSuccess) return;
                    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
                };

                const std::size_t D_u = dim * dim;
                if (D_u == 0 || D_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("CUDA RK4: state dimension too large for int indexing");
                }
                if (D_u > std::numeric_limits<std::size_t>::max() / D_u) {
                    throw std::overflow_error("CUDA RK4: dense matrix size overflow");
                }

                if (want_fp32) {
                    propagate_rk4_dense_cuda_fp32(L0, build_L_at, r, dim, n_steps, save_stride, dt, order,
                                                  out_index, write_state);
                } else {
                const int D = static_cast<int>(D_u);
                const std::size_t L_elems = D_u * D_u;
                const std::size_t vbytes = D_u * sizeof(cuDoubleComplex);
                const std::size_t Lbytes = L_elems * sizeof(cuDoubleComplex);
                const cudaStream_t stream = 0;

                // Pack/unpack helpers (Eigen is complex<double>; device uses cuDoubleComplex).
                auto pack_vec = [&](const Eigen::VectorXcd& src, std::vector<cuDoubleComplex>& dst) {
                    dst.resize(D_u);
                    for (std::size_t i = 0; i < D_u; ++i) {
                        const cd z = src(static_cast<Eigen::Index>(i));
                        dst[i] = make_cuDoubleComplex(z.real(), z.imag());
                    }
                };
                auto unpack_vec = [&](const std::vector<cuDoubleComplex>& src, Eigen::VectorXcd& dst) {
                    for (std::size_t i = 0; i < D_u; ++i) {
                        const auto z = src[i];
                        dst(static_cast<Eigen::Index>(i)) = cd(z.x, z.y);
                    }
                };
                auto pack_mat = [&](const Eigen::MatrixXcd& src, std::vector<cuDoubleComplex>& dst) {
                    dst.resize(L_elems);
                    const auto* p = src.data(); // column-major
                    for (std::size_t i = 0; i < L_elems; ++i) dst[i] = make_cuDoubleComplex(p[i].real(), p[i].imag());
                };

                std::vector<cuDoubleComplex> h_r;
                std::vector<cuDoubleComplex> h_L;

                cuDoubleComplex* d_r = nullptr;
                cuda_check(cudaMalloc(&d_r, vbytes), "cudaMalloc(r)");

                auto free_r = [&] { if (d_r) cudaFree(d_r); };

                taco::tcl::Rk4DenseCudaWorkspace ws_cuda;

                try {
                    pack_vec(r, h_r);
                    cuda_check(cudaMemcpy(d_r, h_r.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r H2D)");

                    if (order == 0) {
                        cuDoubleComplex* d_L = nullptr;
                        cuda_check(cudaMalloc(&d_L, Lbytes), "cudaMalloc(L)");
                        auto free_L = [&] {
                            if (d_L) cudaFree(d_L);
                            d_L = nullptr;
                        };

                        try {
                            pack_mat(L0, h_L);
                            cuda_check(cudaMemcpy(d_L, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L H2D)");

                            for (std::size_t step = 0; step < n_steps; ++step) {
                                taco::tcl::rk4_update_cuda(d_L, d_r, D, ws_cuda, dt, stream);

                                const std::size_t step1 = step + 1;
                                if (should_save_step(step1, n_steps, save_stride)) {
                                    cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost),
                                               "cudaMemcpy(r D2H)");
                                    unpack_vec(h_r, r);
                                    write_state(out_index++, step1);
                                }
                            }
                        } catch (...) {
                            free_L();
                            throw;
                        }

                        free_L();
                    } else {
                        cuDoubleComplex* d_L0 = nullptr;
                        cuDoubleComplex* d_L1 = nullptr;
                        cuDoubleComplex* d_Lhalf = nullptr;
                        cuda_check(cudaMalloc(&d_L0, Lbytes), "cudaMalloc(L0)");
                        cuda_check(cudaMalloc(&d_L1, Lbytes), "cudaMalloc(L1)");
                        cuda_check(cudaMalloc(&d_Lhalf, Lbytes), "cudaMalloc(Lhalf)");

                        auto free_mats = [&] {
                            if (d_L0) cudaFree(d_L0);
                            if (d_L1) cudaFree(d_L1);
                            if (d_Lhalf) cudaFree(d_Lhalf);
                        };

                        try {
                            Eigen::MatrixXcd L_cur = build_L_at(0);
                            Eigen::MatrixXcd L_next = build_L_at(1);

                            pack_mat(L_cur, h_L);
                            cuda_check(cudaMemcpy(d_L0, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L0 H2D)");
                            pack_mat(L_next, h_L);
                            cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L1 H2D)");

                            for (std::size_t step = 0; step < n_steps; ++step) {
                                taco::tcl::half_sum_cuda(d_L0, d_L1, d_Lhalf, L_elems, stream);
                                taco::tcl::rk4_update_cuda(d_L0, d_Lhalf, d_L1, d_r, D, ws_cuda, dt, stream);

                                const std::size_t step1 = step + 1;
                                if (should_save_step(step1, n_steps, save_stride)) {
                                    cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r D2H)");
                                    unpack_vec(h_r, r);
                                    write_state(out_index++, step1);
                                }

                                if (step1 < n_steps) {
                                    L_cur = std::move(L_next);
                                    L_next = build_L_at(step + 2);

                                    std::swap(d_L0, d_L1);
                                    pack_mat(L_next, h_L);
                                    cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(Lnext H2D)");
                                }
                            }
                        } catch (...) {
                            free_mats();
                            throw;
                        }

                        free_mats();
                    }
                } catch (...) {
                    free_r();
                    throw;
                }

                free_r();
                }
                #else
                throw std::runtime_error("CUDA backend requested but taco was built without CUDA (TACO_WITH_CUDA=OFF)");
                #endif
            } else {
                taco::tcl::Rk4DenseWorkspace ws;
                ws.resize(static_cast<Eigen::Index>(dim * dim));

                Eigen::MatrixXcd L_cur = build_L_at(0);
                Eigen::MatrixXcd L_next = build_L_at(1);

                for (std::size_t step = 0; step < n_steps; ++step) {
                    const Eigen::MatrixXcd Lhalf = 0.5 * (L_cur + L_next);
                    taco::tcl::rk4_update_serial(L_cur, Lhalf, L_next, r, ws, dt);

                    const std::size_t step1 = step + 1;
                    if (should_save_step(step1, n_steps, save_stride)) {
                        write_state(out_index++, step1);
                    }

                    if (step1 < n_steps) {
                        L_cur = std::move(L_next);
                        L_next = build_L_at(step + 2);
                    }
                }
            }
        }

        if (out_index != n_saved) throw std::runtime_error("Internal error: saved output count mismatch");
    }

    return {t_out, rho_out};
}

std::tuple<py::array, py::array> tcl_simulate_from_bcf(py::handle H,
                                                       py::handle A,
                                                       py::handle rho0,
                                                       double dt,
                                                       std::size_t n_steps,
                                                       std::size_t save_stride,
                                                       py::handle bcf,
                                                       std::string device,
                                                       std::string precision,
                                                       int order,
                                                       int gpu_id) {
    if (!(dt > 0.0)) throw py::value_error("cfg.dt must be > 0");
    if (save_stride == 0) throw py::value_error("cfg.save_stride must be >= 1");
    if (!(order == 0 || order == 2 || order == 4)) throw py::value_error("cfg.order must be 0, 2, or 4");

    std::size_t N_H = 0;
    std::size_t N_A = 0;
    std::size_t N_rho0 = 0;
    const auto H_arr = require_c_contig_complex128_2d(H, "H", N_H);
    const auto A_arr = require_c_contig_complex128_2d(A, "A", N_A);
    const auto rho0_arr = require_c_contig_complex128_2d(rho0, "rho0", N_rho0);
    if (N_H != N_A || N_H != N_rho0) throw py::value_error("H, A, and rho0 must have the same (N,N) shape");

    const auto bcf_arr = require_c_contig_complex128_1d(bcf, "bcf");

    device = to_lower(std::move(device));
    precision = to_lower(std::move(precision));
    const bool want_cuda = (device == "cuda");
    if (!(device == "cpu" || device == "cuda")) {
        throw py::value_error("device must be 'cpu' or 'cuda'");
    }
    if (!(precision == "fp64" || precision == "fp32")) {
        throw py::value_error("precision must be 'fp64' or 'fp32'");
    }
    if (!want_cuda && precision != "fp64") {
        throw py::value_error("precision='fp32' requires device='cuda'");
    }
    if (want_cuda && gpu_id < 0) throw py::value_error("gpu_id must be >= 0");
    const bool want_fp32 = want_cuda && (precision == "fp32");

    const std::size_t N = N_H;
    const std::size_t n_saved = compute_saved_count(n_steps, save_stride);

    py::array_t<double> t_out(static_cast<py::ssize_t>(n_saved));
    py::array_t<cd> rho_out({static_cast<py::ssize_t>(n_saved), static_cast<py::ssize_t>(N),
                             static_cast<py::ssize_t>(N)});
    auto* t_ptr = static_cast<double*>(t_out.mutable_data());
    auto* rho_ptr = static_cast<cd*>(rho_out.mutable_data());

    const Matrix H_mat = eigen_from_c_rowmajor_complex128(H_arr, N);
    const Matrix A_mat = eigen_from_c_rowmajor_complex128(A_arr, N);
    const Matrix rho0_mat = eigen_from_c_rowmajor_complex128(rho0_arr, N);

    {
        py::gil_scoped_release release;

        if (want_cuda) {
#ifdef TACO_HAS_CUDA
            if (cudaSetDevice(gpu_id) != cudaSuccess) {
                throw std::runtime_error("Failed to set CUDA device (gpu_id)");
            }
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
                throw std::runtime_error("CUDA backend requested but no CUDA device is available");
            }
#else
            throw std::runtime_error("CUDA backend requested but taco was built without CUDA (TACO_WITH_CUDA=OFF)");
#endif
        }

        taco::sys::System system;
        system.build(H_mat, {A_mat}, /*freq_tol=*/1e-9);
        const std::size_t dim = system.eig.dim;
        const std::size_t nf = system.fidx.buckets.size();
        if (dim != N) throw std::runtime_error("Internal error: system dimension mismatch");

        // Omega buckets for this system
        std::vector<double> omegas(nf);
        for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;

        // Simulation-length C(t) (pad with zeros beyond provided BCF horizon).
        const std::size_t Nt_sim = n_steps + 1;
        const std::vector<cd> Ccorr = pad_or_truncate_bcf(bcf_arr, Nt_sim);

        // Gamma series
        Eigen::MatrixXcd gamma_series = taco::gamma::compute_trapz_prefix_multi_matrix(Ccorr, dt, omegas);
        if (gamma_series.rows() != static_cast<Eigen::Index>(Nt_sim)) {
            throw std::runtime_error("Failed to build gamma_series (unexpected length)");
        }

        std::vector<Eigen::MatrixXcd> L4_series;
        if (order == 4) {
            taco::Exec exec;
            if (want_cuda) {
                exec.backend = taco::Backend::Cuda;
                exec.gpu_id = gpu_id;
                exec.streams = 1;
                exec.pinned = true;
                exec.cuda_precision = want_fp32 ? CudaPrecision::Fp32 : CudaPrecision::Fp64;
#ifdef TACO_HAS_CUDA
                L4_series =
                    taco::tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, {},
                                                                      taco::tcl4::FCRMethod::Convolution, exec);
#endif
            } else {
                exec.backend = taco::Backend::Omp;
                L4_series = taco::tcl4::build_correction_series(system, gamma_series, dt,
                                                                taco::tcl4::FCRMethod::Convolution, exec);
            }
            if (L4_series.size() != Nt_sim) throw std::runtime_error("Failed to build L4 series");
        }

        const Eigen::MatrixXcd H0 = system.eig.eps.asDiagonal().toDenseMatrix().cast<cd>();
        const Eigen::MatrixXcd L0 = taco::tcl2::build_unitary_superop(system, H0);

        taco::tcl2::SpectralKernels K2;
        K2.buckets.resize(nf);
        for (std::size_t b = 0; b < nf; ++b) {
            K2.buckets[b].omega = system.fidx.buckets[b].omega;
            K2.buckets[b].Gamma = Eigen::MatrixXcd::Zero(1, 1);
        }

        auto fill_tcl2_kernels = [&](std::size_t time_index) {
            for (std::size_t b = 0; b < nf; ++b) {
                K2.buckets[b].Gamma(0, 0) = gamma_series(static_cast<Eigen::Index>(time_index), static_cast<Eigen::Index>(b));
            }
        };

        auto build_L_at = [&](std::size_t time_index) -> Eigen::MatrixXcd {
            if (order == 0) return L0;
            fill_tcl2_kernels(time_index);
            const taco::tcl2::TCL2Components comps2 = taco::tcl2::build_tcl2_components(system, K2, /*cutoff=*/0.0);
            Eigen::MatrixXcd L = comps2.total();
            if (order == 4) {
                L.noalias() += L4_series[time_index];
            }
            return L;
        };

        const Eigen::MatrixXcd rho0_eig = system.eig.rho_to_eigen(rho0_mat);
        Eigen::VectorXcd r = taco::ops::vec(rho0_eig);

        auto write_state = [&](std::size_t out_index, std::size_t step) {
            const double t = static_cast<double>(step) * dt;
            t_ptr[out_index] = t;

            Eigen::MatrixXcd rho_eig = taco::ops::unvec(r, dim);
            rho_eig = taco::ops::hermitize_and_normalize(rho_eig);
            const Eigen::MatrixXcd rho_lab = system.eig.rho_to_lab(rho_eig);

            const std::size_t base = out_index * dim * dim;
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < dim; ++j) {
                    rho_ptr[base + i * dim + j] = rho_lab(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
                }
            }
        };

        std::size_t out_index = 0;
        write_state(out_index++, /*step=*/0);

        if (n_steps > 0) {
            if (want_cuda) {
#ifdef TACO_HAS_CUDA
                auto cuda_check = [](cudaError_t status, const char* what) {
                    if (status == cudaSuccess) return;
                    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
                };

                const std::size_t D_u = dim * dim;
                if (D_u == 0 || D_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("CUDA RK4: state dimension too large for int indexing");
                }
                if (D_u > std::numeric_limits<std::size_t>::max() / D_u) {
                    throw std::overflow_error("CUDA RK4: dense matrix size overflow");
                }

                if (want_fp32) {
                    propagate_rk4_dense_cuda_fp32(L0, build_L_at, r, dim, n_steps, save_stride, dt, order,
                                                  out_index, write_state);
                } else {
                const int D = static_cast<int>(D_u);
                const std::size_t L_elems = D_u * D_u;
                const std::size_t vbytes = D_u * sizeof(cuDoubleComplex);
                const std::size_t Lbytes = L_elems * sizeof(cuDoubleComplex);
                const cudaStream_t stream = 0;

                // Pack/unpack helpers (Eigen is complex<double>; device uses cuDoubleComplex).
                auto pack_vec = [&](const Eigen::VectorXcd& src, std::vector<cuDoubleComplex>& dst) {
                    dst.resize(D_u);
                    for (std::size_t i = 0; i < D_u; ++i) {
                        const cd z = src(static_cast<Eigen::Index>(i));
                        dst[i] = make_cuDoubleComplex(z.real(), z.imag());
                    }
                };
                auto unpack_vec = [&](const std::vector<cuDoubleComplex>& src, Eigen::VectorXcd& dst) {
                    for (std::size_t i = 0; i < D_u; ++i) {
                        const auto z = src[i];
                        dst(static_cast<Eigen::Index>(i)) = cd(z.x, z.y);
                    }
                };
                auto pack_mat = [&](const Eigen::MatrixXcd& src, std::vector<cuDoubleComplex>& dst) {
                    dst.resize(L_elems);
                    const auto* p = src.data(); // column-major
                    for (std::size_t i = 0; i < L_elems; ++i) dst[i] = make_cuDoubleComplex(p[i].real(), p[i].imag());
                };

                std::vector<cuDoubleComplex> h_r;
                std::vector<cuDoubleComplex> h_L;

                cuDoubleComplex* d_r = nullptr;
                cuda_check(cudaMalloc(&d_r, vbytes), "cudaMalloc(r)");

                auto free_r = [&] { if (d_r) cudaFree(d_r); };

                taco::tcl::Rk4DenseCudaWorkspace ws_cuda;

                try {
                    pack_vec(r, h_r);
                    cuda_check(cudaMemcpy(d_r, h_r.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r H2D)");

                    if (order == 0) {
                        cuDoubleComplex* d_L = nullptr;
                        cuda_check(cudaMalloc(&d_L, Lbytes), "cudaMalloc(L)");
                        auto free_L = [&] {
                            if (d_L) cudaFree(d_L);
                            d_L = nullptr;
                        };

                        try {
                            pack_mat(L0, h_L);
                            cuda_check(cudaMemcpy(d_L, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L H2D)");

                            for (std::size_t step = 0; step < n_steps; ++step) {
                                taco::tcl::rk4_update_cuda(d_L, d_r, D, ws_cuda, dt, stream);

                                const std::size_t step1 = step + 1;
                                if (should_save_step(step1, n_steps, save_stride)) {
                                    cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost),
                                               "cudaMemcpy(r D2H)");
                                    unpack_vec(h_r, r);
                                    write_state(out_index++, step1);
                                }
                            }
                        } catch (...) {
                            free_L();
                            throw;
                        }

                        free_L();
                    } else {
                        cuDoubleComplex* d_L0 = nullptr;
                        cuDoubleComplex* d_L1 = nullptr;
                        cuDoubleComplex* d_Lhalf = nullptr;
                        cuda_check(cudaMalloc(&d_L0, Lbytes), "cudaMalloc(L0)");
                        cuda_check(cudaMalloc(&d_L1, Lbytes), "cudaMalloc(L1)");
                        cuda_check(cudaMalloc(&d_Lhalf, Lbytes), "cudaMalloc(Lhalf)");

                        auto free_mats = [&] {
                            if (d_L0) cudaFree(d_L0);
                            if (d_L1) cudaFree(d_L1);
                            if (d_Lhalf) cudaFree(d_Lhalf);
                        };

                        try {
                            Eigen::MatrixXcd L_cur = build_L_at(0);
                            Eigen::MatrixXcd L_next = build_L_at(1);

                            pack_mat(L_cur, h_L);
                            cuda_check(cudaMemcpy(d_L0, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L0 H2D)");
                            pack_mat(L_next, h_L);
                            cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L1 H2D)");

                            for (std::size_t step = 0; step < n_steps; ++step) {
                                taco::tcl::half_sum_cuda(d_L0, d_L1, d_Lhalf, L_elems, stream);
                                taco::tcl::rk4_update_cuda(d_L0, d_Lhalf, d_L1, d_r, D, ws_cuda, dt, stream);

                                const std::size_t step1 = step + 1;
                                if (should_save_step(step1, n_steps, save_stride)) {
                                    cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost),
                                               "cudaMemcpy(r D2H)");
                                    unpack_vec(h_r, r);
                                    write_state(out_index++, step1);
                                }

                                if (step1 < n_steps) {
                                    L_cur = std::move(L_next);
                                    L_next = build_L_at(step + 2);

                                    std::swap(d_L0, d_L1);
                                    pack_mat(L_next, h_L);
                                    cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice),
                                               "cudaMemcpy(Lnext H2D)");
                                }
                            }
                        } catch (...) {
                            free_mats();
                            throw;
                        }

                        free_mats();
                    }
                } catch (...) {
                    free_r();
                    throw;
                }

                free_r();
                }
#else
                throw std::runtime_error("CUDA backend requested but taco was built without CUDA (TACO_WITH_CUDA=OFF)");
#endif
            } else {
                taco::tcl::Rk4DenseWorkspace ws;
                ws.resize(static_cast<Eigen::Index>(dim * dim));

                Eigen::MatrixXcd L_cur = build_L_at(0);
                Eigen::MatrixXcd L_next = build_L_at(1);

                for (std::size_t step = 0; step < n_steps; ++step) {
                    const Eigen::MatrixXcd Lhalf = 0.5 * (L_cur + L_next);
                    taco::tcl::rk4_update_serial(L_cur, Lhalf, L_next, r, ws, dt);

                    const std::size_t step1 = step + 1;
                    if (should_save_step(step1, n_steps, save_stride)) {
                        write_state(out_index++, step1);
                    }

                    if (step1 < n_steps) {
                        L_cur = std::move(L_next);
                        L_next = build_L_at(step + 2);
                    }
                }
            }
        }

        if (out_index != n_saved) throw std::runtime_error("Internal error: saved output count mismatch");
    }

    return {t_out, rho_out};
}

py::dict tcl4_e2e_cuda_compare_spin_boson(std::size_t Nt_samples,
                                         double dt,
                                         double temperature,
                                         double omega_c,
                                         py::object tidx,
                                         int threads,
                                         int gpu_id,
                                         int gpu_warmup,
                                         std::size_t rk4_steps,
                                         int rk4_order,
                                         std::string rk4_method,
                                         std::string precision,
                                         bool check) {
    if (!(Nt_samples > 0)) throw py::value_error("Nt_samples must be > 0");
    if (!(dt > 0.0)) throw py::value_error("dt must be > 0");
    if (!std::isfinite(omega_c) || omega_c <= 0.0) throw py::value_error("omega_c must be finite and > 0");
    if (threads < 0) throw py::value_error("threads must be >= 0");
    if (gpu_id < 0) throw py::value_error("gpu_id must be >= 0");
    if (gpu_warmup < 0) throw py::value_error("gpu_warmup must be >= 0");
    if (!(rk4_order == 0 || rk4_order == 2 || rk4_order == 4)) {
        throw py::value_error("rk4_order must be 0, 2, or 4");
    }

    precision = to_lower(std::move(precision));
    if (!(precision == "fp64" || precision == "f64" || precision == "double" ||
          precision == "fp32" || precision == "f32" || precision == "float")) {
        throw py::value_error("precision must be 'fp64' or 'fp32'");
    }
    const bool want_fp32 = (precision == "fp32" || precision == "f32" || precision == "float");
    const char* gpu_prec_name = want_fp32 ? "fp32" : "fp64";

    rk4_method = to_lower(std::move(rk4_method));

    struct L4Metrics {
        bool has_gpu{false};
        double max_abs{0.0};
        double max_rel{0.0};
        double cpu_fcr_ms{0.0};
        double gpu_fcr_ms{0.0};
        double cpu_total_ms{0.0};
        double cpu_avg_ms{0.0};
        double gpu_total_ms{0.0};
        double gpu_avg_ms{0.0};
    };
    struct Rk4Metrics {
        bool ran{false};
        bool has_gpu{false};
        std::size_t steps{0};
        int order{0};
        std::string method;
        double max_abs_r{0.0};
        double max_abs_rho{0.0};
        double max_rel_rho{0.0};
        double cpu_rk4_ms{0.0};
        double gpu_rk4_ms{0.0};
        double gpu_fcr_ms_rk4{0.0};
    };

    L4Metrics l4;
    Rk4Metrics rk4;
    std::size_t Nt = 0;
    std::vector<std::size_t> tidx_list;

    {
        // Compute the system + gamma series without holding the GIL.
        py::gil_scoped_release release;

        Eigen::MatrixXcd H = 0.5 * taco::ops::sigma_x();
        Eigen::MatrixXcd A = 0.5 * taco::ops::sigma_z();

        taco::sys::System system;
        system.build(H, {A}, 1e-9);

        const std::size_t nf = system.fidx.buckets.size();
        std::vector<double> omegas(nf);
        for (std::size_t b = 0; b < nf; ++b) omegas[b] = system.fidx.buckets[b].omega;

        const double beta = beta_from_temperature(temperature);

        std::vector<double> tgrid;
        std::vector<cd> Ccorr;
        const auto J = [&](double w) { return (w > 0.0) ? (w * std::exp(-w / omega_c)) : 0.0; };
        bcf::bcf_fft_fun(Nt_samples, dt, J, beta, tgrid, Ccorr);

        Eigen::MatrixXcd gamma_series = taco::gamma::compute_trapz_prefix_multi_matrix(Ccorr, dt, omegas);
        Nt = static_cast<std::size_t>(gamma_series.rows());
        if (Nt == 0) {
            throw std::runtime_error("gamma_series is empty");
        }

        // Parse tidx list under the GIL (depends on Nt).
        {
            py::gil_scoped_acquire acquire;
            tidx_list = parse_tidx_from_python(tidx, Nt);
        }
        if (tidx_list.empty()) {
            throw std::runtime_error("tidx selection is empty");
        }

        const double count = static_cast<double>(tidx_list.size());

        taco::Exec exec_cpu;
#ifdef _OPENMP
        exec_cpu.backend = taco::Backend::Omp;
        exec_cpu.threads = threads;
#else
        exec_cpu.backend = taco::Backend::Serial;
        (void)threads;
#endif

        // CPU reference: F/C/R kernels once, then build L4 at each tidx.
        std::vector<Eigen::MatrixXcd> L4_cpu_list;
        L4_cpu_list.reserve(tidx_list.size());

        const auto t_cpu_kernel_start = std::chrono::high_resolution_clock::now();
        const auto kernels = taco::tcl4::compute_triple_kernels(system, gamma_series, dt, /*nmax*/ 2,
                                                                taco::tcl4::FCRMethod::Convolution, exec_cpu);
        const taco::tcl4::Tcl4Map map = taco::tcl4::build_map(system, /*time_grid*/ {});
        const auto t_cpu_kernel_end = std::chrono::high_resolution_clock::now();
        l4.cpu_fcr_ms =
            std::chrono::duration<double, std::milli>(t_cpu_kernel_end - t_cpu_kernel_start).count();

        double cpu_total_ms = 0.0;
        for (std::size_t tidx_i : tidx_list) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto mikx = taco::tcl4::build_mikx(map, kernels, tidx_i);
            const Eigen::MatrixXcd GW = taco::tcl4::assemble_liouvillian(mikx, system.A_eig);
            const Eigen::MatrixXcd L4_cpu = taco::tcl4::gw_to_liouvillian(GW, system.eig.dim);
            const auto t1 = std::chrono::high_resolution_clock::now();
            cpu_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            L4_cpu_list.push_back(L4_cpu);
        }
        l4.cpu_total_ms = l4.cpu_fcr_ms + cpu_total_ms;
        l4.cpu_avg_ms = l4.cpu_total_ms / count;

#ifdef TACO_HAS_CUDA
        if (cuda_is_available()) {
            taco::Exec exec_gpu;
            exec_gpu.backend = taco::Backend::Cuda;
            exec_gpu.gpu_id = gpu_id;
            exec_gpu.cuda_precision = want_fp32 ? taco::CudaPrecision::Fp32 : taco::CudaPrecision::Fp64;

            for (int w = 0; w < gpu_warmup; ++w) {
                (void)taco::tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, tidx_list,
                                                                        taco::tcl4::FCRMethod::Convolution, exec_gpu,
                                                                        nullptr);
            }

            const auto t_gpu_start = std::chrono::high_resolution_clock::now();
            const auto L4_gpu_list =
                taco::tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, tidx_list,
                                                                  taco::tcl4::FCRMethod::Convolution, exec_gpu,
                                                                  &l4.gpu_fcr_ms);
            const auto t_gpu_end = std::chrono::high_resolution_clock::now();
            l4.gpu_total_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();
            l4.gpu_avg_ms = l4.gpu_total_ms / count;

            l4.has_gpu = true;
            for (std::size_t i = 0; i < tidx_list.size(); ++i) {
                const Eigen::MatrixXcd& L4_cpu = L4_cpu_list[i];
                const Eigen::MatrixXcd& L4_gpu = L4_gpu_list[i];
                const double err = max_abs_diff(L4_cpu, L4_gpu);
                const double ref = std::max(1.0, L4_cpu.cwiseAbs().maxCoeff());
                const double rel = err / ref;
                l4.max_abs = std::max(l4.max_abs, err);
                l4.max_rel = std::max(l4.max_rel, rel);
            }

            if (check) {
                const double tol = want_fp32 ? 1e-4 : 1e-8;
                if (l4.max_abs > tol && l4.max_rel > tol) {
                    throw std::runtime_error("L4 mismatch above tolerance");
                }
            }

            if (rk4_steps > 0 && Nt >= 2) {
                rk4.ran = true;
                rk4.has_gpu = true;
                rk4.steps = std::min(rk4_steps, Nt - 1);
                rk4.order = rk4_order;
                rk4.method = rk4_method;

                const std::size_t dim = system.eig.dim;
                const std::size_t D_u = dim * dim;
                if (D_u == 0 || D_u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("RK4 compare: state dimension too large for int indexing");
                }
                if (D_u > std::numeric_limits<std::size_t>::max() / D_u) {
                    throw std::overflow_error("RK4 compare: dense matrix size overflow");
                }

                std::vector<std::size_t> rk_tidx(rk4.steps + 1);
                for (std::size_t k = 0; k <= rk4.steps; ++k) rk_tidx[k] = k;

                std::vector<Eigen::MatrixXcd> L4_cpu_series;
                std::vector<Eigen::MatrixXcd> L4_gpu_series;

                if (rk4_order == 4) {
                    L4_cpu_series.reserve(rk_tidx.size());
                    for (std::size_t tidx_i : rk_tidx) {
                        auto mikx = taco::tcl4::build_mikx(map, kernels, tidx_i);
                        const Eigen::MatrixXcd GW = taco::tcl4::assemble_liouvillian(mikx, system.A_eig);
                        L4_cpu_series.push_back(taco::tcl4::gw_to_liouvillian(GW, system.eig.dim));
                    }
                    for (int w = 0; w < gpu_warmup; ++w) {
                        (void)taco::tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, rk_tidx,
                                                                                taco::tcl4::FCRMethod::Convolution,
                                                                                exec_gpu, nullptr);
                    }
                    L4_gpu_series =
                        taco::tcl4::build_TCL4_generator_cuda_fused_batch(system, gamma_series, dt, rk_tidx,
                                                                          taco::tcl4::FCRMethod::Convolution, exec_gpu,
                                                                          &rk4.gpu_fcr_ms_rk4);
                } else {
                    L4_cpu_series.assign(rk_tidx.size(), Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(D_u),
                                                                                static_cast<Eigen::Index>(D_u)));
                    L4_gpu_series = L4_cpu_series;
                }

                const Eigen::MatrixXcd H0 = system.eig.eps.asDiagonal().toDenseMatrix().cast<cd>();
                const Eigen::MatrixXcd L0 = taco::tcl2::build_unitary_superop(system, H0);

                taco::tcl2::SpectralKernels K2;
                K2.buckets.resize(nf);
                for (std::size_t b = 0; b < nf; ++b) {
                    K2.buckets[b].omega = system.fidx.buckets[b].omega;
                    K2.buckets[b].Gamma = Eigen::MatrixXcd::Zero(1, 1);
                }

                auto fill_tcl2_kernels = [&](std::size_t time_index) {
                    for (std::size_t b = 0; b < nf; ++b) {
                        K2.buckets[b].Gamma(0, 0) =
                            gamma_series(static_cast<Eigen::Index>(time_index), static_cast<Eigen::Index>(b));
                    }
                };

                auto build_L_at = [&](std::size_t time_index, const std::vector<Eigen::MatrixXcd>& L4_series) -> Eigen::MatrixXcd {
                    if (rk4_order == 0) return L0;
                    fill_tcl2_kernels(time_index);
                    const taco::tcl2::TCL2Components comps2 = taco::tcl2::build_tcl2_components(system, K2, /*cutoff=*/0.0);
                    Eigen::MatrixXcd L = comps2.total();
                    if (rk4_order == 4) {
                        L.noalias() += L4_series[time_index];
                    }
                    return L;
                };

                Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
                rho0(0, 0) = 1.0;
                const Eigen::MatrixXcd rho0_eig = system.eig.rho_to_eigen(rho0);

                Eigen::VectorXcd r_cpu = taco::ops::vec(rho0_eig);
                Eigen::VectorXcd r_gpu = r_cpu;

                const auto cpu_start = std::chrono::high_resolution_clock::now();
                {
                    taco::tcl::Rk4DenseWorkspace ws;
                    ws.resize(static_cast<Eigen::Index>(D_u));

                    Eigen::MatrixXcd L_cur = build_L_at(0, L4_cpu_series);
                    Eigen::MatrixXcd L_next = build_L_at(1, L4_cpu_series);
                    for (std::size_t step = 0; step < rk4.steps; ++step) {
                        const Eigen::MatrixXcd Lhalf = 0.5 * (L_cur + L_next);
                        taco::tcl::rk4_update_serial(L_cur, Lhalf, L_next, r_cpu, ws, dt);

                        const std::size_t step1 = step + 1;
                        if (step1 < rk4.steps) {
                            L_cur = std::move(L_next);
                            L_next = build_L_at(step + 2, L4_cpu_series);
                        }
                    }
                }
                const auto cpu_end = std::chrono::high_resolution_clock::now();
                rk4.cpu_rk4_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

                taco::tcl::Rk4DenseCudaMethod rk4_cuda_method = taco::tcl::Rk4DenseCudaMethod::WarpKernel;
                if (rk4_method == "warp" || rk4_method == "kernel") {
                    rk4_cuda_method = taco::tcl::Rk4DenseCudaMethod::WarpKernel;
                } else if (rk4_method == "cublas" || rk4_method == "cublasgemv") {
                    rk4_cuda_method = taco::tcl::Rk4DenseCudaMethod::CublasGemv;
                } else {
                    throw std::invalid_argument("rk4_method must be 'warp' or 'cublas'");
                }

                const auto gpu_start_rk4 = std::chrono::high_resolution_clock::now();
                {
                    cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice(rk4)");

                    const int D = static_cast<int>(D_u);
                    const std::size_t L_elems = D_u * D_u;
                    const cudaStream_t stream = 0;

                    if (want_fp32) {
                        const float dt_f32 = static_cast<float>(dt);
                        const std::size_t vbytes = D_u * sizeof(cuFloatComplex);
                        const std::size_t Lbytes = L_elems * sizeof(cuFloatComplex);

                        auto pack_vec = [&](const Eigen::VectorXcd& src, std::vector<cuFloatComplex>& dst) {
                            dst.resize(D_u);
                            for (std::size_t i = 0; i < D_u; ++i) {
                                const cd z = src(static_cast<Eigen::Index>(i));
                                dst[i] = make_cuFloatComplex(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                            }
                        };
                        auto unpack_vec = [&](const std::vector<cuFloatComplex>& src, Eigen::VectorXcd& dst) {
                            for (std::size_t i = 0; i < D_u; ++i) {
                                const auto z = src[i];
                                dst(static_cast<Eigen::Index>(i)) = cd(static_cast<double>(z.x), static_cast<double>(z.y));
                            }
                        };
                        auto pack_mat = [&](const Eigen::MatrixXcd& src, std::vector<cuFloatComplex>& dst) {
                            dst.resize(L_elems);
                            const auto* p = src.data(); // column-major
                            for (std::size_t i = 0; i < L_elems; ++i) {
                                dst[i] = make_cuFloatComplex(static_cast<float>(p[i].real()), static_cast<float>(p[i].imag()));
                            }
                        };

                        std::vector<cuFloatComplex> h_r;
                        std::vector<cuFloatComplex> h_L;

                        cuFloatComplex* d_r = nullptr;
                        cuda_check(cudaMalloc(&d_r, vbytes), "cudaMalloc(r_f32)");
                        auto free_r = [&] {
                            if (d_r) cudaFree(d_r);
                            d_r = nullptr;
                        };

                        taco::tcl::Rk4DenseCudaWorkspaceF32 ws_cuda;

                        try {
                            pack_vec(r_gpu, h_r);
                            cuda_check(cudaMemcpy(d_r, h_r.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r_f32 H2D)");

                            if (rk4_order == 0) {
                                Eigen::MatrixXcd Lconst = build_L_at(0, L4_gpu_series);
                                cuFloatComplex* d_L = nullptr;
                                cuda_check(cudaMalloc(&d_L, Lbytes), "cudaMalloc(L_f32)");
                                auto free_L = [&] {
                                    if (d_L) cudaFree(d_L);
                                    d_L = nullptr;
                                };

                                try {
                                    pack_mat(Lconst, h_L);
                                    cuda_check(cudaMemcpy(d_L, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L_f32 H2D)");
                                    for (std::size_t step = 0; step < rk4.steps; ++step) {
                                        taco::tcl::rk4_update_cuda_f32(d_L, d_r, D, ws_cuda, dt_f32, stream, rk4_cuda_method);
                                    }
                                } catch (...) {
                                    free_L();
                                    throw;
                                }
                                free_L();
                            } else {
                                cuFloatComplex* d_L0 = nullptr;
                                cuFloatComplex* d_L1 = nullptr;
                                cuFloatComplex* d_Lhalf = nullptr;
                                cuda_check(cudaMalloc(&d_L0, Lbytes), "cudaMalloc(L0_f32)");
                                cuda_check(cudaMalloc(&d_L1, Lbytes), "cudaMalloc(L1_f32)");
                                cuda_check(cudaMalloc(&d_Lhalf, Lbytes), "cudaMalloc(Lhalf_f32)");

                                auto free_mats = [&] {
                                    if (d_L0) cudaFree(d_L0);
                                    if (d_L1) cudaFree(d_L1);
                                    if (d_Lhalf) cudaFree(d_Lhalf);
                                };

                                try {
                                    Eigen::MatrixXcd L_cur = build_L_at(0, L4_gpu_series);
                                    Eigen::MatrixXcd L_next = build_L_at(1, L4_gpu_series);

                                    pack_mat(L_cur, h_L);
                                    cuda_check(cudaMemcpy(d_L0, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L0_f32 H2D)");
                                    pack_mat(L_next, h_L);
                                    cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L1_f32 H2D)");

                                    for (std::size_t step = 0; step < rk4.steps; ++step) {
                                        taco::tcl::half_sum_cuda_f32(d_L0, d_L1, d_Lhalf, L_elems, stream);
                                        taco::tcl::rk4_update_cuda_f32(d_L0, d_Lhalf, d_L1, d_r, D, ws_cuda, dt_f32, stream, rk4_cuda_method);

                                        const std::size_t step1 = step + 1;
                                        if (step1 < rk4.steps) {
                                            L_cur = std::move(L_next);
                                            L_next = build_L_at(step + 2, L4_gpu_series);

                                            std::swap(d_L0, d_L1);
                                            pack_mat(L_next, h_L);
                                            cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice),
                                                       "cudaMemcpy(Lnext_f32 H2D)");
                                        }
                                    }
                                } catch (...) {
                                    free_mats();
                                    throw;
                                }

                                free_mats();
                            }

                            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(rk4_f32)");
                            cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r_f32 D2H)");
                            unpack_vec(h_r, r_gpu);
                        } catch (...) {
                            free_r();
                            throw;
                        }

                        free_r();
                    } else {
                        const std::size_t vbytes = D_u * sizeof(cuDoubleComplex);
                        const std::size_t Lbytes = L_elems * sizeof(cuDoubleComplex);

                        auto pack_vec = [&](const Eigen::VectorXcd& src, std::vector<cuDoubleComplex>& dst) {
                            dst.resize(D_u);
                            for (std::size_t i = 0; i < D_u; ++i) {
                                const cd z = src(static_cast<Eigen::Index>(i));
                                dst[i] = make_cuDoubleComplex(z.real(), z.imag());
                            }
                        };
                        auto unpack_vec = [&](const std::vector<cuDoubleComplex>& src, Eigen::VectorXcd& dst) {
                            for (std::size_t i = 0; i < D_u; ++i) {
                                const auto z = src[i];
                                dst(static_cast<Eigen::Index>(i)) = cd(z.x, z.y);
                            }
                        };
                        auto pack_mat = [&](const Eigen::MatrixXcd& src, std::vector<cuDoubleComplex>& dst) {
                            dst.resize(L_elems);
                            const auto* p = src.data(); // column-major
                            for (std::size_t i = 0; i < L_elems; ++i) {
                                dst[i] = make_cuDoubleComplex(p[i].real(), p[i].imag());
                            }
                        };

                        std::vector<cuDoubleComplex> h_r;
                        std::vector<cuDoubleComplex> h_L;

                        cuDoubleComplex* d_r = nullptr;
                        cuda_check(cudaMalloc(&d_r, vbytes), "cudaMalloc(r)");
                        auto free_r = [&] {
                            if (d_r) cudaFree(d_r);
                            d_r = nullptr;
                        };

                        taco::tcl::Rk4DenseCudaWorkspace ws_cuda;

                        try {
                            pack_vec(r_gpu, h_r);
                            cuda_check(cudaMemcpy(d_r, h_r.data(), vbytes, cudaMemcpyHostToDevice), "cudaMemcpy(r H2D)");

                            if (rk4_order == 0) {
                                Eigen::MatrixXcd Lconst = build_L_at(0, L4_gpu_series);
                                cuDoubleComplex* d_L = nullptr;
                                cuda_check(cudaMalloc(&d_L, Lbytes), "cudaMalloc(L)");
                                auto free_L = [&] {
                                    if (d_L) cudaFree(d_L);
                                    d_L = nullptr;
                                };

                                try {
                                    pack_mat(Lconst, h_L);
                                    cuda_check(cudaMemcpy(d_L, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L H2D)");
                                    for (std::size_t step = 0; step < rk4.steps; ++step) {
                                        taco::tcl::rk4_update_cuda(d_L, d_r, D, ws_cuda, dt, stream, rk4_cuda_method);
                                    }
                                } catch (...) {
                                    free_L();
                                    throw;
                                }
                                free_L();
                            } else {
                                cuDoubleComplex* d_L0 = nullptr;
                                cuDoubleComplex* d_L1 = nullptr;
                                cuDoubleComplex* d_Lhalf = nullptr;
                                cuda_check(cudaMalloc(&d_L0, Lbytes), "cudaMalloc(L0)");
                                cuda_check(cudaMalloc(&d_L1, Lbytes), "cudaMalloc(L1)");
                                cuda_check(cudaMalloc(&d_Lhalf, Lbytes), "cudaMalloc(Lhalf)");

                                auto free_mats = [&] {
                                    if (d_L0) cudaFree(d_L0);
                                    if (d_L1) cudaFree(d_L1);
                                    if (d_Lhalf) cudaFree(d_Lhalf);
                                };

                                try {
                                    Eigen::MatrixXcd L_cur = build_L_at(0, L4_gpu_series);
                                    Eigen::MatrixXcd L_next = build_L_at(1, L4_gpu_series);

                                    pack_mat(L_cur, h_L);
                                    cuda_check(cudaMemcpy(d_L0, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L0 H2D)");
                                    pack_mat(L_next, h_L);
                                    cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice), "cudaMemcpy(L1 H2D)");

                                    for (std::size_t step = 0; step < rk4.steps; ++step) {
                                        taco::tcl::half_sum_cuda(d_L0, d_L1, d_Lhalf, L_elems, stream);
                                        taco::tcl::rk4_update_cuda(d_L0, d_Lhalf, d_L1, d_r, D, ws_cuda, dt, stream, rk4_cuda_method);

                                        const std::size_t step1 = step + 1;
                                        if (step1 < rk4.steps) {
                                            L_cur = std::move(L_next);
                                            L_next = build_L_at(step + 2, L4_gpu_series);

                                            std::swap(d_L0, d_L1);
                                            pack_mat(L_next, h_L);
                                            cuda_check(cudaMemcpy(d_L1, h_L.data(), Lbytes, cudaMemcpyHostToDevice),
                                                       "cudaMemcpy(Lnext H2D)");
                                        }
                                    }
                                } catch (...) {
                                    free_mats();
                                    throw;
                                }

                                free_mats();
                            }

                            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(rk4)");
                            cuda_check(cudaMemcpy(h_r.data(), d_r, vbytes, cudaMemcpyDeviceToHost), "cudaMemcpy(r D2H)");
                            unpack_vec(h_r, r_gpu);
                        } catch (...) {
                            free_r();
                            throw;
                        }

                        free_r();
                    }
                }
                const auto gpu_end_rk4 = std::chrono::high_resolution_clock::now();
                rk4.gpu_rk4_ms = std::chrono::duration<double, std::milli>(gpu_end_rk4 - gpu_start_rk4).count();

                const auto rho_from_r = [&](const Eigen::VectorXcd& r) {
                    Eigen::MatrixXcd rho_eig = taco::ops::unvec(r, static_cast<std::size_t>(dim));
                    rho_eig = taco::ops::hermitize_and_normalize(rho_eig);
                    return system.eig.rho_to_lab(rho_eig);
                };

                const Eigen::MatrixXcd rho_cpu = rho_from_r(r_cpu);
                const Eigen::MatrixXcd rho_gpu = rho_from_r(r_gpu);

                rk4.max_abs_r = max_abs_diff(r_cpu, r_gpu);
                rk4.max_abs_rho = max_abs_diff(rho_cpu, rho_gpu);
                const double rho_ref = std::max(1.0, rho_cpu.cwiseAbs().maxCoeff());
                rk4.max_rel_rho = rk4.max_abs_rho / rho_ref;

                if (check) {
                    const double rk4_tol = want_fp32 ? 1e-4 : 1e-5;
                    if (rk4.max_abs_rho > rk4_tol && rk4.max_rel_rho > rk4_tol) {
                        throw std::runtime_error("RK4 propagation mismatch above tolerance");
                    }
                }
            }
        }
#endif // TACO_HAS_CUDA
    }

    py::dict out;
    out["Nt_samples"] = Nt_samples;
    out["dt"] = dt;
    out["temperature"] = temperature;
    out["omega_c"] = omega_c;
    out["precision"] = gpu_prec_name;

#ifdef TACO_HAS_CUDA
    out["cuda_enabled"] = true;
    out["cuda_available"] = cuda_is_available();
#else
    out["cuda_enabled"] = false;
    out["cuda_available"] = false;
#endif

    py::list tidx_py;
    for (std::size_t v : tidx_list) tidx_py.append(py::int_(static_cast<unsigned long long>(v)));
    out["tidx"] = tidx_py;

    py::dict l4_py;
    auto maybe_float = [](bool enabled, double value) -> py::object {
        if (enabled) return py::float_(value);
        return py::none();
    };
    l4_py["has_gpu"] = l4.has_gpu;
    l4_py["max_abs"] = maybe_float(l4.has_gpu, l4.max_abs);
    l4_py["max_rel"] = maybe_float(l4.has_gpu, l4.max_rel);
    l4_py["cpu_fcr_ms"] = l4.cpu_fcr_ms;
    l4_py["gpu_fcr_ms"] = maybe_float(l4.has_gpu, l4.gpu_fcr_ms);
    l4_py["cpu_total_ms"] = l4.cpu_total_ms;
    l4_py["cpu_avg_ms"] = l4.cpu_avg_ms;
    l4_py["gpu_total_ms"] = maybe_float(l4.has_gpu, l4.gpu_total_ms);
    l4_py["gpu_avg_ms"] = maybe_float(l4.has_gpu, l4.gpu_avg_ms);
    out["l4"] = l4_py;

    if (!rk4.ran) {
        out["rk4"] = py::none();
    } else {
        py::dict rk4_py;
        rk4_py["has_gpu"] = rk4.has_gpu;
        rk4_py["steps"] = rk4.steps;
        rk4_py["order"] = rk4.order;
        rk4_py["method"] = rk4.method;
        rk4_py["max_abs_r"] = maybe_float(rk4.has_gpu, rk4.max_abs_r);
        rk4_py["max_abs_rho"] = maybe_float(rk4.has_gpu, rk4.max_abs_rho);
        rk4_py["max_rel_rho"] = maybe_float(rk4.has_gpu, rk4.max_rel_rho);
        rk4_py["cpu_rk4_ms"] = rk4.cpu_rk4_ms;
        rk4_py["gpu_rk4_ms"] = maybe_float(rk4.has_gpu, rk4.gpu_rk4_ms);
        rk4_py["gpu_fcr_ms_rk4"] = maybe_float(rk4.has_gpu, rk4.gpu_fcr_ms_rk4);
        out["rk4"] = rk4_py;
    }

    return out;
}

} // namespace taco::python

PYBIND11_MODULE(_taco, m) {
    m.doc() = "TACO native bindings";

    m.def("version", &taco::version);

    m.def("build_info", &taco::python::build_info);

    m.def("cuda_is_available", &taco::python::cuda_is_available);
    m.def("cuda_device_count", &taco::python::cuda_device_count);

    m.def("tcl_precompute_bcf", &taco::python::tcl_precompute_bcf,
          py::arg("temperature"),
          py::arg("dt"),
          py::arg("bcf_end_time"),
          py::arg("omega"),
          py::arg("J"));

    m.def("tcl_simulate", &taco::python::tcl_simulate,
          py::arg("H"),
          py::arg("A"),
          py::arg("rho0"),
          py::arg("temperature"),
          py::arg("dt"),
          py::arg("n_steps"),
          py::arg("save_stride"),
          py::arg("bcf_end_time"),
          py::arg("omega"),
          py::arg("J"),
          py::arg("device") = "cpu",
          py::arg("precision") = "fp64",
          py::arg("order") = 4,
          py::arg("gpu_id") = 0);

    m.def("tcl_simulate_from_bcf", &taco::python::tcl_simulate_from_bcf,
          py::arg("H"),
          py::arg("A"),
          py::arg("rho0"),
          py::arg("dt"),
          py::arg("n_steps"),
          py::arg("save_stride"),
          py::arg("bcf"),
          py::arg("device") = "cpu",
          py::arg("precision") = "fp64",
          py::arg("order") = 4,
          py::arg("gpu_id") = 0);

    m.def("tcl4_e2e_cuda_compare_spin_boson", &taco::python::tcl4_e2e_cuda_compare_spin_boson,
          py::arg("Nt_samples") = 100000,
          py::arg("dt") = 0.000625,
          py::arg("temperature") = 2.0,
          py::arg("omega_c") = 10.0,
          py::arg("tidx") = py::none(),
          py::arg("threads") = 0,
          py::arg("gpu_id") = 0,
          py::arg("gpu_warmup") = 1,
          py::arg("rk4_steps") = 50,
          py::arg("rk4_order") = 4,
          py::arg("rk4_method") = "warp",
          py::arg("precision") = "fp64",
          py::arg("check") = true);
}
