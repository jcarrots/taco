#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cctype>
#include <complex>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hdf5.h"

#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_kernels.hpp"
#include "taco/tcl4_mikx.hpp"

namespace {

using cd = std::complex<double>;

// ----------------------------- HDF5 helpers -----------------------------

struct H5File {
    hid_t id{-1};
    explicit H5File(const std::string& path) {
        id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (id < 0) throw std::runtime_error("Failed to open HDF5 file: " + path);
    }
    ~H5File() {
        if (id >= 0) H5Fclose(id);
    }
    H5File(const H5File&) = delete;
    H5File& operator=(const H5File&) = delete;
};

struct H5Dataset {
    hid_t id{-1};
    explicit H5Dataset(hid_t id_) : id(id_) {}
    ~H5Dataset() {
        if (id >= 0) H5Dclose(id);
    }
    H5Dataset(const H5Dataset&) = delete;
    H5Dataset& operator=(const H5Dataset&) = delete;
};

struct H5Space {
    hid_t id{-1};
    explicit H5Space(hid_t id_) : id(id_) {}
    ~H5Space() {
        if (id >= 0) H5Sclose(id);
    }
    H5Space(const H5Space&) = delete;
    H5Space& operator=(const H5Space&) = delete;
};

bool dataset_exists(hid_t file_id, const std::string& path) {
    return H5Lexists(file_id, path.c_str(), H5P_DEFAULT) > 0;
}

std::vector<hsize_t> get_dims(hid_t dset_id) {
    H5Space space(H5Dget_space(dset_id));
    if (space.id < 0) throw std::runtime_error("Failed to get dataspace");
    const int nd = H5Sget_simple_extent_ndims(space.id);
    if (nd < 0) throw std::runtime_error("Failed to get dataspace rank");
    std::vector<hsize_t> dims(static_cast<std::size_t>(nd), 0);
    if (nd > 0) {
        if (H5Sget_simple_extent_dims(space.id, dims.data(), nullptr) < 0) {
            throw std::runtime_error("Failed to read dataspace dims");
        }
    }
    return dims;
}

std::vector<hsize_t> get_dataset_dims(hid_t file_id, const std::string& path) {
    H5Dataset dset(H5Dopen2(file_id, path.c_str(), H5P_DEFAULT));
    if (dset.id < 0) throw std::runtime_error("Failed to open dataset: " + path);
    return get_dims(dset.id);
}

std::string dims_to_string(const std::vector<hsize_t>& dims) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) oss << ",";
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

std::size_t numel(const std::vector<hsize_t>& dims) {
    std::size_t out = 1;
    for (hsize_t d : dims) out *= static_cast<std::size_t>(d);
    return out;
}

template <class T>
std::vector<T> read_array(hid_t file_id,
                          const std::string& path,
                          hid_t mem_type,
                          std::vector<hsize_t>* dims_out = nullptr) {
    H5Dataset dset(H5Dopen2(file_id, path.c_str(), H5P_DEFAULT));
    if (dset.id < 0) throw std::runtime_error("Failed to open dataset: " + path);
    const auto dims = get_dims(dset.id);
    std::vector<T> out(numel(dims));
    if (!out.empty()) {
        if (H5Dread(dset.id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
            throw std::runtime_error("Failed to read dataset: " + path);
        }
    }
    if (dims_out) *dims_out = dims;
    return out;
}

std::vector<cd> read_complex(hid_t file_id,
                             const std::string& base,
                             std::vector<hsize_t>* dims_out = nullptr) {
    std::vector<hsize_t> dims_re;
    std::vector<hsize_t> dims_im;
    auto re = read_array<double>(file_id, base + "/re", H5T_NATIVE_DOUBLE, &dims_re);
    auto im = read_array<double>(file_id, base + "/im", H5T_NATIVE_DOUBLE, &dims_im);
    if (dims_re != dims_im || re.size() != im.size()) {
        throw std::runtime_error("Complex dataset dims mismatch at: " + base);
    }
    std::vector<cd> out(re.size());
    for (std::size_t i = 0; i < re.size(); ++i) out[i] = cd{re[i], im[i]};
    if (dims_out) *dims_out = dims_re;
    return out;
}

double read_scalar_double(hid_t file_id, const std::string& path) {
    auto v = read_array<double>(file_id, path, H5T_NATIVE_DOUBLE);
    if (v.size() != 1) throw std::runtime_error("Expected scalar at: " + path);
    return v[0];
}

struct DatasetInfo {
    std::string path;
    std::vector<hsize_t> dims;
};

herr_t list_cb(hid_t /*obj*/, const char* name, const H5O_info2_t* info, void* op_data) {
    if (!op_data || !name || !info) return 0;
    auto* out = static_cast<std::vector<std::string>*>(op_data);
    if (info->type == H5O_TYPE_DATASET) out->emplace_back(std::string("/") + name);
    return 0;
}

std::vector<DatasetInfo> list_datasets(hid_t file_id) {
    std::vector<std::string> names;
    if (H5Ovisit3(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, list_cb, &names, H5O_INFO_BASIC) < 0) {
        throw std::runtime_error("Failed to list HDF5 datasets");
    }
    std::sort(names.begin(), names.end());
    std::vector<DatasetInfo> out;
    out.reserve(names.size());
    for (const auto& path : names) {
        out.push_back(DatasetInfo{path, get_dataset_dims(file_id, path)});
    }
    return out;
}

// MATLAB writes multi-d arrays in column-major; interpret raw buffers accordingly.
cd at_colmajor_2d(const std::vector<cd>& data,
                  const std::vector<hsize_t>& dims,
                  std::size_t i0,
                  std::size_t i1) {
    return data[i0 + static_cast<std::size_t>(dims[0]) * i1];
}

cd at_colmajor_4d(const std::vector<cd>& data,
                  const std::vector<hsize_t>& dims,
                  std::size_t i0,
                  std::size_t i1,
                  std::size_t i2,
                  std::size_t i3) {
    const std::size_t d0 = static_cast<std::size_t>(dims[0]);
    const std::size_t d1 = static_cast<std::size_t>(dims[1]);
    const std::size_t d2 = static_cast<std::size_t>(dims[2]);
    return data[i0 + d0 * (i1 + d1 * (i2 + d2 * i3))];
}

std::vector<hsize_t> reverse_dims(const std::vector<hsize_t>& dims) {
    std::vector<hsize_t> out = dims;
    std::reverse(out.begin(), out.end());
    return out;
}

// ----------------------------- Data wrappers -----------------------------

struct FlatSeries {
    std::vector<cd> data;
    std::vector<hsize_t> dims_file; // file dims (rank-2)
    std::vector<hsize_t> dims_use;  // always [Nt, flat_len] in column-major
    std::size_t flat_len{0};
    std::size_t Nt{0};

    cd at(std::size_t tidx, std::size_t flat_idx) const {
        return at_colmajor_2d(data, dims_use, tidx, flat_idx);
    }
};

FlatSeries load_flat_series(hid_t file_id, const std::string& base, std::size_t flat_len) {
    std::vector<hsize_t> dims;
    auto data = read_complex(file_id, base, &dims);
    if (dims.size() != 2) {
        throw std::runtime_error(base + " must be rank-2, got dims=" + dims_to_string(dims));
    }
    const std::size_t d0 = static_cast<std::size_t>(dims[0]);
    const std::size_t d1 = static_cast<std::size_t>(dims[1]);
    FlatSeries out;
    out.data = std::move(data);
    out.dims_file = dims;
    out.flat_len = flat_len;
    if (d1 == flat_len) {
        out.Nt = d0;
        out.dims_use = dims; // [Nt, flat_len]
        return out;
    }
    if (d0 == flat_len) {
        out.Nt = d1;
        out.dims_use = {dims[1], dims[0]}; // interpret file as [Nt, flat_len]
        return out;
    }
    throw std::runtime_error(base + " dims do not match flat_len=" + std::to_string(flat_len) +
                             ": " + dims_to_string(dims));
}

struct KernelSeries {
    std::vector<cd> data;
    std::vector<hsize_t> dims_use; // always [Nt, nf, nf, nf]
    std::size_t nf{0};
    std::size_t Nt{0};

    cd at(std::size_t tidx, std::size_t i, std::size_t j, std::size_t k) const {
        return at_colmajor_4d(data, dims_use, tidx, i, j, k);
    }
};

KernelSeries load_kernel_series(hid_t file_id, const std::string& base, std::size_t nf) {
    std::vector<hsize_t> dims;
    auto data = read_complex(file_id, base, &dims);
    if (dims.size() != 4) {
        throw std::runtime_error(base + " must be rank-4, got dims=" + dims_to_string(dims));
    }
    KernelSeries out;
    out.data = std::move(data);
    out.nf = nf;

    const std::size_t d0 = static_cast<std::size_t>(dims[0]);
    const std::size_t d1 = static_cast<std::size_t>(dims[1]);
    const std::size_t d2 = static_cast<std::size_t>(dims[2]);
    const std::size_t d3 = static_cast<std::size_t>(dims[3]);

    // MATLAB commonly writes time-major arrays (Nt x nf x nf x nf) and HDF5 presents nf x nf x nf x Nt.
    if (d1 == nf && d2 == nf && d3 == nf) {
        out.dims_use = dims; // already [Nt,nf,nf,nf]
        out.Nt = d0;
        return out;
    }
    if (d0 == nf && d1 == nf && d2 == nf) {
        out.dims_use = reverse_dims(dims); // interpret as [Nt,nf,nf,nf]
        out.Nt = static_cast<std::size_t>(out.dims_use[0]);
        return out;
    }
    throw std::runtime_error(base + " dims do not match nf=" + std::to_string(nf) +
                             ": " + dims_to_string(dims));
}

Eigen::MatrixXcd read_matrix(hid_t file_id, const std::string& base) {
    std::vector<hsize_t> dims;
    auto data = read_complex(file_id, base, &dims);
    if (dims.size() != 2) {
        throw std::runtime_error(base + " must be rank-2, got dims=" + dims_to_string(dims));
    }
    const std::size_t rows = static_cast<std::size_t>(dims[0]);
    const std::size_t cols = static_cast<std::size_t>(dims[1]);
    Eigen::MatrixXcd out(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t r = 0; r < rows; ++r) {
            out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                at_colmajor_2d(data, dims, r, c);
        }
    }
    return out;
}

// ----------------------------- Compare helpers -----------------------------

struct ErrSummary {
    double max_abs{0.0};
    double max_rel{0.0};
    bool ok{true};
};

void update_err(ErrSummary& s, const cd& got, const cd& expect, double atol, double rtol) {
    const double diff = std::abs(got - expect);
    const double rel = diff / std::max(1.0, std::abs(expect));
    s.max_abs = std::max(s.max_abs, diff);
    s.max_rel = std::max(s.max_rel, rel);
    if (diff > atol + rtol * std::max(1.0, std::abs(expect))) s.ok = false;
}

enum class GammaRule {
    Rect,
    Trapz
};

Eigen::MatrixXcd compute_gamma_prefix_matrix(const std::vector<cd>& C,
                                             double dt,
                                             const std::vector<double>& omegas,
                                             GammaRule rule) {
    const std::size_t Nt = C.size();
    const std::size_t nf = omegas.size();
    if (Nt == 0 || nf == 0 || !(dt > 0.0)) return Eigen::MatrixXcd();
    Eigen::MatrixXcd G(static_cast<Eigen::Index>(Nt), static_cast<Eigen::Index>(nf));
    G.row(0).setZero();
    const cd half_dt(dt / 2.0, 0.0);
    for (std::size_t b = 0; b < nf; ++b) {
        const cd step = std::exp(cd{0.0, omegas[b] * dt});
        cd phi(1.0, 0.0);
        cd acc(0.0, 0.0);
        if (rule == GammaRule::Rect) {
            acc += dt * phi * C[0];
            G(static_cast<Eigen::Index>(0), static_cast<Eigen::Index>(b)) = acc;
        }
        for (std::size_t k = 1; k < Nt; ++k) {
            const cd phi_next = phi * step;
            if (rule == GammaRule::Trapz) {
                acc += half_dt * (phi * C[k - 1] + phi_next * C[k]);
            } else {
                acc += dt * phi_next * C[k];
            }
            G(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(b)) = acc;
            phi = phi_next;
        }
    }
    return G;
}

long long detect_ij_base(const std::vector<long long>& map_ij, std::size_t nf) {
    if (map_ij.empty()) return 0;
    const auto min_ij = *std::min_element(map_ij.begin(), map_ij.end());
    const auto max_ij = *std::max_element(map_ij.begin(), map_ij.end());
    if (min_ij >= 1 && max_ij <= static_cast<long long>(nf)) return 1;
    if (min_ij >= 0 && max_ij <= static_cast<long long>(nf) - 1) return 0;
    if (min_ij == 0 && max_ij == static_cast<long long>(nf)) return 1;
    return -1;
}

// ----------------------------- CLI helpers -----------------------------

std::string trim_copy(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string lower_copy(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

std::vector<std::size_t> resolve_tidx_list(const std::vector<std::string>& tokens,
                                           std::size_t Nt,
                                           bool one_based) {
    std::vector<std::size_t> out;
    for (const auto& tok : tokens) {
        std::size_t idx = 0;
        if (tok == "mid") {
            idx = Nt / 2;
        } else if (tok == "last") {
            if (Nt == 0) throw std::runtime_error("Nt is zero");
            idx = Nt - 1;
        } else {
            long long val = std::stoll(tok);
            if (one_based) val -= 1;
            if (val < 0) throw std::runtime_error("tidx must be >= 0");
            idx = static_cast<std::size_t>(val);
        }
        if (idx >= Nt) {
            throw std::runtime_error("tidx out of range: " + std::to_string(idx) +
                                     " (Nt=" + std::to_string(Nt) + ")");
        }
        if (std::find(out.begin(), out.end(), idx) == out.end()) out.push_back(idx);
    }
    return out;
}

struct Options {
    std::string file{"tests/tcl_test.h5"};
    std::string tidx_list{"0,mid,last"};
    bool one_based{false};

    bool list{false};
    bool dump_map{false};

    bool compare_gt{false};
    bool compare_fcr{false};
    bool compare_gw{false};

    bool print_gt{false};
    bool print_fcr{false};
    bool print_mikx{false};
    bool print_gw{false};
    bool print_redt{false};
    bool print_tcl{false};

    std::size_t gt_offset{0};
    std::size_t fcr_offset{0};
    std::size_t fcr_fft_pad{0};

    double atol{1e-6};
    double rtol{1e-6};

    GammaRule gamma_rule{GammaRule::Rect};
    taco::tcl4::FCRMethod method{taco::tcl4::FCRMethod::Convolution};
};

void print_usage() {
    std::cout
        << "Usage: tcl4_h5_compare.exe [--file=PATH] [--tidx=LIST] [--one-based]\n"
        << "                           [--list] [--dump-map]\n"
        << "                           [--compare-gt] [--compare-fcr] [--compare-gw]\n"
        << "                           [--print-gt] [--print-fcr] [--print-mikx] [--print-gw]\n"
        << "                           [--print-redt] [--print-tcl]\n"
        << "                           [--gt-offset=N] [--fcr-offset=N] [--fcr-fft-pad=N]\n"
        << "                           [--gamma-rule=rect|trapz] [--method=direct|convolution]\n"
        << "                           [--atol=VAL] [--rtol=VAL]\n"
        << "Defaults: file=tests/tcl_test.h5, tidx=0,mid,last; if no compare/print flags are given, runs --compare-gt\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options opt;
        bool compare_or_print_seen = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg.rfind("--file=", 0) == 0) {
                opt.file = arg.substr(7);
                continue;
            }
            if (arg.rfind("--tidx=", 0) == 0) {
                opt.tidx_list = arg.substr(7);
                continue;
            }
            if (arg == "--one-based") {
                opt.one_based = true;
                continue;
            }
            if (arg == "--list") {
                opt.list = true;
                continue;
            }
            if (arg == "--dump-map") {
                opt.dump_map = true;
                continue;
            }
            if (arg == "--compare-gt") {
                opt.compare_gt = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--compare-fcr") {
                opt.compare_fcr = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--compare-gw") {
                opt.compare_gw = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--print-gt") {
                opt.print_gt = true;
                opt.compare_gt = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--print-fcr") {
                opt.print_fcr = true;
                opt.compare_fcr = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--print-mikx") {
                opt.print_mikx = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--print-gw") {
                opt.print_gw = true;
                opt.compare_gw = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--print-redt") {
                opt.print_redt = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg == "--print-tcl") {
                opt.print_tcl = true;
                compare_or_print_seen = true;
                continue;
            }
            if (arg.rfind("--gt-offset=", 0) == 0) {
                opt.gt_offset = static_cast<std::size_t>(std::stoull(arg.substr(12)));
                continue;
            }
            if (arg.rfind("--fcr-offset=", 0) == 0) {
                opt.fcr_offset = static_cast<std::size_t>(std::stoull(arg.substr(13)));
                continue;
            }
            if (arg.rfind("--fcr-fft-pad=", 0) == 0) {
                opt.fcr_fft_pad = static_cast<std::size_t>(std::stoull(arg.substr(14)));
                continue;
            }
            if (arg.rfind("--gamma-rule=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(13));
                opt.gamma_rule = (val == "trapz") ? GammaRule::Trapz : GammaRule::Rect;
                continue;
            }
            if (arg.rfind("--method=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(9));
                opt.method = (val == "direct") ? taco::tcl4::FCRMethod::Direct
                                               : taco::tcl4::FCRMethod::Convolution;
                continue;
            }
            if (arg.rfind("--atol=", 0) == 0) {
                opt.atol = std::stod(arg.substr(7));
                continue;
            }
            if (arg.rfind("--rtol=", 0) == 0) {
                opt.rtol = std::stod(arg.substr(7));
                continue;
            }
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 1;
        }
        if (!compare_or_print_seen) {
            opt.compare_gt = true;
        }

        H5File h5(opt.file);

        if (opt.list) {
            for (const auto& d : list_datasets(h5.id)) {
                std::cout << d.path << " dims=" << dims_to_string(d.dims) << "\n";
            }
            if (!opt.dump_map && !compare_or_print_seen) return 0;
        }

        const double dt = read_scalar_double(h5.id, "/params/dt");

        if (!dataset_exists(h5.id, "/bath/C/re")) {
            throw std::runtime_error("/bath/C not found in file");
        }
        std::vector<hsize_t> dims_c;
        auto C_full = read_complex(h5.id, "/bath/C", &dims_c);
        if (C_full.empty()) throw std::runtime_error("Empty /bath/C");
        const std::size_t Nt_c = C_full.size();

        std::vector<double> tvals;
        if (dataset_exists(h5.id, "/time/t")) {
            std::vector<hsize_t> dims_t;
            tvals = read_array<double>(h5.id, "/time/t", H5T_NATIVE_DOUBLE, &dims_t);
        }

        const std::size_t N = static_cast<std::size_t>(read_scalar_double(h5.id, "/map/N"));
        const std::size_t nf = static_cast<std::size_t>(read_scalar_double(h5.id, "/map/nf"));
        if (N == 0 || nf == 0) throw std::runtime_error("map/N and map/nf must be > 0");

        std::vector<hsize_t> dims_ij;
        auto map_ij_raw = read_array<long long>(h5.id, "/map/ij", H5T_NATIVE_LLONG, &dims_ij);
        if (map_ij_raw.size() != N * N) {
            throw std::runtime_error("map/ij length must be N^2");
        }
        const long long base = detect_ij_base(map_ij_raw, nf);
        if (base < 0) {
            const auto minv = *std::min_element(map_ij_raw.begin(), map_ij_raw.end());
            const auto maxv = *std::max_element(map_ij_raw.begin(), map_ij_raw.end());
            throw std::runtime_error("map/ij values must be 0..nf-1 or 1..nf (min=" +
                                     std::to_string(minv) + ", max=" + std::to_string(maxv) +
                                     ", nf=" + std::to_string(nf) + ")");
        }
        std::vector<int> map_ij0(map_ij_raw.size(), 0);
        for (std::size_t ii = 0; ii < map_ij_raw.size(); ++ii) {
            const long long v = map_ij_raw[ii] - base;
            if (v < 0 || v >= static_cast<long long>(nf)) {
                throw std::runtime_error("map/ij contains out-of-range bucket indices");
            }
            map_ij0[ii] = static_cast<int>(v);
        }

        taco::sys::System system;
        if (dataset_exists(h5.id, "/system/Eig/re")) {
            std::vector<hsize_t> dims_e;
            auto eig_re = read_array<double>(h5.id, "/system/Eig/re", H5T_NATIVE_DOUBLE, &dims_e);
            system.eig.dim = eig_re.size();
            system.eig.eps = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(eig_re.size()));
            for (std::size_t ii = 0; ii < eig_re.size(); ++ii) {
                system.eig.eps(static_cast<Eigen::Index>(ii)) = eig_re[ii];
            }
            system.eig.U = Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(system.eig.dim),
                                                      static_cast<Eigen::Index>(system.eig.dim));
            system.eig.U_dag = system.eig.U;
        } else if (dataset_exists(h5.id, "/system/H/re")) {
            system.eig = taco::sys::Eigensystem(read_matrix(h5.id, "/system/H"));
        } else {
            throw std::runtime_error("Need /system/Eig or /system/H to derive Bohr frequencies");
        }
        system.bf = taco::sys::BohrFrequencies(system.eig.eps);

        if (!dataset_exists(h5.id, "/system/A/re")) {
            throw std::runtime_error("Need /system/A for MIKX/GW");
        }
        const Eigen::MatrixXcd A_eig = read_matrix(h5.id, "/system/A");
        if (static_cast<std::size_t>(A_eig.rows()) != N || static_cast<std::size_t>(A_eig.cols()) != N) {
            throw std::runtime_error("/system/A dims do not match map/N");
        }

        // Derive bucket omegas in map/ij bucket order.
        const double omega_tol = 1e-9;
        std::vector<double> omegas_u(nf, std::numeric_limits<double>::quiet_NaN());
        std::vector<char> seen(nf, 0);
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t k = 0; k < N; ++k) {
                const std::size_t idx = j + N * k;
                const int b = map_ij0[idx];
                const double w = system.bf.omega(static_cast<Eigen::Index>(j),
                                                 static_cast<Eigen::Index>(k));
                if (!seen[static_cast<std::size_t>(b)]) {
                    omegas_u[static_cast<std::size_t>(b)] = w;
                    seen[static_cast<std::size_t>(b)] = 1;
                } else if (std::abs(omegas_u[static_cast<std::size_t>(b)] - w) > omega_tol) {
                    throw std::runtime_error("map/ij bucket contains inconsistent omega values");
                }
            }
        }

        if (opt.dump_map) {
            if (dataset_exists(h5.id, "/map/omegas")) {
                std::vector<hsize_t> dims_om;
                auto omegas_raw = read_array<double>(h5.id, "/map/omegas", H5T_NATIVE_DOUBLE, &dims_om);
                std::cout << "map/omegas raw:";
                for (double w : omegas_raw) std::cout << " " << w;
                std::cout << "\n";
            }
            std::cout << "omegas_u (bucket order):";
            for (double w : omegas_u) std::cout << " " << w;
            std::cout << "\nmap.ij (raw):\n";
            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t k = 0; k < N; ++k) {
                    std::cout << map_ij_raw[j + N * k] << (k + 1 < N ? " " : "");
                }
                std::cout << "\n";
            }
            std::cout << "map.ij (0-based):\n";
            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t k = 0; k < N; ++k) {
                    std::cout << map_ij0[j + N * k] << (k + 1 < N ? " " : "");
                }
                std::cout << "\n";
            }
        }

        Eigen::MatrixXcd gamma_series = compute_gamma_prefix_matrix(C_full, dt, omegas_u, opt.gamma_rule);
        if (gamma_series.rows() != static_cast<Eigen::Index>(Nt_c) ||
            gamma_series.cols() != static_cast<Eigen::Index>(nf)) {
            throw std::runtime_error("Failed to build gamma_series");
        }

        system.fidx.tol = omega_tol;
        system.fidx.buckets.assign(nf, taco::sys::FrequencyBucket{});
        for (std::size_t b = 0; b < nf; ++b) system.fidx.buckets[b].omega = omegas_u[b];
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t k = 0; k < N; ++k) {
                system.fidx.buckets[static_cast<std::size_t>(map_ij0[j + N * k])]
                    .pairs.emplace_back(static_cast<int>(j), static_cast<int>(k));
            }
        }
        system.A_eig = {A_eig};
        system.A_eig_parts = taco::sys::decompose_operators_by_frequency(system.A_eig, system.bf, system.fidx);

        if (opt.fcr_fft_pad > 0) taco::tcl4::set_fcr_fft_pad_factor(opt.fcr_fft_pad);

        taco::tcl4::TripleKernelSeries kernels;
        taco::tcl4::Tcl4Map map;
        const bool need_kernels = opt.compare_fcr || opt.compare_gw || opt.print_mikx || opt.print_fcr;
        const bool need_map = opt.compare_gw || opt.print_mikx;
        if (need_kernels) kernels = taco::tcl4::compute_triple_kernels(system, gamma_series, dt, 2, opt.method);
        if (need_map) map = taco::tcl4::build_map(system, {});

        // Load file outputs (only when needed).
        const std::size_t N2 = N * N;
        FlatSeries Gt_file;
        FlatSeries GW_file;
        FlatSeries RedT_file;
        FlatSeries TCL_file;
        KernelSeries F_file;
        KernelSeries C_file;
        KernelSeries R_file;

        std::size_t Nt_avail = Nt_c;
        if (opt.compare_gt) {
            if (!dataset_exists(h5.id, "/out/Gt_flat/re")) throw std::runtime_error("Missing /out/Gt_flat");
            Gt_file = load_flat_series(h5.id, "/out/Gt_flat", N2);
            if (opt.gt_offset >= Gt_file.Nt) throw std::runtime_error("gt-offset exceeds Gt length");
            Nt_avail = std::min(Nt_avail, Gt_file.Nt - opt.gt_offset);
        }
        if (opt.compare_gw) {
            if (!dataset_exists(h5.id, "/out/GW_flat/re")) throw std::runtime_error("Missing /out/GW_flat");
            GW_file = load_flat_series(h5.id, "/out/GW_flat", N2 * N2);
            Nt_avail = std::min(Nt_avail, GW_file.Nt);
        }
        if (opt.compare_fcr) {
            if (!dataset_exists(h5.id, "/kernels/F_all/re") ||
                !dataset_exists(h5.id, "/kernels/C_all/re") ||
                !dataset_exists(h5.id, "/kernels/R_all/re")) {
                throw std::runtime_error("Missing /kernels/F_all,C_all,R_all");
            }
            F_file = load_kernel_series(h5.id, "/kernels/F_all", nf);
            C_file = load_kernel_series(h5.id, "/kernels/C_all", nf);
            R_file = load_kernel_series(h5.id, "/kernels/R_all", nf);
            if (opt.fcr_offset >= F_file.Nt) throw std::runtime_error("fcr-offset exceeds kernel length");
            Nt_avail = std::min(Nt_avail, F_file.Nt - opt.fcr_offset);
        }
        if (opt.print_redt) {
            if (!dataset_exists(h5.id, "/out/RedT_flat/re")) throw std::runtime_error("Missing /out/RedT_flat");
            RedT_file = load_flat_series(h5.id, "/out/RedT_flat", N2 * N2);
            Nt_avail = std::min(Nt_avail, RedT_file.Nt);
        }
        if (opt.print_tcl) {
            if (!dataset_exists(h5.id, "/out/TCL_flat/re")) throw std::runtime_error("Missing /out/TCL_flat");
            TCL_file = load_flat_series(h5.id, "/out/TCL_flat", N2 * N2);
            Nt_avail = std::min(Nt_avail, TCL_file.Nt);
        }

        const auto tidx_tokens = split_csv(opt.tidx_list);
        const auto tidx_list = resolve_tidx_list(tidx_tokens, Nt_avail, opt.one_based);

        auto t_at = [&](std::size_t tidx) -> double {
            if (!tvals.empty() && tidx < tvals.size()) return tvals[tidx];
            return dt * static_cast<double>(tidx);
        };

        std::cout << std::setprecision(12);
        std::size_t failures = 0;

        if (opt.compare_gt) {
            ErrSummary gtstat;
            for (std::size_t tidx : tidx_list) {
                const std::size_t file_tidx = tidx + opt.gt_offset;
                if (opt.print_gt) std::cout << "Gt tidx=" << tidx << " t=" << t_at(tidx) << "\n";
                for (std::size_t j = 0; j < N; ++j) {
                    for (std::size_t k = 0; k < N; ++k) {
                        const std::size_t idx = j + N * k;
                        const int b = map_ij0[idx];
                        const cd got = gamma_series(static_cast<Eigen::Index>(tidx), b);
                        const cd expect = Gt_file.at(file_tidx, idx);
                        update_err(gtstat, got, expect, opt.atol, opt.rtol);
                        if (opt.print_gt) {
                            std::cout << "  (" << j << "," << k << ") omega=" << omegas_u[static_cast<std::size_t>(b)]
                                      << " Gt=(" << got.real() << "," << got.imag() << ")"
                                      << " file=(" << expect.real() << "," << expect.imag() << ")\n";
                        }
                    }
                }
            }
            if (!gtstat.ok) failures++;
            std::cout << "Gt(matrix) max_abs=" << gtstat.max_abs
                      << " max_rel=" << gtstat.max_rel
                      << (gtstat.ok ? " ok\n" : " FAIL\n");
        }

        if (opt.compare_fcr) {
            ErrSummary fstat, cstat, rstat;
            for (std::size_t tidx : tidx_list) {
                const std::size_t file_tidx = tidx + opt.fcr_offset;
                if (opt.print_fcr) std::cout << "FCR tidx=" << tidx << " t=" << t_at(tidx) << "\n";
                for (std::size_t i0 = 0; i0 < nf; ++i0) {
                    for (std::size_t i1 = 0; i1 < nf; ++i1) {
                        for (std::size_t i2 = 0; i2 < nf; ++i2) {
                            const cd got_f = kernels.F[i0][i1][i2](static_cast<Eigen::Index>(tidx));
                            const cd got_c = kernels.C[i0][i1][i2](static_cast<Eigen::Index>(tidx));
                            const cd got_r = kernels.R[i0][i1][i2](static_cast<Eigen::Index>(tidx));
                            const cd exp_f = F_file.at(file_tidx, i0, i1, i2);
                            const cd exp_c = C_file.at(file_tidx, i0, i1, i2);
                            const cd exp_r = R_file.at(file_tidx, i0, i1, i2);
                            update_err(fstat, got_f, exp_f, opt.atol, opt.rtol);
                            update_err(cstat, got_c, exp_c, opt.atol, opt.rtol);
                            update_err(rstat, got_r, exp_r, opt.atol, opt.rtol);
                            if (opt.print_fcr) {
                                std::cout << "  (" << i0 << "," << i1 << "," << i2 << ") "
                                          << "F=(" << got_f.real() << "," << got_f.imag() << ")"
                                          << " file=(" << exp_f.real() << "," << exp_f.imag() << ")\n";
                            }
                        }
                    }
                }
            }
            if (!fstat.ok || !cstat.ok || !rstat.ok) failures++;
            std::cout << "F_all max_abs=" << fstat.max_abs << " max_rel=" << fstat.max_rel
                      << (fstat.ok ? " ok\n" : " FAIL\n");
            std::cout << "C_all max_abs=" << cstat.max_abs << " max_rel=" << cstat.max_rel
                      << (cstat.ok ? " ok\n" : " FAIL\n");
            std::cout << "R_all max_abs=" << rstat.max_abs << " max_rel=" << rstat.max_rel
                      << (rstat.ok ? " ok\n" : " FAIL\n");
        }

        if (opt.print_mikx) {
            if (!need_map || !need_kernels) throw std::runtime_error("MIKX requires computed kernels");
            for (std::size_t tidx : tidx_list) {
                const taco::tcl4::MikxTensors mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
                std::cout << "MIKX tidx=" << tidx << " t=" << t_at(tidx) << " N=" << mikx.N << "\n";
                std::cout << "M:\n";
                for (int r = 0; r < mikx.M.rows(); ++r) {
                    for (int c = 0; c < mikx.M.cols(); ++c) {
                        const cd v = mikx.M(r, c);
                        std::cout << "  (" << r << "," << c << ")=(" << v.real() << "," << v.imag() << ")\n";
                    }
                }
                std::cout << "I:\n";
                for (int r = 0; r < mikx.I.rows(); ++r) {
                    for (int c = 0; c < mikx.I.cols(); ++c) {
                        const cd v = mikx.I(r, c);
                        std::cout << "  (" << r << "," << c << ")=(" << v.real() << "," << v.imag() << ")\n";
                    }
                }
                std::cout << "K:\n";
                for (int r = 0; r < mikx.K.rows(); ++r) {
                    for (int c = 0; c < mikx.K.cols(); ++c) {
                        const cd v = mikx.K(r, c);
                        std::cout << "  (" << r << "," << c << ")=(" << v.real() << "," << v.imag() << ")\n";
                    }
                }
                std::cout << "X (flat, column-major j,k,p,q,r,s):\n";
                const std::size_t Nloc = static_cast<std::size_t>(mikx.N);
                for (std::size_t idx = 0; idx < mikx.X.size(); ++idx) {
                    std::size_t tmp = idx;
                    const std::size_t j = tmp % Nloc; tmp /= Nloc;
                    const std::size_t k = tmp % Nloc; tmp /= Nloc;
                    const std::size_t p = tmp % Nloc; tmp /= Nloc;
                    const std::size_t q = tmp % Nloc; tmp /= Nloc;
                    const std::size_t r = tmp % Nloc; tmp /= Nloc;
                    const std::size_t s = tmp % Nloc;
                    const cd v = mikx.X[idx];
                    std::cout << "  (" << j << "," << k << "," << p << "," << q << "," << r << "," << s
                              << ")=(" << v.real() << "," << v.imag() << ")\n";
                }
            }
        }

        if (opt.compare_gw) {
            if (!need_map || !need_kernels) throw std::runtime_error("GW requires computed kernels");
            ErrSummary gwstat;
            for (std::size_t tidx : tidx_list) {
                taco::tcl4::MikxTensors mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
                Eigen::MatrixXcd GW = taco::tcl4::assemble_liouvillian(mikx, system.A_eig);
                if (opt.print_gw) std::cout << "GW tidx=" << tidx << " t=" << t_at(tidx) << "\n";
                for (std::size_t r = 0; r < N2; ++r) {
                    for (std::size_t c = 0; c < N2; ++c) {
                        const cd expect = GW_file.at(tidx, r + N2 * c);
                        update_err(gwstat, GW(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)),
                                   expect, opt.atol, opt.rtol);
                        if (opt.print_gw) {
                            const cd got = GW(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
                            std::cout << "  (" << r << "," << c << ") GW=("
                                      << got.real() << "," << got.imag() << ")"
                                      << " file=(" << expect.real() << "," << expect.imag() << ")\n";
                        }
                    }
                }
                std::cout << "GW tidx=" << tidx << " t=" << t_at(tidx)
                          << " max_abs=" << gwstat.max_abs
                          << " max_rel=" << gwstat.max_rel
                          << (gwstat.ok ? " ok\n" : " FAIL\n");
            }
            if (!gwstat.ok) failures++;
        }

        if (opt.print_redt) {
            for (std::size_t tidx : tidx_list) {
                std::cout << "RedT(file) tidx=" << tidx << " t=" << t_at(tidx) << "\n";
                for (std::size_t r = 0; r < N2; ++r) {
                    for (std::size_t c = 0; c < N2; ++c) {
                        const cd v = RedT_file.at(tidx, r + N2 * c);
                        std::cout << "  (" << r << "," << c << ")=(" << v.real() << "," << v.imag() << ")\n";
                    }
                }
            }
        }

        if (opt.print_tcl) {
            for (std::size_t tidx : tidx_list) {
                std::cout << "TCL(file) tidx=" << tidx << " t=" << t_at(tidx) << "\n";
                for (std::size_t r = 0; r < N2; ++r) {
                    for (std::size_t c = 0; c < N2; ++c) {
                        const cd v = TCL_file.at(tidx, r + N2 * c);
                        std::cout << "  (" << r << "," << c << ")=(" << v.real() << "," << v.imag() << ")\n";
                    }
                }
            }
        }

        if (failures > 0) {
            std::cout << "FAILED: " << failures << " comparison(s)\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
