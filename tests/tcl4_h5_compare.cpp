#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "taco/gamma.hpp"
#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_mikx.hpp"

#include "hdf5.h"

namespace {

struct H5File {
    hid_t id{-1};
    explicit H5File(const std::string& path) {
        id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (id < 0) throw std::runtime_error("Failed to open HDF5 file: " + path);
    }
    ~H5File() { if (id >= 0) H5Fclose(id); }
};

struct H5Dataset {
    hid_t id{-1};
    explicit H5Dataset(hid_t id_) : id(id_) {}
    ~H5Dataset() { if (id >= 0) H5Dclose(id); }
};

struct H5Space {
    hid_t id{-1};
    explicit H5Space(hid_t id_) : id(id_) {}
    ~H5Space() { if (id >= 0) H5Sclose(id); }
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

std::vector<hsize_t> squeeze_dims(const std::vector<hsize_t>& dims) {
    std::vector<hsize_t> out;
    out.reserve(dims.size());
    for (hsize_t d : dims) {
        if (d != 1 || dims.size() == 1) out.push_back(d);
    }
    if (out.empty()) out.push_back(1);
    return out;
}

std::vector<std::size_t> squeezed_index_map(const std::vector<hsize_t>& dims) {
    std::vector<std::size_t> map;
    map.reserve(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] != 1 || dims.size() == 1) map.push_back(i);
    }
    if (map.empty()) map.push_back(0);
    return map;
}

std::size_t numel(const std::vector<hsize_t>& dims) {
    std::size_t n = 1;
    for (hsize_t d : dims) n *= static_cast<std::size_t>(d);
    return n;
}

template <typename T>
std::vector<T> read_array(hid_t file_id, const std::string& path, hid_t native_type,
                          std::vector<hsize_t>* dims_out) {
    H5Dataset dset(H5Dopen2(file_id, path.c_str(), H5P_DEFAULT));
    if (dset.id < 0) throw std::runtime_error("Failed to open dataset: " + path);
    auto dims = get_dims(dset.id);
    const std::size_t count = numel(dims);
    std::vector<T> data(count);
    if (count > 0) {
        if (H5Dread(dset.id, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
            throw std::runtime_error("Failed to read dataset: " + path);
        }
    }
    if (dims_out) *dims_out = dims;
    return data;
}

template <typename T>
std::vector<T> read_array_prefix_dim(hid_t file_id, const std::string& path, hid_t native_type,
                                     std::size_t max_count, std::size_t dim,
                                     std::vector<hsize_t>* dims_out) {
    H5Dataset dset(H5Dopen2(file_id, path.c_str(), H5P_DEFAULT));
    if (dset.id < 0) throw std::runtime_error("Failed to open dataset: " + path);
    auto dims = get_dims(dset.id);
    if (dims.empty()) throw std::runtime_error("Dataset has no dimensions: " + path);
    if (dim >= dims.size()) throw std::runtime_error("Prefix dim out of range: " + path);
    const hsize_t useN = static_cast<hsize_t>(std::min<std::size_t>(max_count, dims[dim]));
    std::vector<hsize_t> count = dims;
    count[dim] = useN;
    std::vector<hsize_t> start(dims.size(), 0);
    const std::size_t count_total = numel(count);
    std::vector<T> data(count_total);
    H5Space space(H5Dget_space(dset.id));
    if (space.id < 0) throw std::runtime_error("Failed to get dataspace");
    if (H5Sselect_hyperslab(space.id, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr) < 0) {
        throw std::runtime_error("Failed to select hyperslab: " + path);
    }
    H5Space memspace(H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr));
    if (memspace.id < 0) throw std::runtime_error("Failed to create memspace");
    if (count_total > 0) {
        if (H5Dread(dset.id, native_type, memspace.id, space.id, H5P_DEFAULT, data.data()) < 0) {
            throw std::runtime_error("Failed to read dataset: " + path);
        }
    }
    if (dims_out) *dims_out = count;
    return data;
}

template <typename T>
std::vector<T> read_array_prefix(hid_t file_id, const std::string& path, hid_t native_type,
                                 std::size_t max_first, std::vector<hsize_t>* dims_out) {
    return read_array_prefix_dim<T>(file_id, path, native_type, max_first, 0, dims_out);
}

double read_scalar_double(hid_t file_id, const std::string& path) {
    std::vector<hsize_t> dims;
    auto data = read_array<double>(file_id, path, H5T_NATIVE_DOUBLE, &dims);
    if (data.size() != 1) throw std::runtime_error("Expected scalar at: " + path);
    return data[0];
}

std::vector<std::complex<double>> read_complex_array(hid_t file_id,
                                                     const std::string& base,
                                                     std::vector<hsize_t>* dims_out) {
    std::vector<hsize_t> dims_re;
    std::vector<hsize_t> dims_im;
    auto re = read_array<double>(file_id, base + "/re", H5T_NATIVE_DOUBLE, &dims_re);
    auto im = read_array<double>(file_id, base + "/im", H5T_NATIVE_DOUBLE, &dims_im);
    if (dims_re != dims_im) throw std::runtime_error("Complex dims mismatch at: " + base);
    if (re.size() != im.size()) throw std::runtime_error("Complex size mismatch at: " + base);
    std::vector<std::complex<double>> out;
    out.reserve(re.size());
    for (std::size_t i = 0; i < re.size(); ++i) out.emplace_back(re[i], im[i]);
    if (dims_out) *dims_out = dims_re;
    return out;
}

std::vector<std::complex<double>> read_complex_array_prefix(hid_t file_id,
                                                            const std::string& base,
                                                            std::size_t max_first,
                                                            std::vector<hsize_t>* dims_out) {
    std::vector<hsize_t> dims_re;
    std::vector<hsize_t> dims_im;
    auto re = read_array_prefix<double>(file_id, base + "/re", H5T_NATIVE_DOUBLE, max_first, &dims_re);
    auto im = read_array_prefix<double>(file_id, base + "/im", H5T_NATIVE_DOUBLE, max_first, &dims_im);
    if (dims_re != dims_im) throw std::runtime_error("Complex dims mismatch at: " + base);
    if (re.size() != im.size()) throw std::runtime_error("Complex size mismatch at: " + base);
    std::vector<std::complex<double>> out;
    out.reserve(re.size());
    for (std::size_t i = 0; i < re.size(); ++i) out.emplace_back(re[i], im[i]);
    if (dims_out) *dims_out = dims_re;
    return out;
}

std::vector<std::complex<double>> read_complex_array_prefix_dim(hid_t file_id,
                                                                const std::string& base,
                                                                std::size_t max_count,
                                                                std::size_t dim,
                                                                std::vector<hsize_t>* dims_out) {
    std::vector<hsize_t> dims_re;
    std::vector<hsize_t> dims_im;
    auto re = read_array_prefix_dim<double>(file_id, base + "/re", H5T_NATIVE_DOUBLE, max_count, dim, &dims_re);
    auto im = read_array_prefix_dim<double>(file_id, base + "/im", H5T_NATIVE_DOUBLE, max_count, dim, &dims_im);
    if (dims_re != dims_im) throw std::runtime_error("Complex dims mismatch at: " + base);
    if (re.size() != im.size()) throw std::runtime_error("Complex size mismatch at: " + base);
    std::vector<std::complex<double>> out;
    out.reserve(re.size());
    for (std::size_t i = 0; i < re.size(); ++i) out.emplace_back(re[i], im[i]);
    if (dims_out) *dims_out = dims_re;
    return out;
}

Eigen::MatrixXcd to_matrix_colmajor(const std::vector<std::complex<double>>& data,
                                    std::size_t rows,
                                    std::size_t cols) {
    if (data.size() != rows * cols) {
        throw std::runtime_error("Matrix size mismatch");
    }
    using MapMat = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;
    MapMat map(data.data(), static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    return Eigen::MatrixXcd(map);
}

std::complex<double> at_colmajor_2d(const std::vector<std::complex<double>>& data,
                                    const std::vector<hsize_t>& dims,
                                    std::size_t row,
                                    std::size_t col) {
    return data[row + static_cast<std::size_t>(dims[0]) * col];
}

std::complex<double> at_colmajor_3d(const std::vector<std::complex<double>>& data,
                                    const std::vector<hsize_t>& dims,
                                    std::size_t i0,
                                    std::size_t i1,
                                    std::size_t i2) {
    const std::size_t d0 = static_cast<std::size_t>(dims[0]);
    const std::size_t d1 = static_cast<std::size_t>(dims[1]);
    return data[i0 + d0 * (i1 + d1 * i2)];
}

std::complex<double> at_colmajor_4d(const std::vector<std::complex<double>>& data,
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

std::string dims_to_string(const std::vector<hsize_t>& dims) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) os << ",";
        os << dims[i];
    }
    os << "]";
    return os.str();
}

struct GwLayout {
    enum class Mode {
        FlatTimeRow,
        FlatTimeCol,
        MatTimeFirst,
        MatTimeMiddle,
        MatTimeLast
    };
    Mode mode{Mode::FlatTimeRow};
    std::vector<hsize_t> dims;
    std::size_t Nt{0};
    std::size_t time_dim{0};
};

GwLayout infer_gw_layout(const std::vector<hsize_t>& dims, std::size_t N2) {
    const std::size_t flat_len = N2 * N2;
    GwLayout layout;
    layout.dims = dims;
    if (dims.size() == 2) {
        if (static_cast<std::size_t>(dims[1]) == flat_len) {
            layout.mode = GwLayout::Mode::FlatTimeRow;
            layout.Nt = static_cast<std::size_t>(dims[0]);
            layout.time_dim = 0;
            return layout;
        }
        if (static_cast<std::size_t>(dims[0]) == flat_len) {
            layout.mode = GwLayout::Mode::FlatTimeCol;
            layout.Nt = static_cast<std::size_t>(dims[1]);
            layout.time_dim = 1;
            return layout;
        }
    } else if (dims.size() == 3) {
        const std::size_t d0 = static_cast<std::size_t>(dims[0]);
        const std::size_t d1 = static_cast<std::size_t>(dims[1]);
        const std::size_t d2 = static_cast<std::size_t>(dims[2]);
        if (d0 == N2 && d1 == N2) {
            layout.mode = GwLayout::Mode::MatTimeLast;
            layout.Nt = d2;
            layout.time_dim = 2;
            return layout;
        }
        if (d1 == N2 && d2 == N2) {
            layout.mode = GwLayout::Mode::MatTimeFirst;
            layout.Nt = d0;
            layout.time_dim = 0;
            return layout;
        }
        if (d0 == N2 && d2 == N2) {
            layout.mode = GwLayout::Mode::MatTimeMiddle;
            layout.Nt = d1;
            layout.time_dim = 1;
            return layout;
        }
    }
    throw std::runtime_error("Unsupported GW_flat dims: " + dims_to_string(dims));
}

std::complex<double> gw_value(const GwLayout& layout,
                              const std::vector<std::complex<double>>& data,
                              std::size_t tidx,
                              std::size_t row,
                              std::size_t col,
                              std::size_t N2) {
    const std::size_t flat_idx = row * N2 + col;
    switch (layout.mode) {
        case GwLayout::Mode::FlatTimeRow:
            return at_colmajor_2d(data, layout.dims, tidx, flat_idx);
        case GwLayout::Mode::FlatTimeCol:
            return at_colmajor_2d(data, layout.dims, flat_idx, tidx);
        case GwLayout::Mode::MatTimeLast:
            return at_colmajor_3d(data, layout.dims, row, col, tidx);
        case GwLayout::Mode::MatTimeFirst:
            return at_colmajor_3d(data, layout.dims, tidx, row, col);
        case GwLayout::Mode::MatTimeMiddle:
            return at_colmajor_3d(data, layout.dims, row, tidx, col);
    }
    return std::complex<double>(0.0, 0.0);
}

struct FcrLayout {
    enum class Mode {
        TimeFirst,
        TimeLast
    };
    Mode mode{Mode::TimeFirst};
    std::vector<hsize_t> dims;
    std::size_t Nt{0};
    std::size_t time_dim{0};
};

FcrLayout infer_fcr_layout(const std::vector<hsize_t>& dims, std::size_t nf) {
    if (dims.size() != 4) {
        throw std::runtime_error("Unsupported F/C/R dims: " + dims_to_string(dims));
    }
    FcrLayout layout;
    layout.dims = dims;
    if (static_cast<std::size_t>(dims[1]) == nf &&
        static_cast<std::size_t>(dims[2]) == nf &&
        static_cast<std::size_t>(dims[3]) == nf) {
        layout.mode = FcrLayout::Mode::TimeFirst;
        layout.Nt = static_cast<std::size_t>(dims[0]);
        layout.time_dim = 0;
        return layout;
    }
    if (static_cast<std::size_t>(dims[0]) == nf &&
        static_cast<std::size_t>(dims[1]) == nf &&
        static_cast<std::size_t>(dims[2]) == nf) {
        layout.mode = FcrLayout::Mode::TimeLast;
        layout.Nt = static_cast<std::size_t>(dims[3]);
        layout.time_dim = 3;
        return layout;
    }
    throw std::runtime_error("Unsupported F/C/R dims: " + dims_to_string(dims));
}

std::complex<double> fcr_value(const FcrLayout& layout,
                               const std::vector<std::complex<double>>& data,
                               std::size_t tidx,
                               std::size_t i,
                               std::size_t j,
                               std::size_t k) {
    if (layout.mode == FcrLayout::Mode::TimeFirst) {
        return at_colmajor_4d(data, layout.dims, tidx, i, j, k);
    }
    return at_colmajor_4d(data, layout.dims, i, j, k, tidx);
}

long long detect_ij_base(const std::vector<long long>& map_ij, std::size_t nf) {
    if (map_ij.empty()) return 0;
    long long min_ij = *std::min_element(map_ij.begin(), map_ij.end());
    long long max_ij = *std::max_element(map_ij.begin(), map_ij.end());
    if (min_ij >= 1 && max_ij <= static_cast<long long>(nf)) return 1;
    if (min_ij >= 0 && max_ij <= static_cast<long long>(nf) - 1) return 0;
    // Heuristic for 1-based with trailing zeros (or 0-based with sparse negatives not expected)
    if (min_ij == 0 && max_ij == static_cast<long long>(nf)) return 1;
    return -1;
}

std::string trim_copy(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
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

struct DatasetInfo {
    std::string path;
    std::vector<hsize_t> dims;
};

herr_t list_cb(hid_t /*obj*/, const char* name, const H5O_info2_t* info, void* op_data) {
    if (!op_data || !name || !info) return 0;
    auto* out = static_cast<std::vector<std::string>*>(op_data);
    if (info->type == H5O_TYPE_DATASET) {
        out->emplace_back(std::string("/") + name);
    }
    return 0;
}

std::vector<DatasetInfo> list_datasets(hid_t file_id) {
    std::vector<std::string> names;
    if (H5Ovisit3(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, list_cb, &names, H5O_INFO_BASIC) < 0) {
        throw std::runtime_error("Failed to list HDF5 datasets");
    }
    std::vector<DatasetInfo> out;
    out.reserve(names.size());
    for (const auto& path : names) {
        DatasetInfo info;
        info.path = path;
        info.dims = get_dataset_dims(file_id, path);
        out.push_back(std::move(info));
    }
    return out;
}

struct ErrSummary {
    double max_abs{0.0};
    double max_rel{0.0};
    bool ok{true};
};

void update_err(ErrSummary& s,
                const std::complex<double>& got,
                const std::complex<double>& expect,
                double atol,
                double rtol) {
    const double diff = std::abs(got - expect);
    const double rel = diff / std::max(1.0, std::abs(expect));
    s.max_abs = std::max(s.max_abs, diff);
    s.max_rel = std::max(s.max_rel, rel);
    if (diff > atol + rtol * std::max(1.0, std::abs(expect))) s.ok = false;
}

void print_usage() {
    std::cout
        << "Usage: tcl4_h5_compare.exe [--file=PATH] [--tidx=LIST] [--one-based]\n"
        << "                           [--method=direct|convolution] [--atol=VAL] [--rtol=VAL]\n"
        << "                           [--compare-fcr] [--nt=COUNT] [--list]\n"
        << "Defaults: file=tests/tcl_test.h5, tidx=0,mid,last, method=convolution\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string file = "tests/tcl_test.h5";
    std::vector<std::string> tidx_tokens;
    bool one_based = false;
    bool compare_fcr = false;
    double atol = 1e-8;
    double rtol = 1e-6;
    std::size_t max_steps = 0;
    bool list_only = false;
    auto method = taco::tcl4::FCRMethod::Convolution;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        if (arg.rfind("--file=", 0) == 0) {
            file = arg.substr(7);
            continue;
        }
        if (arg.rfind("--tidx=", 0) == 0) {
            tidx_tokens = split_csv(arg.substr(7));
            continue;
        }
        if (arg == "--one-based") {
            one_based = true;
            continue;
        }
        if (arg.rfind("--method=", 0) == 0) {
            const std::string val = arg.substr(9);
            if (val == "direct") method = taco::tcl4::FCRMethod::Direct;
            else method = taco::tcl4::FCRMethod::Convolution;
            continue;
        }
        if (arg.rfind("--atol=", 0) == 0) {
            atol = std::stod(arg.substr(7));
            continue;
        }
        if (arg.rfind("--rtol=", 0) == 0) {
            rtol = std::stod(arg.substr(7));
            continue;
        }
        if (arg == "--compare-fcr") {
            compare_fcr = true;
            continue;
        }
        if (arg.rfind("--nt=", 0) == 0) {
            max_steps = static_cast<std::size_t>(std::stoull(arg.substr(5)));
            continue;
        }
        if (arg == "--list") {
            list_only = true;
            continue;
        }
        std::cerr << "Unknown arg: " << arg << "\n";
        print_usage();
        return 2;
    }

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    try {
        H5File h5(file);
        if (list_only) {
            auto datasets = list_datasets(h5.id);
            std::sort(datasets.begin(), datasets.end(),
                      [](const DatasetInfo& a, const DatasetInfo& b) { return a.path < b.path; });
            for (const auto& d : datasets) {
                std::cout << d.path << " dims=" << dims_to_string(d.dims) << "\n";
            }
            return 0;
        }
        if (!dataset_exists(h5.id, "/params/dt")) {
            throw std::runtime_error("Missing /params/dt in HDF5");
        }

        const double dt = read_scalar_double(h5.id, "/params/dt");

        std::vector<hsize_t> dims_H;
        auto Hc = read_complex_array(h5.id, "/system/H", &dims_H);
        auto H_dims = squeeze_dims(dims_H);
        if (H_dims.size() != 2) throw std::runtime_error("system/H is not 2D");
        Eigen::MatrixXcd H = to_matrix_colmajor(Hc, H_dims[0], H_dims[1]);

        std::vector<hsize_t> dims_A;
        auto Ac = read_complex_array(h5.id, "/system/A", &dims_A);
        auto A_dims = squeeze_dims(dims_A);
        if (A_dims.size() != 2) throw std::runtime_error("system/A is not 2D");
        Eigen::MatrixXcd A_eig = to_matrix_colmajor(Ac, A_dims[0], A_dims[1]);

        Eigen::VectorXd eig_vals;
        if (dataset_exists(h5.id, "/system/Eig/re")) {
            std::vector<hsize_t> dims_E;
            auto Ec = read_complex_array(h5.id, "/system/Eig", &dims_E);
            auto E_dims = squeeze_dims(dims_E);
            const std::size_t En = numel(E_dims);
            eig_vals.resize(static_cast<Eigen::Index>(En));
            for (std::size_t i = 0; i < En; ++i) eig_vals(static_cast<Eigen::Index>(i)) = Ec[i].real();
        }

        std::vector<double> map_omegas;
        std::vector<long long> map_ij;
        bool have_map = false;
        if (dataset_exists(h5.id, "/map/omegas") && dataset_exists(h5.id, "/map/ij")) {
            std::vector<hsize_t> dims_om;
            map_omegas = read_array<double>(h5.id, "/map/omegas", H5T_NATIVE_DOUBLE, &dims_om);
            std::vector<hsize_t> dims_ij;
            map_ij = read_array<long long>(h5.id, "/map/ij", H5T_NATIVE_LLONG, &dims_ij);
            have_map = !map_omegas.empty() && !map_ij.empty();
        }

        taco::sys::System system;
        if (eig_vals.size() > 0) {
            const std::size_t En = static_cast<std::size_t>(eig_vals.size());
            if (H.rows() != static_cast<Eigen::Index>(En) || H.cols() != static_cast<Eigen::Index>(En)) {
                throw std::runtime_error("system/Eig size does not match system/H");
            }
            system.eig.dim = En;
            system.eig.eps = eig_vals;
            system.eig.U = Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(En), static_cast<Eigen::Index>(En));
            system.eig.U_dag = system.eig.U;
        } else {
            system.eig = taco::sys::Eigensystem(H);
        }
        system.bf = taco::sys::BohrFrequencies(system.eig.eps);
        if (have_map) {
            const std::size_t N = static_cast<std::size_t>(H.rows());
            const std::size_t nf = map_omegas.size();
            if (map_ij.size() != N * N) {
                throw std::runtime_error("map/ij length does not match N^2");
            }
            long long base = detect_ij_base(map_ij, nf);
            if (base < 0) {
                std::ostringstream msg;
                msg << "map/ij values must be 0..nf-1 or 1..nf (min="
                    << *std::min_element(map_ij.begin(), map_ij.end())
                    << ", max=" << *std::max_element(map_ij.begin(), map_ij.end())
                    << ", nf=" << nf << ")";
                throw std::runtime_error(msg.str());
            }

            taco::sys::FrequencyIndex fidx;
            fidx.tol = 1e-9;
            fidx.buckets.resize(nf);
            for (std::size_t b = 0; b < nf; ++b) {
                fidx.buckets[b].omega = map_omegas[b];
            }
            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t k = 0; k < N; ++k) {
                    const std::size_t idx = j + N * k;
                    const long long b = map_ij[idx] - base;
                    if (b < 0 || b >= static_cast<long long>(nf)) {
                        std::ostringstream msg;
                        msg << "map/ij bucket index out of range at (" << j << "," << k
                            << "): value=" << map_ij[idx] << ", base=" << base
                            << ", nf=" << nf;
                        throw std::runtime_error(msg.str());
                    }
                    fidx.buckets[static_cast<std::size_t>(b)].pairs.emplace_back(static_cast<int>(j),
                                                                                 static_cast<int>(k));
                }
            }
            system.fidx = std::move(fidx);
        } else {
            system.fidx = taco::sys::build_frequency_buckets(system.bf, 1e-9);
        }
        system.A_eig = {A_eig};
        system.A_lab = {system.eig.to_lab(A_eig)};
        system.A_eig_parts = taco::sys::decompose_operators_by_frequency(system.A_eig, system.bf, system.fidx);

        std::vector<double> omegas;
        if (have_map) {
            omegas = map_omegas;
        } else {
            omegas.reserve(system.fidx.buckets.size());
            for (const auto& b : system.fidx.buckets) omegas.push_back(b.omega);
        }

        const std::size_t N = static_cast<std::size_t>(H.rows());
        const std::size_t N2 = N * N;

        auto dims_c_re_raw = get_dataset_dims(h5.id, "/bath/C/re");
        std::size_t c_time_dim = 0;
        if (dims_c_re_raw.size() == 2 && dims_c_re_raw[0] == 1 && dims_c_re_raw[1] > 1) {
            c_time_dim = 1;
        }
        const std::size_t Nt_c = static_cast<std::size_t>(dims_c_re_raw.empty() ? 0 : dims_c_re_raw[c_time_dim]);
        if (Nt_c == 0) throw std::runtime_error("Empty /bath/C");

        auto dims_gw_re_raw = get_dataset_dims(h5.id, "/out/GW_flat/re");
        auto gw_dims_base = squeeze_dims(dims_gw_re_raw);
        auto gw_map = squeezed_index_map(dims_gw_re_raw);
        GwLayout gw_layout = infer_gw_layout(gw_dims_base, N2);
        const std::size_t Nt_gw = gw_layout.Nt;

        bool have_fcr = false;
        FcrLayout fcr_layout;
        std::size_t Nt_fcr = Nt_c;
        if (compare_fcr && dataset_exists(h5.id, "/kernels/F_all/re") &&
            dataset_exists(h5.id, "/kernels/C_all/re") &&
            dataset_exists(h5.id, "/kernels/R_all/re")) {
            auto dims_f_raw = get_dataset_dims(h5.id, "/kernels/F_all/re");
            auto dims_c2_raw = get_dataset_dims(h5.id, "/kernels/C_all/re");
            auto dims_r_raw = get_dataset_dims(h5.id, "/kernels/R_all/re");
            auto dims_f = squeeze_dims(dims_f_raw);
            auto dims_c2 = squeeze_dims(dims_c2_raw);
            auto dims_r = squeeze_dims(dims_r_raw);
            if (dims_f == dims_c2 && dims_f == dims_r) {
                fcr_layout = infer_fcr_layout(dims_f, omegas.size());
                Nt_fcr = fcr_layout.Nt;
                have_fcr = true;
                fcr_layout.time_dim = squeezed_index_map(dims_f_raw)[fcr_layout.time_dim];
            }
        }

        std::size_t Nt_avail = std::min(Nt_c, Nt_gw);
        if (compare_fcr && have_fcr) Nt_avail = std::min(Nt_avail, Nt_fcr);
        if (max_steps > 0) Nt_avail = std::min(Nt_avail, max_steps);
        if (Nt_avail == 0) throw std::runtime_error("No overlapping time samples to compare");
        if (Nt_avail < Nt_c || Nt_avail < Nt_gw || (compare_fcr && have_fcr && Nt_avail < Nt_fcr)) {
            std::cout << "Using Nt=" << Nt_avail << " (C=" << Nt_c << ", GW=" << Nt_gw;
            if (compare_fcr && have_fcr) std::cout << ", FCR=" << Nt_fcr;
            std::cout << ")\n";
        }

        std::vector<std::string> tokens = tidx_tokens;
        if (tokens.empty()) {
            tokens = {"0", "mid", "last"};
        }
        std::vector<std::size_t> tidx_list;
        tidx_list.reserve(tokens.size());
        for (const auto& tok : tokens) {
            if (tok == "last") {
                tidx_list.push_back(Nt_avail - 1);
            } else if (tok == "mid") {
                tidx_list.push_back(Nt_avail / 2);
            } else {
                const long long v = std::stoll(tok);
                if (v < 0) throw std::runtime_error("Negative tidx: " + tok);
                tidx_list.push_back(static_cast<std::size_t>(v));
            }
        }
        if (one_based) {
            for (auto& t : tidx_list) {
                if (t == 0) throw std::runtime_error("one-based tidx must be >= 1");
                t -= 1;
            }
        }
        for (auto t : tidx_list) {
            if (t >= Nt_avail) throw std::runtime_error("tidx out of range");
        }

        const std::size_t Nt_use = Nt_avail;

        std::vector<hsize_t> dims_c;
        auto Cc = read_complex_array_prefix_dim(h5.id, "/bath/C", Nt_use, c_time_dim, &dims_c);

        std::vector<double> tvals;
        if (dataset_exists(h5.id, "/time/t")) {
            std::vector<hsize_t> dims_t;
            auto dims_t_raw = get_dataset_dims(h5.id, "/time/t");
            std::size_t t_time_dim = 0;
            if (dims_t_raw.size() == 2 && dims_t_raw[0] == 1 && dims_t_raw[1] > 1) {
                t_time_dim = 1;
            }
            tvals = read_array_prefix_dim<double>(h5.id, "/time/t", H5T_NATIVE_DOUBLE, Nt_use, t_time_dim, &dims_t);
        }

        std::vector<hsize_t> dims_gw;
        const std::size_t gw_time_dim_raw = gw_map[gw_layout.time_dim];
        auto GW_flat = read_complex_array_prefix_dim(h5.id, "/out/GW_flat", Nt_use, gw_time_dim_raw, &dims_gw);
        gw_layout.dims = squeeze_dims(dims_gw);
        gw_layout.Nt = Nt_use;

        auto gamma_series = taco::gamma::compute_trapz_prefix_multi_matrix(Cc, dt, omegas);
        auto kernels = taco::tcl4::compute_triple_kernels(system, gamma_series, dt, 2, method);
        auto map = taco::tcl4::build_map(system, {});

        int failures = 0;
        for (std::size_t tidx : tidx_list) {
            const double tval = (!tvals.empty() && tidx < tvals.size()) ? tvals[tidx] : (dt * static_cast<double>(tidx));
            auto mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
            Eigen::MatrixXcd GW = taco::tcl4::assemble_liouvillian(mikx, system.A_eig);

            ErrSummary stat;
            for (int r = 0; r < GW.rows(); ++r) {
                for (int c = 0; c < GW.cols(); ++c) {
                    const auto expect = gw_value(gw_layout, GW_flat, tidx,
                                                 static_cast<std::size_t>(r),
                                                 static_cast<std::size_t>(c),
                                                 N2);
                    update_err(stat, GW(r, c), expect, atol, rtol);
                }
            }
            const bool ok = stat.ok;
            if (!ok) failures++;
            std::cout << "GW tidx=" << tidx << " t=" << tval
                      << " max_abs=" << stat.max_abs
                      << " max_rel=" << stat.max_rel
                      << (ok ? " ok\n" : " FAIL\n");
        }

        if (compare_fcr) {
            if (have_fcr) {
                std::vector<hsize_t> dimsF;
                auto F_all = read_complex_array_prefix_dim(h5.id, "/kernels/F_all", Nt_use,
                                                           fcr_layout.time_dim, &dimsF);
                std::vector<hsize_t> dimsC;
                auto C_all = read_complex_array_prefix_dim(h5.id, "/kernels/C_all", Nt_use,
                                                           fcr_layout.time_dim, &dimsC);
                std::vector<hsize_t> dimsR;
                auto R_all = read_complex_array_prefix_dim(h5.id, "/kernels/R_all", Nt_use,
                                                           fcr_layout.time_dim, &dimsR);
                if (dimsF == dimsC && dimsF == dimsR) {
                    FcrLayout fcr_use = fcr_layout;
                    fcr_use.dims = squeeze_dims(dimsF);
                    const std::size_t nf = omegas.size();
                    auto compare_kernel = [&](const char* name,
                                              const std::vector<std::complex<double>>& data,
                                              auto getter) {
                        ErrSummary stat;
                        for (std::size_t tidx : tidx_list) {
                            for (std::size_t i = 0; i < nf; ++i) {
                                for (std::size_t j = 0; j < nf; ++j) {
                                    for (std::size_t k = 0; k < nf; ++k) {
                                        const auto expect = fcr_value(fcr_use, data, tidx, i, j, k);
                                        const auto got = getter(i, j, k, tidx);
                                        update_err(stat, got, expect, atol, rtol);
                                    }
                                }
                            }
                        }
                        const bool ok = stat.ok;
                        if (!ok) failures++;
                        std::cout << name << " max_abs=" << stat.max_abs
                                  << " max_rel=" << stat.max_rel
                                  << (ok ? " ok\n" : " FAIL\n");
                    };
                    compare_kernel("F", F_all,
                                   [&](std::size_t i, std::size_t j, std::size_t k, std::size_t t) {
                                       return kernels.F[i][j][k](static_cast<Eigen::Index>(t));
                                   });
                    compare_kernel("C", C_all,
                                   [&](std::size_t i, std::size_t j, std::size_t k, std::size_t t) {
                                       return kernels.C[i][j][k](static_cast<Eigen::Index>(t));
                                   });
                    compare_kernel("R", R_all,
                                   [&](std::size_t i, std::size_t j, std::size_t k, std::size_t t) {
                                       return kernels.R[i][j][k](static_cast<Eigen::Index>(t));
                                   });
                } else {
                    std::cerr << "F/C/R dataset dims mismatch; skipping FCR compare\n";
                }
            } else {
                std::cerr << "F/C/R datasets not found or unsupported layout; skipping FCR compare\n";
            }
        }

        if (failures) {
            std::cerr << "FAILED: " << failures << " comparison(s)\n";
            return 1;
        }
        std::cout << "All comparisons passed.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
