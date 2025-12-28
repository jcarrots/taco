#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "taco/system.hpp"
#include "taco/tcl4.hpp"
#include "taco/tcl4_assemble.hpp"
#include "taco/tcl4_mikx.hpp"

#include "hdf5.h"

namespace {

using cd = std::complex<double>;

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

std::vector<cd> read_complex_array(hid_t file_id,
                                   const std::string& base,
                                   std::vector<hsize_t>* dims_out) {
    std::vector<hsize_t> dims_re;
    std::vector<hsize_t> dims_im;
    auto re = read_array<double>(file_id, base + "/re", H5T_NATIVE_DOUBLE, &dims_re);
    auto im = read_array<double>(file_id, base + "/im", H5T_NATIVE_DOUBLE, &dims_im);
    if (dims_re != dims_im) throw std::runtime_error("Complex dims mismatch at: " + base);
    if (re.size() != im.size()) throw std::runtime_error("Complex size mismatch at: " + base);
    std::vector<cd> out;
    out.reserve(re.size());
    for (std::size_t i = 0; i < re.size(); ++i) out.emplace_back(re[i], im[i]);
    if (dims_out) *dims_out = dims_re;
    return out;
}

double read_scalar_double(hid_t file_id, const std::string& path) {
    std::vector<hsize_t> dims;
    auto data = read_array<double>(file_id, path, H5T_NATIVE_DOUBLE, &dims);
    if (data.size() != 1) throw std::runtime_error("Expected scalar at: " + path);
    return data[0];
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

std::complex<double> at_colmajor_2d(const std::vector<cd>& data,
                                    const std::vector<hsize_t>& dims,
                                    std::size_t row,
                                    std::size_t col) {
    return data[row + static_cast<std::size_t>(dims[0]) * col];
}

std::complex<double> at_colmajor_3d(const std::vector<cd>& data,
                                    const std::vector<hsize_t>& dims,
                                    std::size_t i0,
                                    std::size_t i1,
                                    std::size_t i2) {
    const std::size_t d0 = static_cast<std::size_t>(dims[0]);
    const std::size_t d1 = static_cast<std::size_t>(dims[1]);
    return data[i0 + d0 * (i1 + d1 * i2)];
}

std::complex<double> at_colmajor_4d(const std::vector<cd>& data,
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

std::complex<double> at_rowmajor_4d(const std::vector<cd>& data,
                                    const std::vector<hsize_t>& dims,
                                    std::size_t i0,
                                    std::size_t i1,
                                    std::size_t i2,
                                    std::size_t i3) {
    const std::size_t d1 = static_cast<std::size_t>(dims[1]);
    const std::size_t d2 = static_cast<std::size_t>(dims[2]);
    const std::size_t d3 = static_cast<std::size_t>(dims[3]);
    return data[(((i0 * d1) + i1) * d2 + i2) * d3 + i3];
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
    const std::size_t N = C.size();
    const std::size_t M = omegas.size();
    if (N == 0 || M == 0 || !(dt > 0.0)) return Eigen::MatrixXcd();
    Eigen::MatrixXcd G(static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(M));
    G.row(0).setZero();
    const cd half_dt(dt / 2.0, 0.0);
    for (std::size_t j = 0; j < M; ++j) {
        const cd step = std::exp(cd{0.0, omegas[j] * dt});
        cd phi(1.0, 0.0);
        cd acc(0.0, 0.0);
        if (rule == GammaRule::Rect) {
            acc += dt * phi * C[0];
            G(static_cast<Eigen::Index>(0), static_cast<Eigen::Index>(j)) = acc;
        }
        for (std::size_t k = 1; k < N; ++k) {
            const cd phi_next = phi * step;
            if (rule == GammaRule::Trapz) {
                acc += half_dt * (phi * C[k - 1] + phi_next * C[k]);
            } else {
                acc += dt * phi_next * C[k];
            }
            G(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(j)) = acc;
            phi = phi_next;
        }
    }
    return G;
}

long long detect_ij_base(const std::vector<long long>& map_ij, std::size_t nf) {
    if (map_ij.empty()) return 0;
    const long long min_ij = *std::min_element(map_ij.begin(), map_ij.end());
    const long long max_ij = *std::max_element(map_ij.begin(), map_ij.end());
    if (min_ij >= 1 && max_ij <= static_cast<long long>(nf)) return 1;
    if (min_ij >= 0 && max_ij <= static_cast<long long>(nf) - 1) return 0;
    if (min_ij == 0 && max_ij == static_cast<long long>(nf)) return 1;
    return -1;
}

struct MatrixSeriesLayout {
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
    bool row_major_flat{true};
    bool transpose{false};
};

MatrixSeriesLayout infer_matrix_series_layout(const std::vector<hsize_t>& dims,
                                              std::size_t rows,
                                              std::size_t cols,
                                              bool row_major_flat,
                                              int time_dim) {
    const std::size_t flat_len = rows * cols;
    MatrixSeriesLayout layout;
    layout.dims = dims;
    layout.row_major_flat = row_major_flat;
    layout.transpose = false;
    if (dims.size() == 2) {
        const std::size_t d0 = static_cast<std::size_t>(dims[0]);
        const std::size_t d1 = static_cast<std::size_t>(dims[1]);
        if (time_dim == 0) {
            layout.mode = MatrixSeriesLayout::Mode::FlatTimeRow;
            if (d1 == flat_len) {
                layout.Nt = d0;
                return layout;
            }
            if (d0 == flat_len) {
                layout.Nt = d1;
                layout.transpose = true;
                return layout;
            }
            throw std::runtime_error("Flat dataset has unexpected dims for time=row: " +
                                     dims_to_string(dims));
        }
        if (time_dim == 1) {
            layout.mode = MatrixSeriesLayout::Mode::FlatTimeCol;
            if (d0 == flat_len) {
                layout.Nt = d1;
                return layout;
            }
            if (d1 == flat_len) {
                layout.Nt = d0;
                layout.transpose = true;
                return layout;
            }
            throw std::runtime_error("Flat dataset has unexpected dims for time=col: " +
                                     dims_to_string(dims));
        }
        if (d1 == flat_len) {
            layout.mode = MatrixSeriesLayout::Mode::FlatTimeRow;
            layout.Nt = d0;
            return layout;
        }
        if (d0 == flat_len) {
            layout.mode = MatrixSeriesLayout::Mode::FlatTimeCol;
            layout.Nt = d1;
            return layout;
        }
    } else if (dims.size() == 3) {
        if (time_dim >= 0) {
            throw std::runtime_error("time=row/col override only applies to flat datasets");
        }
        const std::size_t d0 = static_cast<std::size_t>(dims[0]);
        const std::size_t d1 = static_cast<std::size_t>(dims[1]);
        const std::size_t d2 = static_cast<std::size_t>(dims[2]);
        if (d0 == rows && d1 == cols) {
            layout.mode = MatrixSeriesLayout::Mode::MatTimeLast;
            layout.Nt = d2;
            return layout;
        }
        if (d1 == rows && d2 == cols) {
            layout.mode = MatrixSeriesLayout::Mode::MatTimeFirst;
            layout.Nt = d0;
            return layout;
        }
        if (d0 == rows && d2 == cols) {
            layout.mode = MatrixSeriesLayout::Mode::MatTimeMiddle;
            layout.Nt = d1;
            return layout;
        }
    }
    throw std::runtime_error("Unsupported matrix dims: " + dims_to_string(dims));
}

cd matrix_series_value(const MatrixSeriesLayout& layout,
                       const std::vector<cd>& data,
                       std::size_t tidx,
                       std::size_t row,
                       std::size_t col,
                       std::size_t rows,
                       std::size_t cols) {
    const std::size_t flat_idx = layout.row_major_flat ? (row * cols + col)
                                                       : (col * rows + row);
    std::vector<hsize_t> dims_use = layout.dims;
    if (layout.transpose && dims_use.size() == 2) {
        std::swap(dims_use[0], dims_use[1]);
    }
    switch (layout.mode) {
        case MatrixSeriesLayout::Mode::FlatTimeRow:
            return at_colmajor_2d(data, dims_use, tidx, flat_idx);
        case MatrixSeriesLayout::Mode::FlatTimeCol:
            return at_colmajor_2d(data, dims_use, flat_idx, tidx);
        case MatrixSeriesLayout::Mode::MatTimeLast:
            return at_colmajor_3d(data, dims_use, row, col, tidx);
        case MatrixSeriesLayout::Mode::MatTimeFirst:
            return at_colmajor_3d(data, dims_use, tidx, row, col);
        case MatrixSeriesLayout::Mode::MatTimeMiddle:
            return at_colmajor_3d(data, dims_use, row, tidx, col);
    }
    return cd{0.0, 0.0};
}

struct KernelSeriesLayout {
    enum class TimeMode {
        First,
        Last
    };
    std::vector<hsize_t> dims;
    std::size_t Nt{0};
    TimeMode time_mode{TimeMode::Last};
    bool transpose{false};
    bool row_major{false};
};

KernelSeriesLayout infer_kernel_series_layout(const std::vector<hsize_t>& dims,
                                              std::size_t nf,
                                              int time_dim,
                                              bool row_major) {
    if (dims.size() != 4) {
        throw std::runtime_error("Kernel dataset must be rank-4: " + dims_to_string(dims));
    }
    const std::size_t d0 = static_cast<std::size_t>(dims[0]);
    const std::size_t d1 = static_cast<std::size_t>(dims[1]);
    const std::size_t d2 = static_cast<std::size_t>(dims[2]);
    const std::size_t d3 = static_cast<std::size_t>(dims[3]);
    KernelSeriesLayout layout;
    layout.dims = dims;
    layout.transpose = false;
    layout.row_major = row_major;
    if (time_dim == 0) {
        layout.time_mode = KernelSeriesLayout::TimeMode::First;
        if (d1 == nf && d2 == nf && d3 == nf) {
            layout.Nt = d0;
            return layout;
        }
        if (d0 == nf && d1 == nf && d2 == nf) {
            layout.Nt = d3;
            layout.transpose = true;
            return layout;
        }
        throw std::runtime_error("Kernel dims do not match nf for time=first: " + dims_to_string(dims));
    }
    if (time_dim == 1) {
        layout.time_mode = KernelSeriesLayout::TimeMode::Last;
        if (d0 == nf && d1 == nf && d2 == nf) {
            layout.Nt = d3;
            return layout;
        }
        if (d1 == nf && d2 == nf && d3 == nf) {
            layout.Nt = d0;
            layout.transpose = true;
            return layout;
        }
        throw std::runtime_error("Kernel dims do not match nf for time=last: " + dims_to_string(dims));
    }
    if (d0 == nf && d1 == nf && d2 == nf) {
        layout.time_mode = KernelSeriesLayout::TimeMode::Last;
        layout.Nt = d3;
        return layout;
    }
    if (d1 == nf && d2 == nf && d3 == nf) {
        layout.time_mode = KernelSeriesLayout::TimeMode::First;
        layout.Nt = d0;
        return layout;
    }
    throw std::runtime_error("Kernel dims do not match nf: " + dims_to_string(dims));
}

std::vector<hsize_t> reverse_dims_copy(const std::vector<hsize_t>& dims) {
    std::vector<hsize_t> out = dims;
    std::reverse(out.begin(), out.end());
    return out;
}

cd kernel_series_value(const KernelSeriesLayout& layout,
                       const std::vector<cd>& data,
                       std::size_t tidx,
                       std::size_t i,
                       std::size_t j,
                       std::size_t k) {
    const auto dims_use = layout.transpose ? reverse_dims_copy(layout.dims) : layout.dims;
    switch (layout.time_mode) {
        case KernelSeriesLayout::TimeMode::Last:
            return layout.row_major
                ? at_rowmajor_4d(data, dims_use, i, j, k, tidx)
                : at_colmajor_4d(data, dims_use, i, j, k, tidx);
        case KernelSeriesLayout::TimeMode::First:
            return layout.row_major
                ? at_rowmajor_4d(data, dims_use, tidx, i, j, k)
                : at_colmajor_4d(data, dims_use, tidx, i, j, k);
    }
    return cd{0.0, 0.0};
}

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
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
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
            std::ostringstream msg;
            msg << "tidx out of range: " << idx << " (Nt=" << Nt << ")";
            throw std::runtime_error(msg.str());
        }
        if (std::find(out.begin(), out.end(), idx) == out.end()) {
            out.push_back(idx);
        }
    }
    return out;
}

std::vector<int> build_pair_to_bucket_by_omega(const taco::sys::BohrFrequencies& bf,
                                               const std::vector<double>& omegas,
                                               std::size_t N,
                                               double tol,
                                               bool row_major_flat) {
    const std::size_t N2 = N * N;
    std::vector<int> pair_to_bucket(N2, -1);
    std::vector<char> used(omegas.size(), 0);
    auto assign_bucket = [&](std::size_t idx, double w) {
        int first_match = -1;
        int free_match = -1;
        for (std::size_t b = 0; b < omegas.size(); ++b) {
            if (std::abs(omegas[b] - w) <= tol) {
                if (first_match < 0) first_match = static_cast<int>(b);
                if (!used[b]) {
                    free_match = static_cast<int>(b);
                    break;
                }
            }
        }
        int pick = (free_match >= 0) ? free_match : first_match;
        if (pick >= 0) {
            pair_to_bucket[idx] = pick;
            if (!used[static_cast<std::size_t>(pick)]) used[static_cast<std::size_t>(pick)] = 1;
        }
    };

    if (row_major_flat) {
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t k = 0; k < N; ++k) {
                const std::size_t idx = j + N * k;
                assign_bucket(idx, bf.omega(static_cast<int>(j), static_cast<int>(k)));
            }
        }
    } else {
        for (std::size_t k = 0; k < N; ++k) {
            for (std::size_t j = 0; j < N; ++j) {
                const std::size_t idx = j + N * k;
                assign_bucket(idx, bf.omega(static_cast<int>(j), static_cast<int>(k)));
            }
        }
    }
    return pair_to_bucket;
}

std::vector<int> build_omega_index_map(const std::vector<double>& from_omegas,
                                       const std::vector<double>& to_omegas,
                                       double tol) {
    std::vector<int> out(from_omegas.size(), -1);
    std::vector<char> used(to_omegas.size(), 0);
    for (std::size_t i = 0; i < from_omegas.size(); ++i) {
        int first_match = -1;
        int free_match = -1;
        for (std::size_t j = 0; j < to_omegas.size(); ++j) {
            if (std::abs(from_omegas[i] - to_omegas[j]) <= tol) {
                if (first_match < 0) first_match = static_cast<int>(j);
                if (!used[j]) {
                    free_match = static_cast<int>(j);
                    break;
                }
            }
        }
        int pick = (free_match >= 0) ? free_match : first_match;
        if (pick >= 0) {
            out[i] = pick;
            if (!used[static_cast<std::size_t>(pick)]) used[static_cast<std::size_t>(pick)] = 1;
        }
    }
    return out;
}

taco::sys::FrequencyIndex build_frequency_index_from_map(const std::vector<double>& omegas,
                                                         const std::vector<long long>& map_ij,
                                                         long long map_base,
                                                         std::size_t N,
                                                         double tol) {
    taco::sys::FrequencyIndex fidx;
    fidx.tol = std::max(1e-12, tol);
    fidx.buckets.resize(omegas.size());
    for (std::size_t b = 0; b < omegas.size(); ++b) {
        fidx.buckets[b].omega = omegas[b];
    }
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < N; ++k) {
            const std::size_t idx = j + N * k;
            const long long b = map_ij[idx] - map_base;
            if (b < 0 || b >= static_cast<long long>(omegas.size())) {
                throw std::runtime_error("map/ij index out of range for omegas");
            }
            fidx.buckets[static_cast<std::size_t>(b)].pairs.emplace_back(static_cast<int>(j),
                                                                         static_cast<int>(k));
        }
    }
    return fidx;
}

struct Options {
    std::string file{"tests/tcl_test.h5"};
    std::string tidx_list{"0,mid,last"};
    bool one_based{false};
    bool compare_gt{false};
    bool compare_gw{false};
    bool compare_fcr{false};
    bool compare_fcr_methods{false};
    bool list{false};
    bool dump_map{false};
    bool print_gt{false};
    bool print_gw{false};
    bool print_fcr{false};
    bool print_mikx{false};
    std::size_t fcr_fft_pad{0};
    enum class OmegaOrder {
        File,
        Sorted
    };
    enum class MikxSource {
        Computed,
        File
    };
    OmegaOrder fcr_omega_order{OmegaOrder::File};
    OmegaOrder gt_omega_order{OmegaOrder::File};
    MikxSource mikx_source{MikxSource::Computed};
    std::array<int, 3> fcr_axes{0, 1, 2};
    bool fcr_row_major{false};
    enum class FcrWhich {
        All,
        F,
        C,
        R
    };
    FcrWhich fcr_which{FcrWhich::All};
    bool fcr_filter_enabled{false};
    bool fcr_filter_by_omega{false};
    std::array<int, 3> fcr_ijk{0, 0, 0};
    std::array<double, 3> fcr_omega{0.0, 0.0, 0.0};
    std::size_t fcr_nt{0};
    std::size_t fcr_offset{0};
    int fcr_time_dim{-1};          // -1=auto, 0=first, 1=last
    double atol{1e-6};
    double rtol{1e-6};
    std::size_t gt_offset{0};
    bool gt_row_major_flat{false};  // column-major flatten by default
    bool gw_row_major_flat{false};  // column-major flatten by default
    int gt_time_dim{0};             // 0=row, 1=col, -1=auto
    int gw_time_dim{0};             // 0=row, 1=col, -1=auto
    GammaRule gamma_rule{GammaRule::Rect};
    double gamma_sign{1.0};
    double omega_tol{1e-9};
    enum class GtMapMode {
        Omega,
        Ij
    };
    GtMapMode gt_map_mode{GtMapMode::Omega};
    taco::tcl4::FCRMethod method{taco::tcl4::FCRMethod::Convolution};
};

void print_usage() {
    std::cout
        << "Usage: tcl4_h5_compare.exe [--file=PATH] [--tidx=LIST] [--one-based]\n"
        << "                           [--compare-gt] [--compare-gw] [--compare-fcr]\n"
        << "                           [--compare-fcr-methods] [--gt-offset=N]\n"
        << "                           [--gamma-rule=rect|trapz] [--gamma-sign=+1|-1]\n"
        << "                           [--gt-flat=row|col] [--gt-time=row|col|auto]\n"
        << "                           [--gt-map=omega|ij] [--gt-omega-order=file|sorted]\n"
        << "                           [--omega-tol=VAL]\n"
        << "                           [--gw-flat=row|col] [--gw-time=row|col|auto]\n"
        << "                           [--method=direct|convolution]\n"
        << "                           [--atol=VAL] [--rtol=VAL]\n"
        << "                           [--list] [--dump-map] [--print-gt] [--print-gw] [--print-fcr] [--print-mikx]\n"
        << "                           [--mikx-source=file|computed]\n"
        << "                           [--fcr=all|f|c|r] [--fcr-ijk=i,j,k] [--fcr-omega=w|w1,w2,w3]\n"
        << "                           [--fcr-omega-order=file|sorted] [--fcr-offset=N]\n"
        << "                           [--fcr-time=first|last|auto] [--fcr-axes=0,1,2]\n"
        << "                           [--fcr-order=row|col]\n"
        << "                           [--fcr-nt=COUNT] [--fcr-fft-pad=N]\n"
        << "Defaults: file=tests/tcl_test.h5, tidx=0,mid,last, gamma-rule=rect, gt-time=row, gw-time=row\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options opt;
        bool compare_flag_seen = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
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
            if (arg == "--compare-gt") {
                opt.compare_gt = true;
                compare_flag_seen = true;
                continue;
            }
            if (arg == "--compare-gw") {
                opt.compare_gw = true;
                compare_flag_seen = true;
                continue;
            }
            if (arg == "--compare-fcr") {
                opt.compare_fcr = true;
                compare_flag_seen = true;
                continue;
            }
            if (arg == "--compare-fcr-methods") {
                opt.compare_fcr_methods = true;
                compare_flag_seen = true;
                continue;
            }
            if (arg.rfind("--gt-offset=", 0) == 0) {
                opt.gt_offset = static_cast<std::size_t>(std::stoull(arg.substr(12)));
                continue;
            }
            if (arg.rfind("--gamma-rule=", 0) == 0) {
                const std::string val = arg.substr(13);
                if (val == "trapz") opt.gamma_rule = GammaRule::Trapz;
                else opt.gamma_rule = GammaRule::Rect;
                continue;
            }
            if (arg.rfind("--gamma-sign=", 0) == 0) {
                opt.gamma_sign = std::stod(arg.substr(13));
                if (opt.gamma_sign != 1.0 && opt.gamma_sign != -1.0) {
                    std::cerr << "gamma-sign must be +1 or -1\n";
                    return 1;
                }
                continue;
            }
            if (arg.rfind("--gt-flat=", 0) == 0) {
                const std::string val = arg.substr(10);
                opt.gt_row_major_flat = (val == "row");
                continue;
            }
            if (arg.rfind("--gt-time=", 0) == 0) {
                const std::string val = arg.substr(10);
                if (val == "auto") opt.gt_time_dim = -1;
                else if (val == "col") opt.gt_time_dim = 1;
                else opt.gt_time_dim = 0;
                continue;
            }
            if (arg.rfind("--gt-map=", 0) == 0) {
                const std::string val = arg.substr(9);
                if (val == "ij") opt.gt_map_mode = Options::GtMapMode::Ij;
                else opt.gt_map_mode = Options::GtMapMode::Omega;
                continue;
            }
            if (arg.rfind("--gt-omega-order=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(17));
                if (val == "sorted") opt.gt_omega_order = Options::OmegaOrder::Sorted;
                else opt.gt_omega_order = Options::OmegaOrder::File;
                continue;
            }
            if (arg.rfind("--omega-tol=", 0) == 0) {
                opt.omega_tol = std::stod(arg.substr(12));
                continue;
            }
            if (arg.rfind("--gw-flat=", 0) == 0) {
                const std::string val = arg.substr(10);
                opt.gw_row_major_flat = (val != "col");
                continue;
            }
            if (arg.rfind("--gw-time=", 0) == 0) {
                const std::string val = arg.substr(10);
                if (val == "auto") opt.gw_time_dim = -1;
                else if (val == "col") opt.gw_time_dim = 1;
                else opt.gw_time_dim = 0;
                continue;
            }
            if (arg.rfind("--method=", 0) == 0) {
                const std::string val = arg.substr(9);
                if (val == "direct") opt.method = taco::tcl4::FCRMethod::Direct;
                else opt.method = taco::tcl4::FCRMethod::Convolution;
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
            if (arg == "--list") {
                opt.list = true;
                continue;
            }
            if (arg == "--dump-map") {
                opt.dump_map = true;
                continue;
            }
            if (arg == "--print-gt") {
                opt.print_gt = true;
                continue;
            }
            if (arg == "--print-gw") {
                opt.print_gw = true;
                continue;
            }
            if (arg == "--print-fcr") {
                opt.print_fcr = true;
                continue;
            }
            if (arg == "--print-mikx") {
                opt.print_mikx = true;
                compare_flag_seen = true;
                continue;
            }
            if (arg.rfind("--mikx-source=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(14));
                if (val == "file") opt.mikx_source = Options::MikxSource::File;
                else opt.mikx_source = Options::MikxSource::Computed;
                continue;
            }
            if (arg.rfind("--fcr=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(6));
                if (val == "f") opt.fcr_which = Options::FcrWhich::F;
                else if (val == "c") opt.fcr_which = Options::FcrWhich::C;
                else if (val == "r") opt.fcr_which = Options::FcrWhich::R;
                else opt.fcr_which = Options::FcrWhich::All;
                continue;
            }
            if (arg.rfind("--fcr-ijk=", 0) == 0) {
                auto tokens = split_csv(arg.substr(10));
                if (tokens.size() == 1) {
                    const int v = std::stoi(tokens[0]);
                    opt.fcr_ijk = {v, v, v};
                } else if (tokens.size() == 3) {
                    opt.fcr_ijk = {std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2])};
                } else {
                    throw std::runtime_error("fcr-ijk expects 1 or 3 comma-separated values");
                }
                opt.fcr_filter_enabled = true;
                continue;
            }
            if (arg.rfind("--fcr-omega=", 0) == 0) {
                auto tokens = split_csv(arg.substr(12));
                if (tokens.size() == 1) {
                    const double w = std::stod(tokens[0]);
                    opt.fcr_omega = {w, w, w};
                } else if (tokens.size() == 3) {
                    opt.fcr_omega = {std::stod(tokens[0]), std::stod(tokens[1]), std::stod(tokens[2])};
                } else {
                    throw std::runtime_error("fcr-omega expects 1 or 3 comma-separated values");
                }
                opt.fcr_filter_by_omega = true;
                continue;
            }
            if (arg.rfind("--fcr-omega-order=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(18));
                if (val == "sorted") opt.fcr_omega_order = Options::OmegaOrder::Sorted;
                else opt.fcr_omega_order = Options::OmegaOrder::File;
                continue;
            }
            if (arg.rfind("--fcr-offset=", 0) == 0) {
                opt.fcr_offset = static_cast<std::size_t>(std::stoull(arg.substr(13)));
                continue;
            }
            if (arg.rfind("--fcr-time=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(11));
                if (val == "first") opt.fcr_time_dim = 0;
                else if (val == "last") opt.fcr_time_dim = 1;
                else opt.fcr_time_dim = -1;
                continue;
            }
            if (arg.rfind("--fcr-order=", 0) == 0) {
                const std::string val = lower_copy(arg.substr(12));
                opt.fcr_row_major = (val == "row");
                continue;
            }
            if (arg.rfind("--fcr-axes=", 0) == 0) {
                auto tokens = split_csv(arg.substr(11));
                if (tokens.size() != 3) {
                    throw std::runtime_error("fcr-axes expects 3 comma-separated values");
                }
                std::array<int, 3> axes{
                    std::stoi(tokens[0]),
                    std::stoi(tokens[1]),
                    std::stoi(tokens[2])
                };
                std::array<bool, 3> seen{false, false, false};
                for (int v : axes) {
                    if (v < 0 || v > 2) {
                        throw std::runtime_error("fcr-axes values must be 0,1,2");
                    }
                    if (seen[static_cast<std::size_t>(v)]) {
                        throw std::runtime_error("fcr-axes must be a permutation of 0,1,2");
                    }
                    seen[static_cast<std::size_t>(v)] = true;
                }
                opt.fcr_axes = axes;
                continue;
            }
            if (arg.rfind("--fcr-nt=", 0) == 0) {
                opt.fcr_nt = static_cast<std::size_t>(std::stoull(arg.substr(9)));
                continue;
            }
            if (arg.rfind("--fcr-fft-pad=", 0) == 0) {
                opt.fcr_fft_pad = static_cast<std::size_t>(std::stoull(arg.substr(14)));
                continue;
            }
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 1;
        }
        if (!compare_flag_seen) {
            opt.compare_gt = true;
        }

        H5File h5(opt.file);

        if (opt.list) {
            auto info = list_datasets(h5.id);
            for (const auto& d : info) {
                std::cout << d.path << " dims=" << dims_to_string(d.dims) << "\n";
            }
        }

        const double dt = read_scalar_double(h5.id, "/params/dt");

        if (!dataset_exists(h5.id, "/bath/C/re")) {
            throw std::runtime_error("/bath/C not found in file");
        }
        std::vector<hsize_t> dims_c;
        auto Cc = read_complex_array(h5.id, "/bath/C", &dims_c);
        const std::size_t Nt_c = Cc.size();
        if (Nt_c == 0) throw std::runtime_error("Empty /bath/C");

        std::vector<double> tvals;
        if (dataset_exists(h5.id, "/time/t")) {
            std::vector<hsize_t> dims_t;
            tvals = read_array<double>(h5.id, "/time/t", H5T_NATIVE_DOUBLE, &dims_t);
        }

        std::vector<double> map_omegas;
        std::vector<long long> map_ij;
        if (dataset_exists(h5.id, "/map/omegas")) {
            std::vector<hsize_t> dims_om;
            map_omegas = read_array<double>(h5.id, "/map/omegas", H5T_NATIVE_DOUBLE, &dims_om);
        }
        if (dataset_exists(h5.id, "/map/ij")) {
            std::vector<hsize_t> dims_ij;
            map_ij = read_array<long long>(h5.id, "/map/ij", H5T_NATIVE_LLONG, &dims_ij);
        }

        if ((opt.compare_gt || opt.compare_gw || opt.compare_fcr || opt.print_mikx) &&
            (map_omegas.empty() || map_ij.empty())) {
            throw std::runtime_error("map/omegas and map/ij are required for comparison");
        }

        std::size_t N = 0;
        if (dataset_exists(h5.id, "/map/N")) {
            N = static_cast<std::size_t>(read_scalar_double(h5.id, "/map/N"));
        }
        if (N == 0 && dataset_exists(h5.id, "/system/H/re")) {
            auto dims_h = get_dataset_dims(h5.id, "/system/H/re");
            if (dims_h.size() >= 2) N = static_cast<std::size_t>(dims_h[0]);
        }
        if (N == 0) throw std::runtime_error("Failed to determine N");

        const std::size_t N2 = N * N;
        std::size_t nf = map_omegas.size();
        if (dataset_exists(h5.id, "/map/nf")) {
            const auto map_nf = static_cast<std::size_t>(read_scalar_double(h5.id, "/map/nf"));
            if (map_nf > 0) nf = map_nf;
        }
        if (map_omegas.size() < nf) {
            throw std::runtime_error("map/omegas length is smaller than map/nf");
        }
        if (map_omegas.size() > nf) {
            map_omegas.resize(nf);
        }
        if (opt.fcr_filter_by_omega) {
            if (opt.fcr_filter_enabled) {
                throw std::runtime_error("Use either fcr-ijk or fcr-omega, not both");
            }
            auto find_idx = [&](double w) -> int {
                for (std::size_t b = 0; b < map_omegas.size(); ++b) {
                    if (std::abs(map_omegas[b] - w) <= opt.omega_tol) {
                        return static_cast<int>(b);
                    }
                }
                return -1;
            };
            const int i = find_idx(opt.fcr_omega[0]);
            const int j = find_idx(opt.fcr_omega[1]);
            const int k = find_idx(opt.fcr_omega[2]);
            if (i < 0 || j < 0 || k < 0) {
                throw std::runtime_error("fcr-omega not found in map/omegas");
            }
            opt.fcr_ijk = {i, j, k};
            opt.fcr_filter_enabled = true;
        }
        if (map_ij.size() != N2) {
            throw std::runtime_error("map/ij length does not match N^2");
        }
        const long long map_base = detect_ij_base(map_ij, nf);
        if (map_base < 0) {
            std::ostringstream msg;
            msg << "map/ij values must be 0..nf-1 or 1..nf (min="
                << *std::min_element(map_ij.begin(), map_ij.end())
                << ", max=" << *std::max_element(map_ij.begin(), map_ij.end())
                << ", nf=" << nf << ")";
            throw std::runtime_error(msg.str());
        }

        if (opt.dump_map) {
            std::cout << "map.omegas:";
            for (double w : map_omegas) std::cout << " " << w;
            std::cout << "\nmap.ij (raw):\n";
            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t k = 0; k < N; ++k) {
                    const std::size_t idx = j + N * k;
                    std::cout << map_ij[idx] << (k + 1 < N ? " " : "");
                }
                std::cout << "\n";
            }
            std::cout << "map.ij (0-based):\n";
            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t k = 0; k < N; ++k) {
                    const std::size_t idx = j + N * k;
                    std::cout << (map_ij[idx] - map_base) << (k + 1 < N ? " " : "");
                }
                std::cout << "\n";
            }
        }

        std::vector<double> gt_omegas = map_omegas;
        std::vector<int> gt_sorted_to_map;
        std::vector<int> gt_map_to_sorted;
        if (opt.compare_gt) {
            if (opt.gt_omega_order == Options::OmegaOrder::Sorted) {
                std::vector<double> sorted_omegas = map_omegas;
                std::sort(sorted_omegas.begin(), sorted_omegas.end());
                gt_sorted_to_map = build_omega_index_map(sorted_omegas, map_omegas, opt.omega_tol);
                gt_map_to_sorted = build_omega_index_map(map_omegas, sorted_omegas, opt.omega_tol);
                gt_omegas = std::move(sorted_omegas);
            } else {
                gt_sorted_to_map.resize(map_omegas.size());
                gt_map_to_sorted.resize(map_omegas.size());
                for (std::size_t i = 0; i < map_omegas.size(); ++i) {
                    gt_sorted_to_map[i] = static_cast<int>(i);
                    gt_map_to_sorted[i] = static_cast<int>(i);
                }
            }
            for (int idx : gt_sorted_to_map) {
                if (idx < 0) {
                    throw std::runtime_error("Failed to map omega indices for Gt compare");
                }
            }
            for (int idx : gt_map_to_sorted) {
                if (idx < 0) {
                    throw std::runtime_error("Failed to map omega indices for Gt compare");
                }
            }
        }

        std::vector<double> omegas_gamma = map_omegas;
        if (opt.gamma_sign < 0.0) {
            for (auto& w : omegas_gamma) w = -w;
        }
        auto gamma_series = compute_gamma_prefix_matrix(Cc, dt, omegas_gamma, opt.gamma_rule);

        MatrixSeriesLayout gt_layout;
        std::vector<cd> Gt_flat;
        std::size_t Nt_gt = Nt_c;
        if (opt.compare_gt) {
            if (!dataset_exists(h5.id, "/out/Gt_flat/re")) {
                throw std::runtime_error("/out/Gt_flat not found");
            }
            std::vector<hsize_t> dims_gt;
            Gt_flat = read_complex_array(h5.id, "/out/Gt_flat", &dims_gt);
            auto dims_gt_use = squeeze_dims(dims_gt);
            gt_layout = infer_matrix_series_layout(dims_gt_use, N, N,
                                                   opt.gt_row_major_flat, opt.gt_time_dim);
            Nt_gt = gt_layout.Nt;
            if (opt.gt_offset >= Nt_gt) {
                throw std::runtime_error("gt-offset exceeds Gt time length");
            }
        }

        MatrixSeriesLayout gw_layout;
        std::vector<cd> GW_flat;
        std::size_t Nt_gw = Nt_c;
        if (opt.compare_gw) {
            if (!dataset_exists(h5.id, "/out/GW_flat/re")) {
                throw std::runtime_error("/out/GW_flat not found");
            }
            std::vector<hsize_t> dims_gw;
            GW_flat = read_complex_array(h5.id, "/out/GW_flat", &dims_gw);
            auto dims_gw_use = squeeze_dims(dims_gw);
            gw_layout = infer_matrix_series_layout(dims_gw_use, N2, N2,
                                                   opt.gw_row_major_flat, opt.gw_time_dim);
            Nt_gw = gw_layout.Nt;
        }

        KernelSeriesLayout f_layout;
        KernelSeriesLayout c_layout;
        KernelSeriesLayout r_layout;
        std::vector<cd> F_all;
        std::vector<cd> C_all;
        std::vector<cd> R_all;
        std::size_t Nt_fcr = Nt_c;
        std::vector<int> fcr_index_map;
        const bool need_file_kernels =
            opt.compare_fcr || (opt.print_mikx && opt.mikx_source == Options::MikxSource::File);
        if (need_file_kernels) {
            if (!dataset_exists(h5.id, "/kernels/F_all/re") ||
                !dataset_exists(h5.id, "/kernels/C_all/re") ||
                !dataset_exists(h5.id, "/kernels/R_all/re")) {
                throw std::runtime_error("/kernels/F_all, C_all, R_all are required for file kernels");
            }
            std::vector<hsize_t> dims_f;
            std::vector<hsize_t> dims_c2;
            std::vector<hsize_t> dims_r;
            F_all = read_complex_array(h5.id, "/kernels/F_all", &dims_f);
            C_all = read_complex_array(h5.id, "/kernels/C_all", &dims_c2);
            R_all = read_complex_array(h5.id, "/kernels/R_all", &dims_r);
            auto dims_f_use = squeeze_dims(dims_f);
            auto dims_c_use = squeeze_dims(dims_c2);
            auto dims_r_use = squeeze_dims(dims_r);
            if (dims_f_use != dims_c_use || dims_f_use != dims_r_use) {
                throw std::runtime_error("F/C/R kernel dims mismatch");
            }
            f_layout = infer_kernel_series_layout(dims_f_use, nf, opt.fcr_time_dim, opt.fcr_row_major);
            c_layout = infer_kernel_series_layout(dims_c_use, nf, opt.fcr_time_dim, opt.fcr_row_major);
            r_layout = infer_kernel_series_layout(dims_r_use, nf, opt.fcr_time_dim, opt.fcr_row_major);
            Nt_fcr = f_layout.Nt;
            if (opt.fcr_omega_order == Options::OmegaOrder::Sorted) {
                std::vector<double> sorted_omegas = map_omegas;
                std::sort(sorted_omegas.begin(), sorted_omegas.end());
                fcr_index_map = build_omega_index_map(map_omegas, sorted_omegas, opt.omega_tol);
            } else {
                fcr_index_map.resize(nf);
                for (std::size_t i = 0; i < nf; ++i) fcr_index_map[i] = static_cast<int>(i);
            }
                for (int idx : fcr_index_map) {
                    if (idx < 0) {
                        throw std::runtime_error("Failed to map omega indices for FCR compare");
                    }
                }
        }

        std::size_t Nt_method = Nt_c;
        if (opt.compare_fcr_methods) {
            if (opt.fcr_nt > 0) {
                Nt_method = std::min(Nt_method, opt.fcr_nt);
            } else if (Nt_method > 4096) {
                Nt_method = 1024;
                std::cout << "compare-fcr-methods: using first " << Nt_method
                          << " samples; override with --fcr-nt=COUNT\n";
            }
        }

        std::size_t Nt_avail = Nt_c;
        if (opt.compare_gt) Nt_avail = std::min(Nt_avail, Nt_gt - opt.gt_offset);
        if (opt.compare_gw) Nt_avail = std::min(Nt_avail, Nt_gw);
        if (need_file_kernels) {
            if (opt.fcr_offset >= Nt_fcr) {
                throw std::runtime_error("fcr-offset exceeds F/C/R time length");
            }
            Nt_avail = std::min(Nt_avail, Nt_fcr - opt.fcr_offset);
        }
        if (opt.compare_fcr_methods) Nt_avail = std::min(Nt_avail, Nt_method);

        auto tidx_tokens = split_csv(opt.tidx_list);
        auto tidx_list = resolve_tidx_list(tidx_tokens, Nt_avail, opt.one_based);

        taco::sys::System system;
        taco::tcl4::Tcl4Map map;
        taco::tcl4::TripleKernelSeries kernels;
        Eigen::MatrixXcd Hm;
        Eigen::MatrixXcd Am;
        bool have_system = false;

        if (dataset_exists(h5.id, "/system/H/re")) {
            std::vector<hsize_t> dims_h;
            auto H = read_complex_array(h5.id, "/system/H", &dims_h);
            auto dims_h_use = squeeze_dims(dims_h);
            Hm.resize(static_cast<Eigen::Index>(dims_h_use[0]), static_cast<Eigen::Index>(dims_h_use[1]));
            for (std::size_t col = 0; col < dims_h_use[1]; ++col) {
                for (std::size_t row = 0; row < dims_h_use[0]; ++row) {
                    Hm(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                        at_colmajor_2d(H, dims_h_use, row, col);
                }
            }
            have_system = true;
        }
        if (dataset_exists(h5.id, "/system/A/re")) {
            std::vector<hsize_t> dims_a;
            auto A = read_complex_array(h5.id, "/system/A", &dims_a);
            auto dims_a_use = squeeze_dims(dims_a);
            Am.resize(static_cast<Eigen::Index>(dims_a_use[0]), static_cast<Eigen::Index>(dims_a_use[1]));
            for (std::size_t col = 0; col < dims_a_use[1]; ++col) {
                for (std::size_t row = 0; row < dims_a_use[0]; ++row) {
                    Am(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                        at_colmajor_2d(A, dims_a_use, row, col);
                }
            }
        }

        if (dataset_exists(h5.id, "/system/Eig/re")) {
            std::vector<hsize_t> dims_e;
            auto eig_re = read_array<double>(h5.id, "/system/Eig/re", H5T_NATIVE_DOUBLE, &dims_e);
            system.eig.dim = eig_re.size();
            system.eig.eps = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(eig_re.size()));
            for (std::size_t i = 0; i < eig_re.size(); ++i) {
                system.eig.eps(static_cast<Eigen::Index>(i)) = eig_re[i];
            }
            system.eig.U = Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(system.eig.dim),
                                                      static_cast<Eigen::Index>(system.eig.dim));
            system.eig.U_dag = system.eig.U;
            system.bf = taco::sys::BohrFrequencies(system.eig.eps);
        } else if (have_system) {
            system.eig = taco::sys::Eigensystem(Hm);
            system.bf = taco::sys::BohrFrequencies(system.eig.eps);
        }

        if (opt.compare_gw || opt.compare_fcr || opt.compare_fcr_methods ||
            (opt.print_mikx && opt.mikx_source == Options::MikxSource::Computed)) {
            system.fidx = build_frequency_index_from_map(map_omegas, map_ij, map_base, N, opt.omega_tol);
        }

        if (opt.fcr_fft_pad > 0) {
            taco::tcl4::set_fcr_fft_pad_factor(opt.fcr_fft_pad);
        }

        if (opt.compare_gw || opt.compare_fcr ||
            (opt.print_mikx && opt.mikx_source == Options::MikxSource::Computed)) {
            kernels = taco::tcl4::compute_triple_kernels(system, gamma_series, dt, 2, opt.method);
        }

        if (opt.compare_gw || (opt.print_mikx && opt.mikx_source == Options::MikxSource::Computed)) {
            if (!have_system || Am.size() == 0) {
                throw std::runtime_error("system/H and system/A required for GW compare");
            }
            system.A_eig = {Am};
            system.A_lab = {system.eig.to_lab(Am)};
            system.A_eig_parts = taco::sys::decompose_operators_by_frequency(system.A_eig, system.bf, system.fidx);
            map = taco::tcl4::build_map(system, {});
        }

        std::size_t failures = 0;
        std::cout << std::setprecision(12);

        if (opt.compare_gt) {
            ErrSummary gtstat;
            std::vector<int> pair_to_bucket;
            if (opt.gt_map_mode == Options::GtMapMode::Omega) {
                if (system.bf.dim == 0) {
                    throw std::runtime_error("system/Eig or system/H required for gt-map=omega");
                }
                pair_to_bucket = build_pair_to_bucket_by_omega(system.bf, gt_omegas, N,
                                                               opt.omega_tol, opt.gt_row_major_flat);
                for (int b : pair_to_bucket) {
                    if (b < 0) {
                        throw std::runtime_error("Failed to map pair to omega bucket; check omega-tol");
                    }
                }
            }
            for (std::size_t tidx : tidx_list) {
                const std::size_t file_tidx = tidx + opt.gt_offset;
                const double tval = (!tvals.empty() && tidx < tvals.size())
                                        ? tvals[tidx]
                                        : dt * static_cast<double>(tidx);
                if (opt.print_gt) {
                    std::cout << "Gt tidx=" << tidx << " t=" << tval << "\n";
                }
                for (std::size_t j = 0; j < N; ++j) {
                    for (std::size_t k = 0; k < N; ++k) {
                        const std::size_t idx = j + N * k;
                        long long b = 0;
                        if (opt.gt_map_mode == Options::GtMapMode::Ij) {
                            b = map_ij[idx] - map_base;
                        } else {
                            b = pair_to_bucket[idx];
                        }
                        if (b < 0 || static_cast<std::size_t>(b) >= gt_omegas.size()) {
                            throw std::runtime_error("Gt omega index out of range");
                        }
                        std::size_t b_map = static_cast<std::size_t>(b);
                        std::size_t b_print = static_cast<std::size_t>(b);
                        if (opt.gt_omega_order == Options::OmegaOrder::Sorted) {
                            if (opt.gt_map_mode == Options::GtMapMode::Omega) {
                                b_map = static_cast<std::size_t>(
                                    gt_sorted_to_map[static_cast<std::size_t>(b)]);
                                b_print = static_cast<std::size_t>(b);
                            } else {
                                b_print = static_cast<std::size_t>(
                                    gt_map_to_sorted[static_cast<std::size_t>(b)]);
                            }
                        }
                        const cd got = gamma_series(static_cast<Eigen::Index>(tidx),
                                                    static_cast<Eigen::Index>(b_map));
                        const cd expect = matrix_series_value(gt_layout, Gt_flat, file_tidx, j, k, N, N);
                        update_err(gtstat, got, expect, opt.atol, opt.rtol);
                        if (opt.print_gt) {
                            std::cout << "  (" << j << "," << k << ") omega=" << gt_omegas[b_print]
                                      << " Gt=(" << got.real() << "," << got.imag() << ")"
                                      << " file=(" << expect.real() << "," << expect.imag() << ")\n";
                        }
                    }
                }
            }
            const bool ok = gtstat.ok;
            if (!ok) failures++;
            std::cout << "Gt(matrix) max_abs=" << gtstat.max_abs
                      << " max_rel=" << gtstat.max_rel
                      << (ok ? " ok\n" : " FAIL\n");
        }

        auto file_kernel_value = [&](const KernelSeriesLayout& layout,
                                     const std::vector<cd>& data,
                                     std::size_t file_tidx,
                                     std::size_t bi,
                                     std::size_t bj,
                                     std::size_t bk) -> cd {
            if (fcr_index_map.empty()) {
                throw std::runtime_error("fcr_index_map is empty; file kernels not loaded");
            }
            const int idx[3] = {static_cast<int>(bi), static_cast<int>(bj), static_cast<int>(bk)};
            const int i0 = idx[opt.fcr_axes[0]];
            const int i1 = idx[opt.fcr_axes[1]];
            const int i2 = idx[opt.fcr_axes[2]];
            if (i0 < 0 || i1 < 0 || i2 < 0 ||
                static_cast<std::size_t>(i0) >= fcr_index_map.size() ||
                static_cast<std::size_t>(i1) >= fcr_index_map.size() ||
                static_cast<std::size_t>(i2) >= fcr_index_map.size()) {
                throw std::runtime_error("File kernel index out of range");
            }
            const std::size_t fi = static_cast<std::size_t>(fcr_index_map[static_cast<std::size_t>(i0)]);
            const std::size_t fj = static_cast<std::size_t>(fcr_index_map[static_cast<std::size_t>(i1)]);
            const std::size_t fk = static_cast<std::size_t>(fcr_index_map[static_cast<std::size_t>(i2)]);
            return kernel_series_value(layout, data, file_tidx, fi, fj, fk);
        };

        auto bucket_from_pair = [&](int a, int b) -> std::size_t {
            const std::size_t idx = static_cast<std::size_t>(a) + N * static_cast<std::size_t>(b);
            const long long val = map_ij[idx] - map_base;
            if (val < 0 || val >= static_cast<long long>(nf)) {
                throw std::runtime_error("map/ij index out of range for kernels");
            }
            return static_cast<std::size_t>(val);
        };

        if (opt.compare_gw || opt.print_mikx) {
            ErrSummary gwstat;
            for (std::size_t tidx : tidx_list) {
                const double tval = (!tvals.empty() && tidx < tvals.size())
                                        ? tvals[tidx]
                                        : dt * static_cast<double>(tidx);
                taco::tcl4::MikxTensors mikx;
                if (opt.print_mikx && opt.mikx_source == Options::MikxSource::File) {
                    const std::size_t file_tidx = tidx + opt.fcr_offset;
                    const int Nloc = static_cast<int>(N);
                    const std::size_t N2loc = N * N;
                    mikx.N = Nloc;
                    mikx.M = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2loc),
                                                    static_cast<Eigen::Index>(N2loc));
                    mikx.I = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2loc),
                                                    static_cast<Eigen::Index>(N2loc));
                    mikx.K = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(N2loc),
                                                    static_cast<Eigen::Index>(N2loc));
                    std::size_t totalX = 1;
                    for (int d = 0; d < 6; ++d) totalX *= N;
                    mikx.X.assign(totalX, cd{0.0, 0.0});

                    for (int j = 0; j < Nloc; ++j) {
                        for (int k = 0; k < Nloc; ++k) {
                            const std::size_t f_jk = bucket_from_pair(j, k);
                            const auto row = static_cast<Eigen::Index>(j + k * Nloc);
                            for (int p = 0; p < Nloc; ++p) {
                                for (int q = 0; q < Nloc; ++q) {
                                    const auto col = static_cast<Eigen::Index>(p + q * Nloc);
                                    const std::size_t f_jq = bucket_from_pair(j, q);
                                    const std::size_t f_pj = bucket_from_pair(p, j);
                                    const std::size_t f_pq = bucket_from_pair(p, q);
                                    const std::size_t f_qk = bucket_from_pair(q, k);
                                    const std::size_t f_kq = bucket_from_pair(k, q);
                                    const std::size_t f_qj = bucket_from_pair(q, j);
                                    const std::size_t f_qp = bucket_from_pair(q, p);

                                    const cd M1 = file_kernel_value(f_layout, F_all, file_tidx, f_jk, f_jq, f_pj);
                                    const cd M2 = file_kernel_value(r_layout, R_all, file_tidx, f_jq, f_pq, f_qk);
                                    mikx.M(row, col) = M1 - M2;

                                    const cd Ival = file_kernel_value(f_layout, F_all, file_tidx, f_jk, f_qp, f_kq);
                                    mikx.I(row, col) = Ival;

                                    const cd Kval = file_kernel_value(r_layout, R_all, file_tidx, f_jk, f_pq, f_qj);
                                    mikx.K(row, col) = Kval;

                                    for (int r = 0; r < Nloc; ++r) {
                                        for (int s = 0; s < Nloc; ++s) {
                                            const std::size_t f_rs = bucket_from_pair(r, s);
                                            const cd Cval = file_kernel_value(c_layout, C_all, file_tidx, f_jk, f_pq, f_rs);
                                            const cd Rval = file_kernel_value(r_layout, R_all, file_tidx, f_jk, f_pq, f_rs);
                                            const std::size_t idx6 = static_cast<std::size_t>(j)
                                                + N * (static_cast<std::size_t>(k)
                                                + N * (static_cast<std::size_t>(p)
                                                + N * (static_cast<std::size_t>(q)
                                                + N * (static_cast<std::size_t>(r)
                                                + N * static_cast<std::size_t>(s)))));
                                            mikx.X[idx6] = Cval + Rval;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
                }
                if (opt.print_mikx) {
                    const int Nloc = mikx.N;
                    std::cout << "MIKX tidx=" << tidx << " t=" << tval << " N=" << Nloc << "\n";
                    std::cout << "M:\n";
                    for (int r = 0; r < mikx.M.rows(); ++r) {
                        for (int c = 0; c < mikx.M.cols(); ++c) {
                            const cd v = mikx.M(r, c);
                            std::cout << "  (" << r << "," << c << ")=(" << v.real()
                                      << "," << v.imag() << ")\n";
                        }
                    }
                    std::cout << "I:\n";
                    for (int r = 0; r < mikx.I.rows(); ++r) {
                        for (int c = 0; c < mikx.I.cols(); ++c) {
                            const cd v = mikx.I(r, c);
                            std::cout << "  (" << r << "," << c << ")=(" << v.real()
                                      << "," << v.imag() << ")\n";
                        }
                    }
                    std::cout << "K:\n";
                    for (int r = 0; r < mikx.K.rows(); ++r) {
                        for (int c = 0; c < mikx.K.cols(); ++c) {
                            const cd v = mikx.K(r, c);
                            std::cout << "  (" << r << "," << c << ")=(" << v.real()
                                      << "," << v.imag() << ")\n";
                        }
                    }
                    std::cout << "X (flat, column-major j,k,p,q,r,s):\n";
                    const std::size_t N6 = mikx.X.size();
                    for (std::size_t idx = 0; idx < N6; ++idx) {
                        std::size_t tmp = idx;
                        const std::size_t j = tmp % Nloc; tmp /= Nloc;
                        const std::size_t k = tmp % Nloc; tmp /= Nloc;
                        const std::size_t p = tmp % Nloc; tmp /= Nloc;
                        const std::size_t q = tmp % Nloc; tmp /= Nloc;
                        const std::size_t r = tmp % Nloc; tmp /= Nloc;
                        const std::size_t s = tmp % Nloc;
                        const cd v = mikx.X[idx];
                        std::cout << "  (" << j << "," << k << "," << p << "," << q
                                  << "," << r << "," << s << ")=("
                                  << v.real() << "," << v.imag() << ")\n";
                    }
                }
                if (opt.compare_gw) {
                    Eigen::MatrixXcd GW = taco::tcl4::assemble_liouvillian(mikx, system.A_eig);
                    if (opt.print_gw) {
                        std::cout << "GW tidx=" << tidx << " t=" << tval << "\n";
                    }
                    for (int r = 0; r < GW.rows(); ++r) {
                        for (int c = 0; c < GW.cols(); ++c) {
                            const cd expect = matrix_series_value(gw_layout, GW_flat, tidx,
                                                                  static_cast<std::size_t>(r),
                                                                  static_cast<std::size_t>(c),
                                                                  N2, N2);
                            update_err(gwstat, GW(r, c), expect, opt.atol, opt.rtol);
                            if (opt.print_gw) {
                                std::cout << "  (" << r << "," << c << ") GW=("
                                          << GW(r, c).real() << "," << GW(r, c).imag() << ")"
                                          << " file=(" << expect.real() << "," << expect.imag() << ")\n";
                            }
                        }
                    }
                    std::cout << "GW tidx=" << tidx << " t=" << tval
                              << " max_abs=" << gwstat.max_abs
                              << " max_rel=" << gwstat.max_rel
                              << (gwstat.ok ? " ok\n" : " FAIL\n");
                }
            }
            if (opt.compare_gw && !gwstat.ok) failures++;
        }

        if (opt.compare_fcr) {
            if (kernels.F.empty() || kernels.F.front().empty() ||
                kernels.F.front().front().empty()) {
                throw std::runtime_error("Kernel series is empty");
            }
            ErrSummary fstat;
            ErrSummary cstat;
            ErrSummary rstat;
            if (opt.fcr_filter_enabled) {
                for (int v : opt.fcr_ijk) {
                    if (v < 0 || v >= static_cast<int>(nf)) {
                        throw std::runtime_error("fcr-ijk out of range for nf");
                    }
                }
            }
            for (std::size_t tidx : tidx_list) {
                const std::size_t file_tidx = tidx + opt.fcr_offset;
                if (opt.print_fcr) {
                    const double tval = (!tvals.empty() && tidx < tvals.size())
                                            ? tvals[tidx]
                                            : dt * static_cast<double>(tidx);
                    std::cout << "FCR tidx=" << tidx << " t=" << tval << "\n";
                }
                const std::size_t i0 = opt.fcr_filter_enabled ? static_cast<std::size_t>(opt.fcr_ijk[0]) : 0;
                const std::size_t j0 = opt.fcr_filter_enabled ? static_cast<std::size_t>(opt.fcr_ijk[1]) : 0;
                const std::size_t k0 = opt.fcr_filter_enabled ? static_cast<std::size_t>(opt.fcr_ijk[2]) : 0;
                const std::size_t i1 = opt.fcr_filter_enabled ? (i0 + 1) : nf;
                const std::size_t j1 = opt.fcr_filter_enabled ? (j0 + 1) : nf;
                const std::size_t k1 = opt.fcr_filter_enabled ? (k0 + 1) : nf;
                for (std::size_t i = i0; i < i1; ++i) {
                    for (std::size_t j = j0; j < j1; ++j) {
                        for (std::size_t k = k0; k < k1; ++k) {
                            const std::array<std::size_t, 3> idx{ i, j, k };
                            const std::size_t fi = static_cast<std::size_t>(
                                fcr_index_map[static_cast<std::size_t>(idx[opt.fcr_axes[0]])]);
                            const std::size_t fj = static_cast<std::size_t>(
                                fcr_index_map[static_cast<std::size_t>(idx[opt.fcr_axes[1]])]);
                            const std::size_t fk = static_cast<std::size_t>(
                                fcr_index_map[static_cast<std::size_t>(idx[opt.fcr_axes[2]])]);
                            const cd got_f = kernels.F[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd got_c = kernels.C[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd got_r = kernels.R[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd exp_f = kernel_series_value(f_layout, F_all, file_tidx, fi, fj, fk);
                            const cd exp_c = kernel_series_value(c_layout, C_all, file_tidx, fi, fj, fk);
                            const cd exp_r = kernel_series_value(r_layout, R_all, file_tidx, fi, fj, fk);
                            update_err(fstat, got_f, exp_f, opt.atol, opt.rtol);
                            update_err(cstat, got_c, exp_c, opt.atol, opt.rtol);
                            update_err(rstat, got_r, exp_r, opt.atol, opt.rtol);
                            if (opt.print_fcr) {
                                const bool show_f = opt.fcr_which == Options::FcrWhich::All ||
                                                    opt.fcr_which == Options::FcrWhich::F;
                                const bool show_c = opt.fcr_which == Options::FcrWhich::All ||
                                                    opt.fcr_which == Options::FcrWhich::C;
                                const bool show_r = opt.fcr_which == Options::FcrWhich::All ||
                                                    opt.fcr_which == Options::FcrWhich::R;
                                if (show_f) {
                                    std::cout << "  (" << i << "," << j << "," << k << ")"
                                              << " F=(" << got_f.real() << "," << got_f.imag() << ")"
                                              << " file=(" << exp_f.real() << "," << exp_f.imag() << ")\n";
                                }
                                if (show_c) {
                                    std::cout << "  (" << i << "," << j << "," << k << ")"
                                              << " C=(" << got_c.real() << "," << got_c.imag() << ")"
                                              << " file=(" << exp_c.real() << "," << exp_c.imag() << ")\n";
                                }
                                if (show_r) {
                                    std::cout << "  (" << i << "," << j << "," << k << ")"
                                              << " R=(" << got_r.real() << "," << got_r.imag() << ")"
                                              << " file=(" << exp_r.real() << "," << exp_r.imag() << ")\n";
                                }
                            }
                        }
                    }
                }
            }
            if (!fstat.ok || !cstat.ok || !rstat.ok) failures++;
            std::cout << "F_all max_abs=" << fstat.max_abs
                      << " max_rel=" << fstat.max_rel
                      << (fstat.ok ? " ok\n" : " FAIL\n");
            std::cout << "C_all max_abs=" << cstat.max_abs
                      << " max_rel=" << cstat.max_rel
                      << (cstat.ok ? " ok\n" : " FAIL\n");
            std::cout << "R_all max_abs=" << rstat.max_abs
                      << " max_rel=" << rstat.max_rel
                      << (rstat.ok ? " ok\n" : " FAIL\n");
        }

        if (opt.compare_fcr_methods) {
            Eigen::MatrixXcd gamma_use = gamma_series;
            if (Nt_method < static_cast<std::size_t>(gamma_series.rows())) {
                gamma_use = gamma_series.topRows(static_cast<Eigen::Index>(Nt_method));
            }
            auto kernels_conv = taco::tcl4::compute_triple_kernels(
                system, gamma_use, dt, 2, taco::tcl4::FCRMethod::Convolution);
            auto kernels_dir = taco::tcl4::compute_triple_kernels(
                system, gamma_use, dt, 2, taco::tcl4::FCRMethod::Direct);
            ErrSummary fstat;
            ErrSummary cstat;
            ErrSummary rstat;
            if (opt.fcr_filter_enabled) {
                for (int v : opt.fcr_ijk) {
                    if (v < 0 || v >= static_cast<int>(nf)) {
                        throw std::runtime_error("fcr-ijk out of range for nf");
                    }
                }
            }
            for (std::size_t tidx : tidx_list) {
                const std::size_t i0 = opt.fcr_filter_enabled ? static_cast<std::size_t>(opt.fcr_ijk[0]) : 0;
                const std::size_t j0 = opt.fcr_filter_enabled ? static_cast<std::size_t>(opt.fcr_ijk[1]) : 0;
                const std::size_t k0 = opt.fcr_filter_enabled ? static_cast<std::size_t>(opt.fcr_ijk[2]) : 0;
                const std::size_t i1 = opt.fcr_filter_enabled ? (i0 + 1) : nf;
                const std::size_t j1 = opt.fcr_filter_enabled ? (j0 + 1) : nf;
                const std::size_t k1 = opt.fcr_filter_enabled ? (k0 + 1) : nf;
                for (std::size_t i = i0; i < i1; ++i) {
                    for (std::size_t j = j0; j < j1; ++j) {
                        for (std::size_t k = k0; k < k1; ++k) {
                            const cd conv_f = kernels_conv.F[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd conv_c = kernels_conv.C[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd conv_r = kernels_conv.R[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd dir_f = kernels_dir.F[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd dir_c = kernels_dir.C[i][j][k](static_cast<Eigen::Index>(tidx));
                            const cd dir_r = kernels_dir.R[i][j][k](static_cast<Eigen::Index>(tidx));
                            update_err(fstat, conv_f, dir_f, opt.atol, opt.rtol);
                            update_err(cstat, conv_c, dir_c, opt.atol, opt.rtol);
                            update_err(rstat, conv_r, dir_r, opt.atol, opt.rtol);
                        }
                    }
                }
            }
            if (!fstat.ok || !cstat.ok || !rstat.ok) failures++;
            std::cout << "F(methods) max_abs=" << fstat.max_abs
                      << " max_rel=" << fstat.max_rel
                      << (fstat.ok ? " ok\n" : " FAIL\n");
            std::cout << "C(methods) max_abs=" << cstat.max_abs
                      << " max_rel=" << cstat.max_rel
                      << (cstat.ok ? " ok\n" : " FAIL\n");
            std::cout << "R(methods) max_abs=" << rstat.max_abs
                      << " max_rel=" << rstat.max_rel
                      << (rstat.ok ? " ok\n" : " FAIL\n");
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
