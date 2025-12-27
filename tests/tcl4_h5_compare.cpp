#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <cctype>
#include <cmath>
#include <iomanip>
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
#include "taco/correlation_fft.hpp"

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

std::vector<hsize_t> reverse_dims(const std::vector<hsize_t>& dims) {
    std::vector<hsize_t> out = dims;
    std::reverse(out.begin(), out.end());
    return out;
}

enum class GammaRule {
    Trapz,
    Left,
    Right,
    Rect
};

enum class GwBasis {
    Pair,
    Omega
};

enum class GtMode {
    Auto,
    Omega,
    Matrix
};

double coth_stable(double x) {
    if (x < 1e-8) return 1.0 / x;
    if (x > 20.0) return 1.0;
    const double e2x = std::exp(2.0 * x);
    return (e2x + 1.0) / (e2x - 1.0);
}

std::vector<std::complex<double>> bcf_fft_ohmic_simple(double beta,
                                                       double dt,
                                                       std::size_t Nt_time,
                                                       double omegac,
                                                       std::size_t pad_factor,
                                                       bool use_pow2) {
    using cd = std::complex<double>;
    if (!(beta > 0.0)) throw std::invalid_argument("bcf_fft_ohmic_simple: beta must be > 0");
    if (!(dt > 0.0)) throw std::invalid_argument("bcf_fft_ohmic_simple: dt must be > 0");
    if (!(omegac > 0.0)) throw std::invalid_argument("bcf_fft_ohmic_simple: omegac must be > 0");

    std::size_t Nfft = 2;
    if (Nt_time > 1) {
        Nfft = std::max<std::size_t>(2, pad_factor * (Nt_time - 1));
    }
    if (use_pow2) {
        Nfft = bcf::next_pow2(Nfft);
    }
    if (Nfft % 2 != 0) ++Nfft;
    if (!use_pow2) {
        if ((Nfft & (Nfft - 1)) != 0) {
            throw std::invalid_argument("bcf_fft_ohmic_simple: Nfft must be power-of-two when use_pow2=false");
        }
    }

    const double pi = bcf::PI;
    const double domega = 2.0 * pi / (static_cast<double>(Nfft) * dt);
    const std::size_t wpos_len = Nfft / 2 + 1;

    std::vector<double> wpos(wpos_len, 0.0);
    for (std::size_t k = 0; k < wpos_len; ++k) wpos[k] = static_cast<double>(k) * domega;

    std::vector<cd> Spos(wpos_len, cd{0.0, 0.0});
    Spos[0] = cd{pi / (2.0 * beta), 0.0};
    if (wpos_len > 2) {
        for (std::size_t k = 1; k + 1 < wpos_len; ++k) {
            const double w = wpos[k];
            const double num = w * std::exp(-w / omegac);
            const double den = 1.0 - std::exp(-beta * w);
            Spos[k] = cd{(pi / 2.0) * (num / den), 0.0};
        }
    }
    const double wNyq = wpos[wpos_len - 1];
    const double x = 0.5 * beta * wNyq;
    const double cothv = coth_stable(x);
    Spos[wpos_len - 1] = cd{0.25 * pi * wNyq * std::exp(-wNyq / omegac) * cothv, 0.0};

    std::vector<cd> S(Nfft, cd{0.0, 0.0});
    for (std::size_t k = 0; k < wpos_len; ++k) S[k] = Spos[k];
    if (Nfft / 2 > 1) {
        for (std::size_t k = 1; k + 1 < wpos_len; ++k) {
            const double kms = std::exp(-beta * wpos[k]);
            S[Nfft - k] = Spos[k] * kms;
        }
    }

    bcf::FFTPlan plan(Nfft);
    plan.exec_forward(S);
    const double scale = domega / pi;

    std::vector<cd> C(Nt_time, cd{0.0, 0.0});
    for (std::size_t n = 0; n < Nt_time; ++n) {
        C[n] = S[n] * scale;
    }
    return C;
}

Eigen::MatrixXcd compute_gamma_prefix_matrix(const std::vector<std::complex<double>>& C,
                                             double dt,
                                             const std::vector<double>& omegas,
                                             GammaRule rule) {
    const std::size_t N = C.size();
    const std::size_t M = omegas.size();
    if (N == 0 || M == 0 || !(dt > 0.0)) return Eigen::MatrixXcd();
    Eigen::MatrixXcd G(static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(M));
    G.row(0).setZero();
    const std::complex<double> half_dt(dt / 2.0, 0.0);
    for (std::size_t j = 0; j < M; ++j) {
        const std::complex<double> step = std::exp(std::complex<double>{0.0, omegas[j] * dt});
        std::complex<double> phi(1.0, 0.0);
        std::complex<double> acc(0.0, 0.0);
        if (rule == GammaRule::Rect) {
            acc += dt * phi * C[0];
            G(static_cast<Eigen::Index>(0), static_cast<Eigen::Index>(j)) = acc;
        }
        for (std::size_t k = 1; k < N; ++k) {
            const auto phi_next = phi * step;
            switch (rule) {
                case GammaRule::Left:
                    acc += dt * phi * C[k - 1];
                    break;
                case GammaRule::Right:
                    acc += dt * phi_next * C[k];
                    break;
                case GammaRule::Rect:
                    acc += dt * phi_next * C[k];
                    break;
                case GammaRule::Trapz:
                default:
                    acc += half_dt * (phi * C[k - 1] + phi_next * C[k]);
                    break;
            }
            G(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(j)) = acc;
            phi = phi_next;
        }
    }
    return G;
}

std::vector<int> build_pair_to_bucket_by_omega(const taco::sys::BohrFrequencies& bf,
                                               const std::vector<double>& omegas,
                                               double tol,
                                               std::vector<int>* bucket_to_pair_out,
                                               std::vector<double>* pair_omega_out) {
    const std::size_t N = bf.dim;
    const std::size_t N2 = N * N;
    std::vector<int> pair_to_bucket(N2, -1);
    if (bucket_to_pair_out) bucket_to_pair_out->assign(omegas.size(), -1);
    if (pair_omega_out) pair_omega_out->assign(N2, 0.0);
    std::vector<char> used(omegas.size(), 0);

    for (std::size_t idx = 0; idx < N2; ++idx) {
        const std::size_t n = idx % N;
        const std::size_t m = idx / N;
        const double w = bf.omega(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(m));
        if (pair_omega_out) (*pair_omega_out)[idx] = w;
        int best = -1;
        for (std::size_t b = 0; b < omegas.size(); ++b) {
            if (std::abs(omegas[b] - w) <= tol && !used[b]) {
                best = static_cast<int>(b);
                break;
            }
        }
        if (best < 0) {
            for (std::size_t b = 0; b < omegas.size(); ++b) {
                if (std::abs(omegas[b] - w) <= tol) {
                    best = static_cast<int>(b);
                    break;
                }
            }
        }
        pair_to_bucket[idx] = best;
        if (best >= 0 && best < static_cast<int>(omegas.size()) && !used[best]) {
            used[best] = 1;
            if (bucket_to_pair_out) (*bucket_to_pair_out)[best] = static_cast<int>(idx);
        }
    }
    return pair_to_bucket;
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
    bool row_major_flat{true};
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
    const std::size_t flat_idx = layout.row_major_flat ? (row * N2 + col)
                                                       : (col * N2 + row);
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

struct MatLayout {
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
    bool row_major_flat{true};
};

MatLayout infer_mat_layout(const std::vector<hsize_t>& dims,
                           std::size_t rows,
                           std::size_t cols) {
    const std::size_t flat_len = rows * cols;
    MatLayout layout;
    layout.dims = dims;
    if (dims.size() == 2) {
        if (static_cast<std::size_t>(dims[1]) == flat_len) {
            layout.mode = MatLayout::Mode::FlatTimeRow;
            layout.Nt = static_cast<std::size_t>(dims[0]);
            layout.time_dim = 0;
            return layout;
        }
        if (static_cast<std::size_t>(dims[0]) == flat_len) {
            layout.mode = MatLayout::Mode::FlatTimeCol;
            layout.Nt = static_cast<std::size_t>(dims[1]);
            layout.time_dim = 1;
            return layout;
        }
    } else if (dims.size() == 3) {
        const std::size_t d0 = static_cast<std::size_t>(dims[0]);
        const std::size_t d1 = static_cast<std::size_t>(dims[1]);
        const std::size_t d2 = static_cast<std::size_t>(dims[2]);
        if (d0 == rows && d1 == cols) {
            layout.mode = MatLayout::Mode::MatTimeLast;
            layout.Nt = d2;
            layout.time_dim = 2;
            return layout;
        }
        if (d1 == rows && d2 == cols) {
            layout.mode = MatLayout::Mode::MatTimeFirst;
            layout.Nt = d0;
            layout.time_dim = 0;
            return layout;
        }
        if (d0 == rows && d2 == cols) {
            layout.mode = MatLayout::Mode::MatTimeMiddle;
            layout.Nt = d1;
            layout.time_dim = 1;
            return layout;
        }
    }
    throw std::runtime_error("Unsupported matrix dims: " + dims_to_string(dims));
}

std::complex<double> mat_value(const MatLayout& layout,
                               const std::vector<std::complex<double>>& data,
                               std::size_t tidx,
                               std::size_t row,
                               std::size_t col,
                               std::size_t rows,
                               std::size_t cols) {
    const std::size_t flat_idx = layout.row_major_flat ? (row * cols + col)
                                                       : (col * rows + row);
    switch (layout.mode) {
        case MatLayout::Mode::FlatTimeRow:
            return at_colmajor_2d(data, layout.dims, tidx, flat_idx);
        case MatLayout::Mode::FlatTimeCol:
            return at_colmajor_2d(data, layout.dims, flat_idx, tidx);
        case MatLayout::Mode::MatTimeLast:
            return at_colmajor_3d(data, layout.dims, row, col, tidx);
        case MatLayout::Mode::MatTimeFirst:
            return at_colmajor_3d(data, layout.dims, tidx, row, col);
        case MatLayout::Mode::MatTimeMiddle:
            return at_colmajor_3d(data, layout.dims, row, tidx, col);
    }
    return std::complex<double>(0.0, 0.0);
}

struct GtSeriesLayout {
    std::vector<hsize_t> dims;
    std::size_t nf{0};
    std::size_t time_dim{0};
};

std::complex<double> gt_series_value(const GtSeriesLayout& layout,
                                     const std::vector<std::complex<double>>& data,
                                     std::size_t tidx,
                                     std::size_t bucket) {
    if (layout.time_dim == 0) {
        return at_colmajor_2d(data, layout.dims, tidx, bucket);
    }
    return at_colmajor_2d(data, layout.dims, bucket, tidx);
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
        << "                           [--compare-fcr] [--nt=COUNT] [--list] [--matlab|--no-matlab]\n"
        << "                           [--gamma-sign=+1|-1] [--gamma-rule=trapz|left|right|rect]\n"
        << "                           [--gw-flat=row|col] [--gw-basis=pair|omega]\n"
        << "                           [--gw-omega=VAL] [--omega-tol=VAL] [--dump-map]\n"
        << "                           [--compare-bcf] [--bcf-pad=N] [--bcf-pow2=0|1] [--bcf-nt=COUNT]\n"
        << "                           [--compare-gt] [--gt-mode=auto|omega|matrix] [--gt-omega=VAL]\n"
        << "                           [--gt-time-dim=0|1]\n"
        << "                           [--gt-offset=N] [--print-gt]\n"
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
    bool matlab_order = false;
    double gamma_sign = 1.0;
    GammaRule gamma_rule = GammaRule::Trapz;
    bool gw_row_major = true;
    GwBasis gw_basis = GwBasis::Pair;
    GtMode gt_mode = GtMode::Auto;
    bool use_gw_omega_filter = false;
    double gw_omega_target = 0.0;
    double omega_tol = 1e-9;
    bool dump_map = false;
    bool compare_bcf = false;
    std::size_t bcf_pad = 10;
    bool bcf_pow2 = true;
    std::size_t bcf_nt = 0;
    bool compare_gt = false;
    std::size_t gt_offset = 0;
    bool print_gt = false;
    bool use_gt_omega_filter = false;
    double gt_omega_target = 0.0;
    int gt_time_dim_override = -1;
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
        if (arg == "--matlab") {
            matlab_order = true;
            continue;
        }
        if (arg == "--no-matlab") {
            matlab_order = false;
            continue;
        }
        if (arg.rfind("--gamma-sign=", 0) == 0) {
            gamma_sign = std::stod(arg.substr(13));
            if (gamma_sign != 1.0 && gamma_sign != -1.0) {
                std::cerr << "gamma-sign must be +1 or -1\n";
                return 2;
            }
            continue;
        }
        if (arg.rfind("--gamma-rule=", 0) == 0) {
            const std::string val = arg.substr(13);
            if (val == "left") gamma_rule = GammaRule::Left;
            else if (val == "right") gamma_rule = GammaRule::Right;
            else if (val == "rect") gamma_rule = GammaRule::Rect;
            else gamma_rule = GammaRule::Trapz;
            continue;
        }
        if (arg.rfind("--gw-flat=", 0) == 0) {
            const std::string val = arg.substr(10);
            if (val == "col") gw_row_major = false;
            else gw_row_major = true;
            continue;
        }
        if (arg.rfind("--gw-basis=", 0) == 0) {
            const std::string val = arg.substr(11);
            if (val == "omega") gw_basis = GwBasis::Omega;
            else gw_basis = GwBasis::Pair;
            continue;
        }
        if (arg.rfind("--gt-mode=", 0) == 0) {
            const std::string val = arg.substr(10);
            if (val == "omega") gt_mode = GtMode::Omega;
            else if (val == "matrix") gt_mode = GtMode::Matrix;
            else gt_mode = GtMode::Auto;
            continue;
        }
        if (arg.rfind("--gw-omega=", 0) == 0) {
            gw_omega_target = std::stod(arg.substr(11));
            use_gw_omega_filter = true;
            continue;
        }
        if (arg.rfind("--gt-omega=", 0) == 0) {
            gt_omega_target = std::stod(arg.substr(11));
            use_gt_omega_filter = true;
            continue;
        }
        if (arg.rfind("--gt-time-dim=", 0) == 0) {
            gt_time_dim_override = std::stoi(arg.substr(14));
            if (gt_time_dim_override != 0 && gt_time_dim_override != 1) {
                std::cerr << "gt-time-dim must be 0 or 1\n";
                return 2;
            }
            continue;
        }
        if (arg.rfind("--omega-tol=", 0) == 0) {
            omega_tol = std::stod(arg.substr(12));
            continue;
        }
        if (arg == "--dump-map") {
            dump_map = true;
            continue;
        }
        if (arg == "--compare-bcf") {
            compare_bcf = true;
            continue;
        }
        if (arg == "--compare-gt") {
            compare_gt = true;
            continue;
        }
        if (arg.rfind("--gt-offset=", 0) == 0) {
            gt_offset = static_cast<std::size_t>(std::stoull(arg.substr(12)));
            continue;
        }
        if (arg == "--print-gt") {
            print_gt = true;
            continue;
        }
        if (arg.rfind("--bcf-pad=", 0) == 0) {
            bcf_pad = static_cast<std::size_t>(std::stoull(arg.substr(10)));
            continue;
        }
        if (arg.rfind("--bcf-pow2=", 0) == 0) {
            bcf_pow2 = (std::stoi(arg.substr(11)) != 0);
            continue;
        }
        if (arg.rfind("--bcf-nt=", 0) == 0) {
            bcf_nt = static_cast<std::size_t>(std::stoull(arg.substr(9)));
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

        if (dump_map) {
            std::cout << "H (re):\n";
            for (Eigen::Index r = 0; r < H.rows(); ++r) {
                for (Eigen::Index c = 0; c < H.cols(); ++c) {
                    std::cout << H(r, c).real() << (c + 1 < H.cols() ? " " : "");
                }
                std::cout << "\n";
            }
            std::cout << "A (re):\n";
            for (Eigen::Index r = 0; r < A_eig.rows(); ++r) {
                for (Eigen::Index c = 0; c < A_eig.cols(); ++c) {
                    std::cout << A_eig(r, c).real() << (c + 1 < A_eig.cols() ? " " : "");
                }
                std::cout << "\n";
            }
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

            if (dump_map) {
                if (eig_vals.size() > 0) {
                    std::cout << "eig:";
                    for (Eigen::Index i = 0; i < eig_vals.size(); ++i) {
                        std::cout << " " << eig_vals(i);
                    }
                    std::cout << "\n";
                }
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
                        std::cout << (map_ij[idx] - base) << (k + 1 < N ? " " : "");
                    }
                    std::cout << "\n";
                }
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

        if (gw_basis == GwBasis::Omega && map_omegas.empty()) {
            throw std::runtime_error("gw-basis=omega requires /map/omegas");
        }

        auto dims_c_re_raw = get_dataset_dims(h5.id, "/bath/C/re");
        std::size_t c_time_dim = 0;
        if (dims_c_re_raw.size() == 2 && dims_c_re_raw[0] == 1 && dims_c_re_raw[1] > 1) {
            c_time_dim = 1;
        }
        const std::size_t Nt_c = static_cast<std::size_t>(dims_c_re_raw.empty() ? 0 : dims_c_re_raw[c_time_dim]);
        if (Nt_c == 0) throw std::runtime_error("Empty /bath/C");

        auto dims_gw_re_raw = get_dataset_dims(h5.id, "/out/GW_flat/re");
        auto gw_dims_base = squeeze_dims(dims_gw_re_raw);
        auto gw_dims_interp = matlab_order ? reverse_dims(gw_dims_base) : gw_dims_base;
        auto gw_map = squeezed_index_map(dims_gw_re_raw);
        const std::size_t gw_side = (gw_basis == GwBasis::Omega ? map_omegas.size() : N2);
        GwLayout gw_layout = infer_gw_layout(gw_dims_interp, gw_side);
        gw_layout.row_major_flat = gw_row_major;
        const std::size_t Nt_gw = gw_layout.Nt;

        bool have_gt = false;
        GtMode gt_mode_use = gt_mode;
        MatLayout gt_layout;
        GtSeriesLayout gt_series_layout;
        std::size_t Nt_gt = Nt_c;
        std::vector<hsize_t> dims_gt_raw;
        std::vector<std::size_t> gt_map;
        if (compare_gt && dataset_exists(h5.id, "/out/Gt_flat/re")) {
            dims_gt_raw = get_dataset_dims(h5.id, "/out/Gt_flat/re");
            auto gt_dims_base = squeeze_dims(dims_gt_raw);
            auto gt_dims_interp = matlab_order ? reverse_dims(gt_dims_base) : gt_dims_base;
            gt_map = squeezed_index_map(dims_gt_raw);
            const std::size_t nf = omegas.size();
            const bool gt_like_omega = (gt_dims_interp.size() == 2) &&
                                       (static_cast<std::size_t>(gt_dims_interp[0]) == nf ||
                                        static_cast<std::size_t>(gt_dims_interp[1]) == nf);
            if (gt_mode_use == GtMode::Auto) {
                gt_mode_use = gt_like_omega ? GtMode::Omega : GtMode::Matrix;
            }
            if (gt_mode_use == GtMode::Omega) {
                if (!gt_like_omega) {
                    throw std::runtime_error("Gt_flat dims do not match omega series layout");
                }
                gt_series_layout.dims = gt_dims_interp;
                gt_series_layout.nf = nf;
                gt_series_layout.time_dim = (static_cast<std::size_t>(gt_dims_interp[0]) == nf) ? 1 : 0;
                if (gt_time_dim_override >= 0) {
                    if (gt_dims_interp.size() != 2) {
                        throw std::runtime_error("gt-time-dim override requires 2D Gt_flat");
                    }
                    gt_series_layout.time_dim = static_cast<std::size_t>(gt_time_dim_override);
                }
                Nt_gt = static_cast<std::size_t>(gt_dims_interp[gt_series_layout.time_dim]);
            } else {
                gt_layout = infer_mat_layout(gt_dims_interp, N, N);
                gt_layout.row_major_flat = gw_row_major;
                Nt_gt = gt_layout.Nt;
            }
            have_gt = true;
        }

        bool have_fcr = false;
        FcrLayout fcr_layout;
        std::size_t Nt_fcr = Nt_c;
        std::size_t fcr_time_dim_raw = 0;
        if (compare_fcr && dataset_exists(h5.id, "/kernels/F_all/re") &&
            dataset_exists(h5.id, "/kernels/C_all/re") &&
            dataset_exists(h5.id, "/kernels/R_all/re")) {
            auto dims_f_raw = get_dataset_dims(h5.id, "/kernels/F_all/re");
            auto dims_c2_raw = get_dataset_dims(h5.id, "/kernels/C_all/re");
            auto dims_r_raw = get_dataset_dims(h5.id, "/kernels/R_all/re");
            auto dims_f = squeeze_dims(dims_f_raw);
            auto dims_c2 = squeeze_dims(dims_c2_raw);
            auto dims_r = squeeze_dims(dims_r_raw);
            auto dims_f_interp = matlab_order ? reverse_dims(dims_f) : dims_f;
            auto dims_c2_interp = matlab_order ? reverse_dims(dims_c2) : dims_c2;
            auto dims_r_interp = matlab_order ? reverse_dims(dims_r) : dims_r;
            if (dims_f_interp == dims_c2_interp && dims_f_interp == dims_r_interp) {
                fcr_layout = infer_fcr_layout(dims_f_interp, omegas.size());
                Nt_fcr = fcr_layout.Nt;
                have_fcr = true;
                std::size_t time_dim_raw = fcr_layout.time_dim;
                if (matlab_order) time_dim_raw = dims_f.size() - 1 - fcr_layout.time_dim;
                fcr_time_dim_raw = squeezed_index_map(dims_f_raw)[time_dim_raw];
            }
        }

        std::size_t Nt_avail = std::min(Nt_c, Nt_gw);
        if (compare_gt && have_gt) {
            if (gt_offset >= Nt_gt) {
                throw std::runtime_error("gt-offset exceeds Gt time length");
            }
            Nt_avail = std::min(Nt_avail, Nt_gt - gt_offset);
        }
        if (compare_fcr && have_fcr) Nt_avail = std::min(Nt_avail, Nt_fcr);
        if (max_steps > 0) Nt_avail = std::min(Nt_avail, max_steps);
        if (Nt_avail == 0) throw std::runtime_error("No overlapping time samples to compare");
        if (Nt_avail < Nt_c || Nt_avail < Nt_gw ||
            (compare_gt && have_gt && Nt_avail < Nt_gt) ||
            (compare_fcr && have_fcr && Nt_avail < Nt_fcr)) {
            std::cout << "Using Nt=" << Nt_avail << " (C=" << Nt_c << ", GW=" << Nt_gw;
            if (compare_gt && have_gt) std::cout << ", Gt=" << Nt_gt;
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

        int failures = 0;
        std::vector<hsize_t> dims_c;
        auto Cc = read_complex_array_prefix_dim(h5.id, "/bath/C", Nt_use, c_time_dim, &dims_c);

        if (compare_bcf) {
            if (!dataset_exists(h5.id, "/params/beta") || !dataset_exists(h5.id, "/params/omegac")) {
                throw std::runtime_error("Missing /params/beta or /params/omegac for BCF compare");
            }
            const double beta = read_scalar_double(h5.id, "/params/beta");
            const double omegac = read_scalar_double(h5.id, "/params/omegac");
            const std::size_t Nt_bcf = (bcf_nt > 0) ? bcf_nt : Nt_c;
            const std::size_t Nt_cmp = std::min(Nt_use, Nt_bcf);
            if (Nt_cmp != Nt_use) {
                std::cout << "BCF compare uses Nt=" << Nt_cmp << " (requested Nt=" << Nt_use << ")\n";
            }
            auto Ccalc = bcf_fft_ohmic_simple(beta, dt, Nt_bcf, omegac, bcf_pad, bcf_pow2);
            ErrSummary cstat;
            for (std::size_t k = 0; k < Nt_cmp; ++k) {
                update_err(cstat, Cc[k], Ccalc[k], atol, rtol);
            }
            std::cout << "BCF max_abs=" << cstat.max_abs
                      << " max_rel=" << cstat.max_rel
                      << (cstat.ok ? " ok\n" : " FAIL\n");
            if (!cstat.ok) failures++;
        }

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
        std::size_t gw_time_dim_raw = gw_layout.time_dim;
        if (matlab_order) gw_time_dim_raw = gw_dims_base.size() - 1 - gw_layout.time_dim;
        gw_time_dim_raw = gw_map[gw_time_dim_raw];
        auto GW_flat = read_complex_array_prefix_dim(h5.id, "/out/GW_flat", Nt_use, gw_time_dim_raw, &dims_gw);
        auto dims_gw_base = squeeze_dims(dims_gw);
        gw_layout.dims = matlab_order ? reverse_dims(dims_gw_base) : dims_gw_base;
        gw_layout.Nt = Nt_use;

        std::vector<std::complex<double>> Gt_flat;
        MatLayout gt_use;
        GtSeriesLayout gt_series_use;
        if (compare_gt && have_gt) {
            std::vector<hsize_t> dims_gt;
            auto gt_dims_base = squeeze_dims(dims_gt_raw);
            std::size_t gt_time_dim_raw = (gt_mode_use == GtMode::Omega)
                                              ? gt_series_layout.time_dim
                                              : gt_layout.time_dim;
            if (matlab_order) gt_time_dim_raw = gt_dims_base.size() - 1 - gt_time_dim_raw;
            gt_time_dim_raw = gt_map[gt_time_dim_raw];
            const std::size_t Nt_gt_read = Nt_use + gt_offset;
            Gt_flat = read_complex_array_prefix_dim(h5.id, "/out/Gt_flat", Nt_gt_read, gt_time_dim_raw, &dims_gt);
            auto dims_gt_use = squeeze_dims(dims_gt);
            auto dims_gt_interp = matlab_order ? reverse_dims(dims_gt_use) : dims_gt_use;
            if (gt_mode_use == GtMode::Omega) {
                gt_series_use = gt_series_layout;
                gt_series_use.dims = dims_gt_interp;
                gt_series_use.time_dim = (static_cast<std::size_t>(dims_gt_interp[0]) == gt_series_layout.nf) ? 1 : 0;
                if (gt_time_dim_override >= 0) {
                    if (dims_gt_interp.size() != 2) {
                        throw std::runtime_error("gt-time-dim override requires 2D Gt_flat");
                    }
                    gt_series_use.time_dim = static_cast<std::size_t>(gt_time_dim_override);
                }
            } else {
                gt_use = gt_layout;
                gt_use.dims = dims_gt_interp;
                gt_use.Nt = Nt_gt_read;
            }
        }

        std::vector<double> omegas_gamma = omegas;
        if (gamma_sign < 0.0) {
            for (auto& w : omegas_gamma) w = -w;
        }
        auto gamma_series = compute_gamma_prefix_matrix(Cc, dt, omegas_gamma, gamma_rule);
        auto kernels = taco::tcl4::compute_triple_kernels(system, gamma_series, dt, 2, method);
        auto map = taco::tcl4::build_map(system, {});

        if (print_gt) {
            std::cout << std::setprecision(12);
            const std::size_t nf = omegas.size();
            for (std::size_t tidx : tidx_list) {
                const double tval = (!tvals.empty() && tidx < tvals.size()) ? tvals[tidx] : (dt * static_cast<double>(tidx));
                std::cout << "Gt(omega) tidx=" << tidx << " t=" << tval << "\n";
                for (std::size_t b = 0; b < nf; ++b) {
                    if (use_gt_omega_filter) {
                        if (std::abs(omegas[b] - gt_omega_target) > omega_tol) continue;
                    }
                    const auto got = gamma_series(static_cast<Eigen::Index>(tidx), static_cast<Eigen::Index>(b));
                    std::cout << "  omega[" << b << "]=" << omegas[b]
                              << " Gt=(" << got.real() << "," << got.imag() << ")";
                    if (compare_gt && have_gt && gt_mode_use == GtMode::Omega) {
                        const auto file_tidx = tidx + gt_offset;
                        const auto expect = gt_series_value(gt_series_use, Gt_flat, file_tidx, b);
                        std::cout << " file=(" << expect.real() << "," << expect.imag() << ")";
                    }
                    std::cout << "\n";
                }
            }
        }

        std::vector<int> pair_to_bucket;
        std::vector<double> pair_omegas;
        if (gw_basis == GwBasis::Omega) {
            pair_to_bucket = build_pair_to_bucket_by_omega(system.bf, map_omegas, omega_tol, nullptr, &pair_omegas);
            std::size_t missing_pairs = 0;
            for (int b : pair_to_bucket) if (b < 0) ++missing_pairs;
            if (missing_pairs > 0) {
                std::cerr << "Warning: " << missing_pairs << " pair indices could not be matched to map/omegas\n";
            }
        }
        for (std::size_t tidx : tidx_list) {
            const double tval = (!tvals.empty() && tidx < tvals.size()) ? tvals[tidx] : (dt * static_cast<double>(tidx));
            auto mikx = taco::tcl4::build_mikx_serial(map, kernels, tidx);
            Eigen::MatrixXcd GW = taco::tcl4::assemble_liouvillian(mikx, system.A_eig);

            ErrSummary stat;
            for (int r = 0; r < GW.rows(); ++r) {
                for (int c = 0; c < GW.cols(); ++c) {
                    if (gw_basis == GwBasis::Omega) {
                        const int brow = pair_to_bucket[static_cast<std::size_t>(r)];
                        const int bcol = pair_to_bucket[static_cast<std::size_t>(c)];
                        if (brow < 0 || bcol < 0) continue;
                        if (use_gw_omega_filter) {
                            if (std::abs(map_omegas[static_cast<std::size_t>(brow)] - gw_omega_target) > omega_tol) continue;
                            if (std::abs(map_omegas[static_cast<std::size_t>(bcol)] - gw_omega_target) > omega_tol) continue;
                        }
                        const auto expect = gw_value(gw_layout, GW_flat, tidx,
                                                     static_cast<std::size_t>(brow),
                                                     static_cast<std::size_t>(bcol),
                                                     gw_side);
                        update_err(stat, GW(r, c), expect, atol, rtol);
                    } else {
                        if (use_gw_omega_filter) {
                            const double wr = system.bf.omega(r % static_cast<int>(N), r / static_cast<int>(N));
                            const double wc = system.bf.omega(c % static_cast<int>(N), c / static_cast<int>(N));
                            if (std::abs(wr - gw_omega_target) > omega_tol) continue;
                            if (std::abs(wc - gw_omega_target) > omega_tol) continue;
                        }
                        const auto expect = gw_value(gw_layout, GW_flat, tidx,
                                                     static_cast<std::size_t>(r),
                                                     static_cast<std::size_t>(c),
                                                     gw_side);
                        update_err(stat, GW(r, c), expect, atol, rtol);
                    }
                }
            }
            const bool ok = stat.ok;
            if (!ok) failures++;
            std::cout << "GW tidx=" << tidx << " t=" << tval
                      << " max_abs=" << stat.max_abs
                      << " max_rel=" << stat.max_rel
                      << (ok ? " ok\n" : " FAIL\n");
        }

        if (compare_gt && have_gt) {
            ErrSummary gtstat;
            if (gt_mode_use == GtMode::Omega) {
                const std::size_t nf = omegas.size();
                for (std::size_t tidx : tidx_list) {
                    for (std::size_t b = 0; b < nf; ++b) {
                        if (use_gt_omega_filter) {
                            if (std::abs(omegas[b] - gt_omega_target) > omega_tol) continue;
                        }
                        const auto file_tidx = tidx + gt_offset;
                        const auto expect = gt_series_value(gt_series_use, Gt_flat, file_tidx, b);
                        const auto got = gamma_series(static_cast<Eigen::Index>(tidx), static_cast<Eigen::Index>(b));
                        update_err(gtstat, got, expect, atol, rtol);
                    }
                }
                const bool ok = gtstat.ok;
                if (!ok) failures++;
                std::cout << "Gt(omega) max_abs=" << gtstat.max_abs
                          << " max_rel=" << gtstat.max_rel
                          << (ok ? " ok\n" : " FAIL\n");
            } else {
                for (std::size_t tidx : tidx_list) {
                    const auto file_tidx = tidx + gt_offset;
                    Eigen::MatrixXcd Gt = taco::tcl4::build_gamma_matrix_at(map, gamma_series, tidx);
                    for (int r = 0; r < Gt.rows(); ++r) {
                        for (int c = 0; c < Gt.cols(); ++c) {
                            const auto expect = mat_value(gt_use, Gt_flat, file_tidx,
                                                          static_cast<std::size_t>(r),
                                                          static_cast<std::size_t>(c),
                                                          N, N);
                            update_err(gtstat, Gt(r, c), expect, atol, rtol);
                        }
                    }
                }
                const bool ok = gtstat.ok;
                if (!ok) failures++;
                std::cout << "Gt(matrix) max_abs=" << gtstat.max_abs
                          << " max_rel=" << gtstat.max_rel
                          << (ok ? " ok\n" : " FAIL\n");
            }
        } else if (compare_gt && !have_gt) {
            std::cerr << "Gt_flat not found; skipping Gt compare\n";
        }

        if (compare_fcr) {
            if (have_fcr) {
                std::vector<hsize_t> dimsF;
                auto F_all = read_complex_array_prefix_dim(h5.id, "/kernels/F_all", Nt_use,
                                                           fcr_time_dim_raw, &dimsF);
                std::vector<hsize_t> dimsC;
                auto C_all = read_complex_array_prefix_dim(h5.id, "/kernels/C_all", Nt_use,
                                                           fcr_time_dim_raw, &dimsC);
                std::vector<hsize_t> dimsR;
                auto R_all = read_complex_array_prefix_dim(h5.id, "/kernels/R_all", Nt_use,
                                                           fcr_time_dim_raw, &dimsR);
                if (dimsF == dimsC && dimsF == dimsR) {
                    FcrLayout fcr_use = fcr_layout;
                    auto dimsF_base = squeeze_dims(dimsF);
                    fcr_use.dims = matlab_order ? reverse_dims(dimsF_base) : dimsF_base;
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
