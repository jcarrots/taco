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
        << "                           [--compare-fcr]\n"
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
        std::cerr << "Unknown arg: " << arg << "\n";
        print_usage();
        return 2;
    }

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    try {
        H5File h5(file);
        if (!dataset_exists(h5.id, "/params/dt")) {
            throw std::runtime_error("Missing /params/dt in HDF5");
        }

        const double dt = read_scalar_double(h5.id, "/params/dt");

        std::vector<hsize_t> dims_c;
        auto Cc = read_complex_array(h5.id, "/bath/C", &dims_c);
        const std::size_t Nt_total = numel(dims_c);
        if (Nt_total == 0) throw std::runtime_error("Empty /bath/C");

        std::vector<double> tvals;
        if (dataset_exists(h5.id, "/time/t")) {
            std::vector<hsize_t> dims_t;
            tvals = read_array<double>(h5.id, "/time/t", H5T_NATIVE_DOUBLE, &dims_t);
            if (!tvals.empty()) {
                tvals.resize(std::min(tvals.size(), Nt_total));
            }
        }

        std::vector<hsize_t> dims_gw;
        auto GW_flat = read_complex_array(h5.id, "/out/GW_flat", &dims_gw);
        auto gw_dims = squeeze_dims(dims_gw);
        if (gw_dims.size() != 2) throw std::runtime_error("GW_flat is not 2D");
        const std::size_t Nt_gw = static_cast<std::size_t>(gw_dims[0]);
        const std::size_t Nflat = static_cast<std::size_t>(gw_dims[1]);
        if (Nt_gw < Nt_total) {
            throw std::runtime_error("GW_flat has fewer time rows than C(t)");
        }
        if (Nflat != 16) {
            throw std::runtime_error("GW_flat second dim is not 16");
        }

        std::vector<std::string> tokens = tidx_tokens;
        if (tokens.empty()) {
            tokens = {"0", "mid", "last"};
        }
        std::vector<std::size_t> tidx_list;
        tidx_list.reserve(tokens.size());
        for (const auto& tok : tokens) {
            if (tok == "last") {
                tidx_list.push_back(Nt_total - 1);
            } else if (tok == "mid") {
                tidx_list.push_back(Nt_total / 2);
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
            if (t >= Nt_total) throw std::runtime_error("tidx out of range");
        }

        const std::size_t Nt_use = *std::max_element(tidx_list.begin(), tidx_list.end()) + 1;
        Cc.resize(Nt_use);

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

        taco::sys::System system;
        system.eig = taco::sys::Eigensystem(H);
        system.bf = taco::sys::BohrFrequencies(system.eig.eps);
        system.fidx = taco::sys::build_frequency_buckets(system.bf, 1e-9);
        system.A_eig = {A_eig};
        system.A_lab = {system.eig.to_lab(A_eig)};
        system.A_eig_parts = taco::sys::decompose_operators_by_frequency(system.A_eig, system.bf, system.fidx);

        std::vector<double> omegas;
        omegas.reserve(system.fidx.buckets.size());
        for (const auto& b : system.fidx.buckets) omegas.push_back(b.omega);

        if (dataset_exists(h5.id, "/map/omegas")) {
            std::vector<hsize_t> dims_om;
            auto om = read_array<double>(h5.id, "/map/omegas", H5T_NATIVE_DOUBLE, &dims_om);
            if (om.size() == omegas.size()) {
                double max_dw = 0.0;
                for (std::size_t i = 0; i < om.size(); ++i) {
                    max_dw = std::max(max_dw, std::abs(om[i] - omegas[i]));
                }
                if (max_dw > 1e-8) {
                    std::cerr << "Warning: map/omegas differs from System buckets (max |dw|=" << max_dw << ")\n";
                }
            }
        }

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
                    const std::size_t col = static_cast<std::size_t>(r * GW.cols() + c);
                    const auto expect = at_colmajor_2d(GW_flat, gw_dims, tidx, col);
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
            if (dataset_exists(h5.id, "/kernels/F_all/re") &&
                dataset_exists(h5.id, "/kernels/C_all/re") &&
                dataset_exists(h5.id, "/kernels/R_all/re")) {
                std::vector<hsize_t> dimsF;
                auto F_all = read_complex_array(h5.id, "/kernels/F_all", &dimsF);
                std::vector<hsize_t> dimsC;
                auto C_all = read_complex_array(h5.id, "/kernels/C_all", &dimsC);
                std::vector<hsize_t> dimsR;
                auto R_all = read_complex_array(h5.id, "/kernels/R_all", &dimsR);
                if (dimsF == dimsC && dimsF == dimsR && dimsF.size() == 4) {
                    if (static_cast<std::size_t>(dimsF[0]) < Nt_use) {
                        throw std::runtime_error("F/C/R datasets shorter than requested tidx range");
                    }
                    if (dimsF[1] != dimsF[2] || dimsF[2] != dimsF[3]) {
                        throw std::runtime_error("F/C/R datasets are not nf x nf x nf");
                    }
                    const std::size_t nf = static_cast<std::size_t>(dimsF[1]);
                    auto compare_kernel = [&](const char* name,
                                              const std::vector<std::complex<double>>& data,
                                              const std::vector<hsize_t>& dims,
                                              auto getter) {
                        ErrSummary stat;
                        for (std::size_t tidx : tidx_list) {
                            for (std::size_t i = 0; i < nf; ++i) {
                                for (std::size_t j = 0; j < nf; ++j) {
                                    for (std::size_t k = 0; k < nf; ++k) {
                                        const auto expect = at_colmajor_4d(data, dims, tidx, i, j, k);
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
                    compare_kernel("F", F_all, dimsF,
                                   [&](std::size_t i, std::size_t j, std::size_t k, std::size_t t) {
                                       return kernels.F[i][j][k](static_cast<Eigen::Index>(t));
                                   });
                    compare_kernel("C", C_all, dimsC,
                                   [&](std::size_t i, std::size_t j, std::size_t k, std::size_t t) {
                                       return kernels.C[i][j][k](static_cast<Eigen::Index>(t));
                                   });
                    compare_kernel("R", R_all, dimsR,
                                   [&](std::size_t i, std::size_t j, std::size_t k, std::size_t t) {
                                       return kernels.R[i][j][k](static_cast<Eigen::Index>(t));
                                   });
                } else {
                    std::cerr << "F/C/R dataset dims mismatch; skipping FCR compare\n";
                }
            } else {
                std::cerr << "F/C/R datasets not found; skipping FCR compare\n";
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
