#include <pybind11/pybind11.h>
#include "taco/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_taco_native, m) {
    m.doc() = "TACO native bindings (MVP)";
    m.def("version", &taco::version);
}
