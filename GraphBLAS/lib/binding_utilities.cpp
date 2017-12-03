#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <graphblas/graphblas.hpp>

namespace py = pybind11;

PYBIND11_MODULE(MODULE, m) {
    py::class_<GraphBLAS::NoMask>(m, "NoMask")
        .def(py::init<>());
}
