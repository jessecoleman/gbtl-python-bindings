#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <graphblas/graphblas.hpp>
//#include <graphblas.hpp>


namespace py = pybind11;

PYBIND11_MODULE(MODULE, m) {
    py::class_<GraphBLAS::NoMask>(m, "NoMask")
        .def(py::init<>());

    py::class_<GraphBLAS::AllIndices>(m, "AllIndices")
        .def(py::init<>());
}
