#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <graphblas/graphblas.hpp>

namespace py = pybind11;

PYBIND11_MODULE(MODULE, m) {
    m.def("diag", &GraphBLAS::diag<AMatrixT, UVectorT>);
    m.def("scaled_identity", &GraphBLAS::scaled_identity<AMatrixT>);
    m.def("split", &GraphBLAS::split<AMatrixT>);
    m.def("normalize_rows", &GraphBLAS::normalize_rows<AMatrixT>);
    m.def("normalize_cols", &GraphBLAS::normalize_cols<AMatrixT>);
}
