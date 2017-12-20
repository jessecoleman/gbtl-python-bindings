#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <graphblas/graphblas.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

#if defined(V_TYPE)
typedef GraphBLAS::Matrix<V_TYPE> MatrixT;
typedef GraphBLAS::Vector<V_TYPE> VectorT;
#elif defined(A_TYPE)
typedef GraphBLAS::Matrix<A_TYPE> MatrixT;
typedef GraphBLAS::Vector<A_TYPE> VectorT;
#endif

PYBIND11_MODULE(MODULE, m) {
    m.def("diagonal",           &GraphBLAS::diag<MatrixT, VectorT>, "v"_a);
    m.def("scaled_identity",    &GraphBLAS::scaled_identity<MatrixT>, "mat_size"_a, "val"_a=static_cast<typename MatrixT::ScalarType>(1));
    m.def("split",              &GraphBLAS::split<MatrixT>, "A"_a, "L"_a, "U"_a);
    m.def("normalize_rows",     &GraphBLAS::normalize_rows<MatrixT>, "A"_a);
    m.def("normalize_cols",     &GraphBLAS::normalize_cols<MatrixT>, "A"_a);
}
