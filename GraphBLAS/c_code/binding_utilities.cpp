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

#if defined(DIAGONAL)
typedef GraphBLAS::Matrix<V_TYPE> MatrixT;
typedef GraphBLAS::Vector<V_TYPE> VectorT;
#elif defined(SCALED_IDENTITY)
typedef GraphBLAS::Matrix<VAL_TYPE> MatrixT;
#elif defined(SPLIT) || defined(NORMALIZE_ROWS) || defined(NORMALIZE_COLS)
typedef GraphBLAS::Matrix<A_TYPE> MatrixT;
#endif

PYBIND11_MODULE(MODULE, m) {
#if defined(DIAGONAL)
    m.def("diagonal", &GraphBLAS::diag<MatrixT, VectorT>, "v"_a);
#elif defined(SCALED_IDENTITY)
    m.def("scaled_identity", &GraphBLAS::scaled_identity<MatrixT>, "mat_size"_a, "val"_a=static_cast<typename MatrixT::ScalarType>(1));
#elif defined(SPLIT)
    m.def("split", &GraphBLAS::split<MatrixT>, "A"_a, "L"_a, "U"_a);
#elif defined(NORMALIZE_ROWS)
    m.def("normalize_rows", &GraphBLAS::normalize_rows<MatrixT>, "A"_a);
#elif defined(NORMALIZE_COLS)
    m.def("normalize_cols", &GraphBLAS::normalize_cols<MatrixT>, "A"_a);
#endif
}
