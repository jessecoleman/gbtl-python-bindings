#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <graphblas/graphblas.hpp>

namespace py = pybind11;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// in type
#if defined(A_MATRIX)
typedef GraphBLAS::Matrix<ATYPE> AMatrixT;
#elif defined(A_VECTOR)
typedef GraphBLAS::Vector<ATYPE> UVectorT;
#elif defined(A_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<ATYPE>> AMatrixT;
#elif defined(A_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<ATYPE>> UVectorT;
#elif defined(A_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<ATYPE>> AMatrixT;
#endif

// right type
#if defined(B_MATRIX)
typedef GraphBLAS::Matrix<BTYPE> BMatrixT;
#elif defined(B_VECTOR)
typedef GraphBLAS::Vector<BTYPE> VVectorT;
#elif defined(B_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<BTYPE>> BMatrixT;
#elif defined(B_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<BTYPE>> VVectorT;
#elif defined(B_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<BTYPE>> BMatrixT;
#endif

// right type
#if defined(C_MATRIX)
typedef GraphBLAS::Matrix<CTYPE> CMatrixT;
#elif defined(C_VECTOR)
typedef GraphBLAS::Vector<CTYPE> WVectorT;
#elif defined(C_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<CTYPE>> CMatrixT;
#elif defined(C_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<CTYPE>> WVectorT;
#elif defined(C_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<CTYPE>> CMatrixT;
#endif

PYBIND11_MODULE(MODULE, m) {
#if defined(DIAGONAL)
    typedef GraphBLAS::Matrix<ATYPE> AMatrixT;
    m.def("diagonal", &GraphBLAS::diag<AMatrixT, UVectorT>);
#elif defined(SCALED_IDENTITY)
    typedef GraphBLAS::Matrix<ATYPE> AMatrixT;
    m.def("scaled_identity", &GraphBLAS::scaled_identity<AMatrixT>);
#elif defined(SPLIT)
    m.def("split", &GraphBLAS::split<AMatrixT>);
#elif defined(NORMALIZE_ROWS)
    m.def("normalize_rows", &GraphBLAS::normalize_rows<AMatrixT>);
#elif defined(NORMALIZE_COLS)
    m.def("normalize_cols", &GraphBLAS::normalize_cols<AMatrixT>);
#endif
}
