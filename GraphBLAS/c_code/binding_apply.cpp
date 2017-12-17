#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;

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

// out type
#if defined(B_MATRIX)
typedef GraphBLAS::Matrix<BTYPE> CMatrixT;
#elif defined(B_VECTOR)
typedef GraphBLAS::Vector<BTYPE> WVectorT;
#endif

// mask type
#if defined(C_MATRIX)
typedef GraphBLAS::Matrix<CTYPE> MMatrixT;
#elif defined(C_VECTOR)
typedef GraphBLAS::Vector<CTYPE> MVectorT;
#elif defined(C_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<CTYPE>> MMatrixT;
#elif defined(C_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<CTYPE>> MVectorT;
#elif defined(C_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<ATYPE>> AMatrixT;
#elif defined(C_NOMASK)
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask MVectorT;
#endif

#ifdef NO_ACCUM
typedef GraphBLAS::NoAccumulate AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<BTYPE> AccumT;
#endif

#ifdef BOUND_CONST
typedef GraphBLAS::BinaryOp_Bind2nd<ATYPE, GraphBLAS::APPLY_OP<ATYPE, BTYPE>> ApplyT;
#else
#define BOUND_CONST
typedef GraphBLAS::APPLY_OP<ATYPE, BTYPE> ApplyT;
#endif

#if defined(A_MATRIX) || defined(A_MATRIXCOMPLEMENT) || defined(A_MATRIXTRANSPOSE)
template <typename MMatrixT>
void applyMatrix(
        CMatrixT &C, 
        MMatrixT const &mask, 
        AMatrixT const &A, 
        bool replace_flag
    )
{ GraphBLAS::apply(C, mask, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }

#elif defined(A_VECTOR) || defined(A_VECTORCOMPLEMENT)
template <typename MVectorT>
void applyVector(
        WVectorT &w, 
        MVectorT const &mask, 
        UVectorT const &u, 
        bool replace_flag
   )
{ GraphBLAS::apply(w, mask, AccumT(), ApplyT(BOUND_CONST), u, replace_flag); }
#endif

PYBIND11_MODULE(MODULE, m) {
#if defined(A_MATRIX) || defined(A_MATRIXCOMPLEMENT) || defined(A_MATRIXTRANSPOSE)
    m.def("apply", &applyMatrix<MMatrixT>);
#elif defined(A_VECTOR) || defined(A_VECTORCOMPLEMENT)
    m.def("apply", &applyVector<MVectorT>);
#endif
}
