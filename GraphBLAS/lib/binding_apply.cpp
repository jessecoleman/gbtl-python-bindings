#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;

/* AScalarT: type of first input
 * BScalarT: type of second input
 * CScalarT: type of output
 */
typedef ATYPE AScalarT;
typedef CTYPE CScalarT;

// matrix storage type
typedef GraphBLAS::Matrix<AScalarT> AMatrixT;
typedef GraphBLAS::Matrix<CScalarT> CMatrixT;
typedef GraphBLAS::Matrix<bool> MMatrixT;
typedef GraphBLAS::MatrixComplementView<MMatrixT> MMatrixCompT;
// vector storage type
typedef GraphBLAS::Vector<AScalarT> UVectorT;
typedef GraphBLAS::Vector<CScalarT> WVectorT;
typedef GraphBLAS::Vector<bool> MVectorT;
typedef GraphBLAS::VectorComplementView<MVectorT> MVectorCompT;

typedef GraphBLAS::NoMask NoMaskT;

#if NO_ACCUM == 1
#define ACCUM_BINOP ACCUM_BINARYOP
#else
#define ACCUM_BINOP ACCUM_BINARYOP<CScalarT>
#endif

typedef GraphBLAS::ACCUM_BINOP AccumT; 
 
#if BOUND_SECOND == 1
typedef GraphBLAS::BinaryOp_Bind2nd<AScalarT, GraphBLAS::APPLY_OP<AScalarT, CScalarT>> ApplyT;
#else
typedef GraphBLAS::APPLY_OP<AScalarT, CScalarT> ApplyT;
#endif

template <typename MMatrixT>
void applyMatrix(
        CMatrixT &C, 
        AMatrixT const &A, 
        MMatrixT const &mask, 
        bool replace_flag
    )
{ GraphBLAS::apply(C, mask, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }

template <typename MVectorT>
void applyVector(
        WVectorT &w, 
        UVectorT const &u, 
        MVectorT const &mask, 
        bool replace_flag
   )
{ GraphBLAS::apply(w, mask, AccumT(), ApplyT(BOUND_CONST), u, replace_flag); }

PYBIND11_MODULE(MODULE, m) {
    m.def("apply", &applyMatrix<NoMaskT>);
    m.def("apply", &applyMatrix<MMatrixT>);
    m.def("apply", &applyMatrix<MMatrixCompT>);
    m.def("apply", &applyVector<NoMaskT>);
    m.def("apply", &applyVector<MVectorT>);
    m.def("apply", &applyVector<MVectorCompT>);
}
