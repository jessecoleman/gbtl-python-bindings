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
// vector storage type
typedef GraphBLAS::Vector<AScalarT> UVectorT;
typedef GraphBLAS::Vector<CScalarT> WVectorT;

// mask types
#if MASK == 0
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask VMatrixT;
#elif MASK == 1
typedef GraphBLAS::Matrix<MTYPE> MMatrixT;
typedef GraphBLAS::Vector<MTYPE> MVectorT;
#elif MASK == 2
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<MTYPE>> MMatrixT;
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<MTYPE>> MVectorT;
#endif

#if NO_ACCUM == 1
typedef GraphBLAS::ACCUM_BINARYOP AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<CScalarT> AccumT;
#endif

#if BOUND_SECOND == 1
typedef GraphBLAS::BinaryOp_Bind2nd<AScalarT, GraphBLAS::APPLY_OP<AScalarT, CScalarT>> ApplyT;
#else
typedef GraphBLAS::APPLY_OP<AScalarT, CScalarT> ApplyT;
#endif

template <typename MMatrixT>
void applyMatrix(
        CMatrixT &C, 
        MMatrixT const &mask, 
        AMatrixT const &A, 
        bool replace_flag
    )
{ GraphBLAS::apply(C, mask, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }

template <typename MVectorT>
void applyVector(
        WVectorT &w, 
        MVectorT const &mask, 
        UVectorT const &u, 
        bool replace_flag
   )
{ GraphBLAS::apply(w, mask, AccumT(), ApplyT(BOUND_CONST), u, replace_flag); }

PYBIND11_MODULE(MODULE, m) {
    m.def("apply", &applyMatrix<MMatrixT>);
    m.def("apply", &applyVector<MVectorT>);
}
