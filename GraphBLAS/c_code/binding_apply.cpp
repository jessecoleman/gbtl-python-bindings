#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;

// in type
#if defined(A_TRANSPOSE)
typedef GraphBLAS::MatrixTransposeView<GraphBLAS::Matrix<ATYPE>> AMatrixT;
typedef GraphBLAS::Vector<ATYPE> UVectorT;
#elif defined(A_COMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<ATYPE>> AMatrixT;
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<ATYPE>> UVectorT;
#else
typedef GraphBLAS::Matrix<ATYPE> AMatrixT;
typedef GraphBLAS::Vector<ATYPE> UVectorT;

// out type
typedef GraphBLAS::Matrix<CTYPE> CMatrixT;
typedef GraphBLAS::Vector<CTYPE> WVectorT;

// mask types
#if defined(MASK)
typedef GraphBLAS::Matrix<MTYPE> MMatrixT;
typedef GraphBLAS::Vector<MTYPE> MVectorT;
#elif defined(COMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<MTYPE>> MMatrixT;
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<MTYPE>> MVectorT;
#else
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask MVectorT;
#endif

#ifdef NO_ACCUM
typedef GraphBLAS::NoAccumulate AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<CTYPE> AccumT;
#endif

#ifdef BOUND_CONST
typedef GraphBLAS::BinaryOp_Bind2nd<ATYPE, GraphBLAS::APPLY_OP<ATYPE, CTYPE>> ApplyT;
#else
typedef GraphBLAS::APPLY_OP<ATYPE, CTYPE> ApplyT;
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
