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
typedef BTYPE BScalarT;
typedef CTYPE CScalarT;

// matrix storage type
typedef GraphBLAS::Matrix<AScalarT> AMatrixT;
typedef GraphBLAS::Matrix<BScalarT> BMatrixT;
typedef GraphBLAS::Matrix<CScalarT> CMatrixT;
// vector mask types
typedef GraphBLAS::Matrix<bool>     MMatrixT;
typedef GraphBLAS::MatrixComplementView<CMatrixT> MatrixCompT;
// vector storage type
typedef GraphBLAS::Vector<AScalarT> UVectorT;
typedef GraphBLAS::Vector<BScalarT> VVectorT;
typedef GraphBLAS::Vector<CScalarT> WVectorT;
// vector mask types
typedef GraphBLAS::Vector<bool>     MVectorT;
typedef GraphBLAS::VectorComplementView<WVectorT> VectorCompT;

typedef GraphBLAS::NoMask NoMaskT;

// in case additive identity isn't plain string
#if MIN_IDENTITY == 1
#define ADD_IDNTY std::numeric_limits<CScalarT>::max()
#else
#define ADD_IDNTY ADD_IDENTITY
#endif

#if NO_ACCUM == 1
#define ACCUM_BINOP ACCUM_BINARYOP
#else
#define ACCUM_BINOP ACCUM_BINARYOP<CScalarT>
#endif

// create monoid and semiring from macro
GEN_GRAPHBLAS_MONOID(Monoid, GraphBLAS::ADD_BINARYOP, ADD_IDNTY)
GEN_GRAPHBLAS_SEMIRING(Semiring, Monoid, GraphBLAS::MULT_BINARYOP)

typedef Monoid<CScalarT> MonoidT;
typedef GraphBLAS::ADD_BINARYOP<AScalarT> AddBinaryOp;
typedef GraphBLAS::MULT_BINARYOP<AScalarT, BScalarT, CScalarT> MultBinaryOp;
typedef Semiring<AScalarT, BScalarT, CScalarT> SemiringT;
typedef GraphBLAS::ACCUM_BINOP AccumT; 
 
template<typename MaskT>
// TODO check order of parameters
void mxm(
        CMatrixT &C, 
        BMatrixT const &B, 
        AMatrixT const &A, 
        MaskT const &mask,
        bool replace_flag
    )
{ GraphBLAS::mxm(C, mask, AccumT(), SemiringT(), B, A, replace_flag); }

template<typename MaskT>
void mxv(
        WVectorT &w, 
        AMatrixT const &A, 
        UVectorT const &u, 
        MaskT const &mask,
        bool replace_flag
    )
{ GraphBLAS::mxv(w, mask, AccumT(), SemiringT(), A, u, replace_flag); }

template<typename MaskT>
void vxm(
        WVectorT &w, 
        UVectorT const &u, 
        AMatrixT const &A, 
        MaskT const &mask,
        bool replace_flag
    )
{ GraphBLAS::vxm(w, mask, AccumT(), SemiringT(), u, A, replace_flag); }

template<typename MaskT>
void eWiseAddMatrix(
        CMatrixT &C, 
        AMatrixT const &A, 
        BMatrixT const &B, 
        MaskT const &mask,
        bool replace_flag
    ) 
{ GraphBLAS::eWiseAdd(C, mask, AccumT(), AddBinaryOp(), A, B, replace_flag); }

template<typename MaskT>
void eWiseAddVector(
        WVectorT &w, 
        UVectorT const &u, 
        VVectorT const &v, 
        MaskT const &mask,
        bool replace_flag
    ) 
{ GraphBLAS::eWiseAdd(w, mask, AccumT(), AddBinaryOp(), u, v, replace_flag); }

template<typename MaskT>
void eWiseMultMatrix(
        CMatrixT &C, 
        AMatrixT const &A, 
        BMatrixT const &B, 
        MaskT const &mask,
        bool replace_flag
    ) 
{ GraphBLAS::eWiseMult(C, mask, AccumT(), MultBinaryOp(), A, B, replace_flag); }

template<typename MaskT>
void eWiseMultVector(
        WVectorT &w, 
        UVectorT const &u, 
        VVectorT const &v, 
        MaskT const &mask,
        bool replace_flag
    ) 
{ GraphBLAS::eWiseMult(w, mask, AccumT(), MultBinaryOp(), u, v, replace_flag); }

PYBIND11_MODULE(MODULE, m) {
    py::class_<MonoidT>(m, "Monoid", py::module_local())
        .def(py::init<>())
        .def("identity", &MonoidT::identity);

    py::class_<SemiringT>(m, "Semiring", py::module_local())
        .def(py::init<>())
        .def("add", &SemiringT::add)
        .def("mult", &SemiringT::mult)
        .def("zero", &SemiringT::zero);

    // unmasked
    m.def("mxm", &mxm<NoMaskT>);
    m.def("mxv", &mxv<NoMaskT>);
    m.def("vxm", &vxm<NoMaskT>);
    m.def("eWiseAdd", &eWiseAddMatrix<NoMaskT>);
    m.def("eWiseAdd", &eWiseAddVector<NoMaskT>);
    m.def("eWiseMult", &eWiseMultMatrix<NoMaskT>);
    m.def("eWiseMult", &eWiseMultVector<NoMaskT>);
    // masked
    m.def("mxm", &mxm<MMatrixT>);
    m.def("mxv", &mxv<MVectorT>);
    m.def("vxm", &vxm<MVectorT>);
    m.def("eWiseAdd", &eWiseAddMatrix<MMatrixT>);
    m.def("eWiseAdd", &eWiseAddVector<MVectorT>);
    m.def("eWiseMult", &eWiseMultMatrix<MMatrixT>);
    m.def("eWiseMult", &eWiseMultVector<MVectorT>);
    // inverted masked
    m.def("mxm", &mxm<MatrixCompT>);
    m.def("mxv", &mxv<VectorCompT>);
    m.def("vxm", &vxm<VectorCompT>);
    m.def("eWiseAdd", &eWiseAddMatrix<MatrixCompT>);
    m.def("eWiseAdd", &eWiseAddVector<VectorCompT>);
    m.def("eWiseMult", &eWiseMultMatrix<MatrixCompT>);
    m.def("eWiseMult", &eWiseMultVector<VectorCompT>);
}
