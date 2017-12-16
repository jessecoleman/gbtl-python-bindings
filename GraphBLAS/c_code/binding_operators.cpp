#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;

// left type
#if defined(A_TRANSPOSE)
typedef GraphBLAS::MatrixTransposeView<GraphBLAS::Matrix<ATYPE>> AMatrixT;
typedef GraphBLAS::Vector<ATYPE>                                 UVectorT;
#elif defined(A_COMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<ATYPE>> AMatrixT;
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<ATYPE>> UVectorT;
#else
typedef GraphBLAS::Matrix<ATYPE> AMatrixT;
typedef GraphBLAS::Vector<ATYPE> UVectorT;
#endif

// right type
#if defined(B_TRANSPOSE)
typedef GraphBLAS::MatrixTransposeView<GraphBLAS::Matrix<BTYPE>> BMatrixT;
typedef GraphBLAS::Vector<BTYPE>                                 VVectorT;
#elif defined(B_COMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<BTYPE>> BMatrixT;
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<BTYPE>> VVectorT;
#else
typedef GraphBLAS::Matrix<BTYPE> BMatrixT;
typedef GraphBLAS::Vector<BTYPE> VVectorT;
#endif

// out type
typedef GraphBLAS::Matrix<CTYPE> CMatrixT;
typedef GraphBLAS::Vector<CTYPE> WVectorT;

// mask type
#if defined(MASK)
typedef GraphBLAS::Matrix<MTYPE> MMatrixT;
typedef GraphBLAS::Vector<MTYPE> MVectorT;
#elif defined(COMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<MTYPE>> MMatrixT;
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<MTYPE>> MVectorT;
#elif defined(NONE)
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask MVectorT;
#endif

#ifdef MIN_IDENTITY
#define IDENTITY std::numeric_limits<CTYPE>::max()
#else
#define IDENTITY ADD_IDENTITY
#endif

#ifdef NO_ACCUM
typedef GraphBLAS::NoAccumulate AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<CTYPE> AccumT;
#endif

// create monoid and semiring from macro
GEN_GRAPHBLAS_MONOID(Monoid, GraphBLAS::ADD_BINARYOP, IDENTITY)
GEN_GRAPHBLAS_SEMIRING(Semiring, Monoid, GraphBLAS::MULT_BINARYOP)

typedef Monoid<CTYPE> MonoidT;
typedef GraphBLAS::ADD_BINARYOP<ATYPE> AddBinaryOp;
typedef GraphBLAS::MULT_BINARYOP<ATYPE, BTYPE, CTYPE> MultBinaryOp;
typedef Semiring<ATYPE, BTYPE, CTYPE> SemiringT;
 
// TODO check order of parameters
void mxm(
        CMatrixT &C,
        MMatrixT const &mask,
        AMatrixT const &A,
        BMatrixT const &B,
        bool replace_flag
    )
{ GraphBLAS::mxm(C, mask, AccumT(), SemiringT(), A, B, replace_flag); }

void mxv(
        WVectorT &w,
        MVectorT const &mask,
        AMatrixT const &A,
        VVectorT const &v,
        bool replace_flag
    )
{ GraphBLAS::mxv(w, mask, AccumT(), SemiringT(), A, v, replace_flag); }

void vxm(
        WVectorT &w,
        MVectorT const &mask,
        UVectorT const &u,
        BMatrixT const &B,
        bool replace_flag
    )
{ GraphBLAS::vxm(w, mask, AccumT(), SemiringT(), u, B, replace_flag); }

void eWiseAddMatrix(
        CMatrixT &C,
        MMatrixT const &mask,
        AMatrixT const &A,
        BMatrixT const &B,
        bool replace_flag
    ) 
{ GraphBLAS::eWiseAdd(C, mask, AccumT(), AddBinaryOp(), A, B, replace_flag); }

void eWiseAddVector(
        WVectorT &w,
        MVectorT const &mask,
        UVectorT const &u,
        VVectorT const &v,
        bool replace_flag
    ) 
{ GraphBLAS::eWiseAdd(w, mask, AccumT(), AddBinaryOp(), u, v, replace_flag); }

void eWiseMultMatrix(
        CMatrixT &C,
        MMatrixT const &mask,
        AMatrixT const &A,
        BMatrixT const &B,
        bool replace_flag
    ) 
{ GraphBLAS::eWiseMult(C, mask, AccumT(), MultBinaryOp(), A, B, replace_flag); }

void eWiseMultVector(
        WVectorT &w,
        MVectorT const &mask,
        UVectorT const &u,
        VVectorT const &v,
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

    m.def("mxm", &mxm);
    m.def("mxv", &mxv);
    m.def("vxm", &vxm);
    m.def("eWiseAdd", &eWiseAddMatrix);
    m.def("eWiseAdd", &eWiseAddVector);
    m.def("eWiseMult", &eWiseMultMatrix);
    m.def("eWiseMult", &eWiseMultVector);
}
