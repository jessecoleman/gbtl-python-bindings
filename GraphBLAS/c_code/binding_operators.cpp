#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// left type
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

// out type
#if defined(C_MATRIX)
typedef GraphBLAS::Matrix<CTYPE> CMatrixT;
#elif defined(C_VECTOR)
typedef GraphBLAS::Vector<CTYPE> WVectorT;
#endif

// mask type
#if defined(M_MATRIX)
typedef GraphBLAS::Matrix<MTYPE> MMatrixT;
#elif defined(M_VECTOR)
typedef GraphBLAS::Vector<MTYPE> MVectorT;
#elif defined(M_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<MTYPE>> MMatrixT;
#elif defined(M_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<MTYPE>> MVectorT;
#elif defined(M_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<MTYPE>> MMatrixT;
#elif defined(M_NOMASK)
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

#if defined(MXM)
// TODO check order of parameters
void mxm(
        CMatrixT &C,
        MMatrixT const &M,
        AMatrixT const &A,
        BMatrixT const &B,
        bool replace_flag
    )
{ GraphBLAS::mxm(C, M, AccumT(), SemiringT(), A, B, replace_flag); }

#elif defined(MXV)
void mxv(
        WVectorT &C,
        MVectorT const &M,
        AMatrixT const &A,
        VVectorT const &B,
        bool replace_flag
    )
{ GraphBLAS::mxv(C, M, AccumT(), SemiringT(), A, B, replace_flag); }

#elif defined(VXM)
void vxm(
        WVectorT &C,
        MVectorT const &M,
        UVectorT const &A,
        BMatrixT const &B,
        bool replace_flag
    )
{ GraphBLAS::vxm(C, M, AccumT(), SemiringT(), A, B, replace_flag); }

#elif defined(EWISEADDMATRIX)
void eWiseAddMatrix(
        CMatrixT &C,
        MMatrixT const &M,
        AMatrixT const &A,
        BMatrixT const &B,
        bool replace_flag
    )
{ GraphBLAS::eWiseAdd(C, M, AccumT(), AddBinaryOp(), A, B, replace_flag); }

#elif defined(EWISEADDVECTOR)
void eWiseAddVector(
        WVectorT &C,
        MVectorT const &M,
        UVectorT const &A,
        VVectorT const &B,
        bool replace_flag
    )
{ GraphBLAS::eWiseAdd(C, M, AccumT(), AddBinaryOp(), A, B, replace_flag); }

#elif defined(EWISEMULTMATRIX)
void eWiseMultMatrix(
        CMatrixT &C,
        MMatrixT const &mask,
        AMatrixT const &A,
        BMatrixT const &B,
        bool replace_flag
    )
{ GraphBLAS::eWiseMult(C, M, AccumT(), MultBinaryOp(), A, B, replace_flag); }

#elif defined(EWISEMULTVECTOR)
void eWiseMultVector(
        WVectorT &C,
        MVectorT const &M,
        UVectorT const &A,
        VVectorT const &B,
        bool replace_flag
    )
{ GraphBLAS::eWiseMult(C, M, AccumT(), MultBinaryOp(), A, B, replace_flag); }
#endif

PYBIND11_MODULE(MODULE, m) {

    py::class_<MonoidT>(m, "Monoid", py::module_local())
        .def(py::init<>())
        .def("identity", &MonoidT::identity);

    py::class_<SemiringT>(m, "Semiring", py::module_local())
        .def(py::init<>())
        .def("add", &SemiringT::add)
        .def("mult", &SemiringT::mult)
        .def("zero", &SemiringT::zero);

#if defined(MXM)
    m.def("mxm", &mxm, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(MXV)
    m.def("mxv", &mxv, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(VXM)
    m.def("vxm", &vxm, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(EWISEADDMATRIX)
    m.def("eWiseAddMatrix", &eWiseAddMatrix, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(EWISEADDVECTOR)
    m.def("eWiseAddVector", &eWiseAddVector, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(EWISEMULTMATRIX)
    m.def("eWiseMultMatrix", &eWiseMultMatrix, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(EWISEMULTVECTOR)
    m.def("eWiseMultVector", &eWiseMultVector, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#endif
}
