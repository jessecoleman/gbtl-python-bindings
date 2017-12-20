#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// left type
#if defined(A_MATRIX)
typedef GraphBLAS::Matrix<A_TYPE> AMatrixT;
#elif defined(A_VECTOR)
typedef GraphBLAS::Vector<A_TYPE> UVectorT;
#elif defined(A_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<A_TYPE>> AMatrixT;
#elif defined(A_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<A_TYPE>> UVectorT;
#elif defined(A_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<A_TYPE>> AMatrixT;
#endif

// right type
#if defined(B_MATRIX)
typedef GraphBLAS::Matrix<B_TYPE> BMatrixT;
#elif defined(B_VECTOR)
typedef GraphBLAS::Vector<B_TYPE> VVectorT;
#elif defined(B_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<B_TYPE>> BMatrixT;
#elif defined(B_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<B_TYPE>> VVectorT;
#elif defined(B_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<B_TYPE>> BMatrixT;
#endif

// out type
#if defined(C_MATRIX)
typedef GraphBLAS::Matrix<C_TYPE> CMatrixT;
#elif defined(C_VECTOR)
typedef GraphBLAS::Vector<C_TYPE> WVectorT;
#endif

// mask type
#if defined(M_MATRIX)
typedef GraphBLAS::Matrix<M_TYPE> MMatrixT;
#elif defined(M_VECTOR)
typedef GraphBLAS::Vector<M_TYPE> MVectorT;
#elif defined(M_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<M_TYPE>> MMatrixT;
#elif defined(M_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<M_TYPE>> MVectorT;
#elif defined(M_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<M_TYPE>> MMatrixT;
#elif defined(M_NOMASK)
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask MVectorT;
#endif

#if defined(SEMIRING) && defined(MIN_IDENTITY)
#define IDENTITY std::numeric_limits<C_TYPE>::max()
#elif defined(SEMIRING)
#define IDENTITY ADD_IDENTITY
#endif

#if defined(APPLY) && defined(BOUND_CONST)
typedef GraphBLAS::BinaryOp_Bind2nd<A_TYPE, GraphBLAS::UNARY_OP<A_TYPE, C_TYPE>> ApplyT;
#elif defined(APPLY)
#define BOUND_CONST
typedef GraphBLAS::UNARY_OP<A_TYPE, C_TYPE> ApplyT;
#endif

#if defined(NO_ACCUM)
typedef GraphBLAS::NoAccumulate AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<C_TYPE> AccumT;
#endif

#if defined(SEMIRING)
GEN_GRAPHBLAS_MONOID(Monoid, GraphBLAS::ADD_BINARYOP, IDENTITY)
GEN_GRAPHBLAS_SEMIRING(Semiring, Monoid, GraphBLAS::MULT_BINARYOP)

typedef Monoid<C_TYPE> MonoidT;
typedef GraphBLAS::ADD_BINARYOP<A_TYPE> AddBinaryOp;
typedef GraphBLAS::MULT_BINARYOP<A_TYPE, B_TYPE, C_TYPE> MultBinaryOp;
typedef Semiring<A_TYPE, B_TYPE, C_TYPE> SemiringT;
#endif

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

#elif defined(APPLY) && defined(C_MATRIX)
void applyMatrix(
        CMatrixT &C,
        MMatrixT const &M,
        AMatrixT const &A,
        bool replace_flag
    )
{ GraphBLAS::apply(C, M, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }

#elif defined(APPLY) && defined(C_VECTOR)
void applyVector(
        WVectorT &C,
        MVectorT const &M,
        UVectorT const &A, 
        bool replace_flag
   )
{ GraphBLAS::apply(C, M, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }
#endif

PYBIND11_MODULE(MODULE, m) {

#if defined(SEMIRING)
    py::class_<MonoidT>(m, "Monoid", py::module_local())
        .def(py::init<>())
        .def("identity", &MonoidT::identity);

    py::class_<SemiringT>(m, "Semiring", py::module_local())
        .def(py::init<>())
        .def("add", &SemiringT::add)
        .def("mult", &SemiringT::mult)
        .def("zero", &SemiringT::zero);
#endif

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
#elif defined(APPLY) && defined(C_MATRIX)
    m.def("apply", &applyMatrix, "C"_a, "M"_a, "A"_a, "replace_flag"_a);
#elif defined(APPLY) && defined(C_VECTOR)
    m.def("apply", &applyVector, "C"_a, "M"_a, "A"_a, "replace_flag"_a);
#endif
}
