#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"
//#include <graphblas/Matrix.hpp>
//#include <graphblas/Vector.hpp>
//#include <graphblas/algebra.hpp>

namespace py = pybind11;

// matrix dtype
typedef DTYPE ScalarT;

// matrix storage type
typedef GraphBLAS::Matrix<ScalarT> GBMatrix;
typedef GraphBLAS::Vector<ScalarT> GBVector;

// in case additive identity isn't plain string
#if ADD_IDENTITY == MinIdentity
#define ADD_IDENTITY std::numeric_limits<ScalarT>::max()
#endif

// create monoid and semiring from macro
GEN_GRAPHBLAS_MONOID(Monoid, GraphBLAS::ADD_BINARYOP, ADD_IDENTITY)
GEN_GRAPHBLAS_SEMIRING(Semiring, Monoid, GraphBLAS::MULT_BINARYOP)

// bind monoid and semiring
#define GEN_PYBIND_MONOID(Monoid)                                               \
    py::class_<Monoid<ScalarT> >(m, "Monoid", py::module_local())               \
        .def(py::init<>())                                                     \
        .def("identity", &Monoid<ScalarT>::identity);                           \

#define GEN_PYBIND_SEMIRING(SRNAME)                                            \
    py::class_<SRNAME<ScalarT> >(m, "Semiring", py::module_local())            \
        .def(py::init<>())                                                     \
        .def("add", &SRNAME<ScalarT>::add)                                     \
        .def("mult", &SRNAME<ScalarT>::mult)                                   \
        .def("zero", &SRNAME<ScalarT>::zero);                                  \

template<typename CMatrixT,
         typename AMatrixT,
         typename BMatrixT,
         typename MMatrixT = GraphBLAS::NoMask,
         typename AccumT = GraphBLAS::NoAccumulate>
CMatrixT mxm(BMatrixT const &B, AMatrixT const &A) 
{
    CMatrixT C(B.nrows(), A.ncols());
    GraphBLAS::mxm(C, MMatrixT(), AccumT(), Semiring<ScalarT>(), B, A);
    return C;
}

template<typename WVectorT,
         typename AMatrixT,
         typename UVectorT,
         typename MaskT = GraphBLAS::NoMask,
         typename AccumT = GraphBLAS::NoAccumulate>
 WVectorT mxv(AMatrixT const &A, UVectorT const &u)
{
    WVectorT w(A.nrows());
    GraphBLAS::mxv(w, MaskT(), AccumT(), Semiring<ScalarT>(), A, u);
    return w;
}

template<typename WVectorT,
         typename UVectorT,
         typename AMatrixT,
         typename MaskT = GraphBLAS::NoMask,
         typename AccumT = GraphBLAS::NoAccumulate>
 WVectorT vxm(AMatrixT  const &A, UVectorT  const &u)
{
    WVectorT w(A.ncols());
    GraphBLAS::vxm(w, MaskT(), AccumT(), Semiring<ScalarT>(), u, A);
    return w;
}

template<typename CMatrixT,
         typename AMatrixT,
         typename BMatrixT,
         typename MaskT = GraphBLAS::NoMask,
         typename AccumT = GraphBLAS::NoAccumulate>
CMatrixT eWiseAddMatrix(AMatrixT const &A, BMatrixT const &B) {
    CMatrixT C(std::max(A.nrows(), B.nrows()), 
               std::max(A.ncols(), B.ncols()));
    GraphBLAS::eWiseAdd(C, MaskT(), AccumT(), 
            GraphBLAS::ADD_BINARYOP<ScalarT>(), A, B);
    return C;
}

template<typename WVectorT,
         typename UVectorT,
         typename VVectorT,
         typename MaskT = GraphBLAS::NoMask,
         typename AccumT = GraphBLAS::NoAccumulate>
WVectorT eWiseAddVector(UVectorT const &u, VVectorT const &v) {
    WVectorT w(std::max(u.size(), v.size()));
    GraphBLAS::eWiseAdd(w, MaskT(), AccumT(), 
            GraphBLAS::ADD_BINARYOP<ScalarT>(), u, v);
    return w;
}

template<typename CMatrixT,
         typename AMatrixT,
         typename BMatrixT,
         typename MaskT = GraphBLAS::NoMask,
         typename AccumT = GraphBLAS::NoAccumulate>
CMatrixT eWiseMultMatrix(AMatrixT const &A, BMatrixT const &B) {
    CMatrixT C(std::max(A.nrows(), B.nrows()), 
               std::max(A.ncols(), B.ncols()));
    GraphBLAS::eWiseMult(C, MaskT(), AccumT(), 
            GraphBLAS::MULT_BINARYOP<ScalarT>(), A, B);
    return C;
} 

template<typename WVectorT,
         typename UVectorT,
         typename VVectorT,
         typename MaskT = GraphBLAS::NoMask,
         typename AccumT = GraphBLAS::NoAccumulate>
WVectorT eWiseMultVector(UVectorT const &u, VVectorT const &v) {
    WVectorT w(std::max(u.size(), v.size()));
    GraphBLAS::eWiseMult(w, MaskT(), AccumT(), 
            GraphBLAS::MULT_BINARYOP<ScalarT>(), u, v);
    return w;
}

PYBIND11_MODULE(MODULE, m) {
    py::class_<Monoid<ScalarT> >(m, "Monoid", py::module_local())
        .def(py::init<>())
        .def("identity", &Monoid<ScalarT>::identity);

    py::class_<Semiring<ScalarT> >(m, "Semiring", py::module_local())
        .def(py::init<>())
        .def("add", &Semiring<ScalarT>::add)
        .def("mult", &Semiring<ScalarT>::mult)
        .def("zero", &Semiring<ScalarT>::zero);

    m.def("mxm", &mxm<GBMatrix, GBMatrix, GBMatrix>);
    m.def("mxv", &mxv<GBVector, GBMatrix, GBVector>);
    m.def("vxm", &vxm<GBVector, GBVector, GBMatrix>);
    m.def("eWiseAdd", &eWiseAddMatrix<GBMatrix, GBMatrix, GBMatrix>);
    m.def("eWiseAdd", &eWiseAddVector<GBVector, GBVector, GBVector>);
    m.def("eWiseMult", &eWiseMultMatrix<GBMatrix, GBMatrix, GBMatrix>);
    m.def("eWiseMult", &eWiseMultVector<GBVector, GBVector, GBVector>);
}
