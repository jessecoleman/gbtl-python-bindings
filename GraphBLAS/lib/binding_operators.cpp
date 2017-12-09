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
<<<<<<< HEAD:GraphBLAS/lib/binding_operators.cpp
=======
// vector mask types
typedef GraphBLAS::Matrix<bool>     MMatrixT;
typedef GraphBLAS::MatrixComplementView<CMatrixT> MatrixCompT;
>>>>>>> b603f8f4b88a3605d1b6cf1fb94f9477294fce6e:GraphBLAS/lib/binding_accumulators.cpp
// vector storage type
typedef GraphBLAS::Vector<AScalarT> UVectorT;
typedef GraphBLAS::Vector<BScalarT> VVectorT;
typedef GraphBLAS::Vector<CScalarT> WVectorT;
<<<<<<< HEAD:GraphBLAS/lib/binding_operators.cpp
=======
// vector mask types
typedef GraphBLAS::Vector<bool>     MVectorT;
typedef GraphBLAS::VectorComplementView<WVectorT> VectorCompT;
>>>>>>> b603f8f4b88a3605d1b6cf1fb94f9477294fce6e:GraphBLAS/lib/binding_accumulators.cpp

// mask types
#if MASK == 0
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask MVectorT;
#elif MASK == 1
typedef GraphBLAS::Matrix<MTYPE> MMatrixT;
typedef GraphBLAS::Vector<MTYPE> MVectorT;
#elif MASK == 2
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<MTYPE>> MMatrixT;
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<MTYPE>> MVectorT;
#endif

// in case additive identity isn't plain string
#if MIN_IDENTITY == 1
#define ADD_IDNTY std::numeric_limits<CScalarT>::max()
#else
#define ADD_IDNTY ADD_IDENTITY
#endif

#if NO_ACCUM == 1
typedef GraphBLAS::ACCUM_BINARYOP AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<CScalarT> AccumT;
#endif

// create monoid and semiring from macro
GEN_GRAPHBLAS_MONOID(Monoid, GraphBLAS::ADD_BINARYOP, ADD_IDNTY)
GEN_GRAPHBLAS_SEMIRING(Semiring, Monoid, GraphBLAS::MULT_BINARYOP)

typedef Monoid<CScalarT> MonoidT;
typedef GraphBLAS::ADD_BINARYOP<AScalarT> AddBinaryOp;
typedef GraphBLAS::MULT_BINARYOP<AScalarT, BScalarT, CScalarT> MultBinaryOp;
typedef Semiring<AScalarT, BScalarT, CScalarT> SemiringT;
 
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
