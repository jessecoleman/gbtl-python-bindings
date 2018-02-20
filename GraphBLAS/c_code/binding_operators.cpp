#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "graphblas/graphblas.hpp"
#include "graphblas/algebra.hpp"
//#include "graphblas.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// left type
#if defined(A_MATRIX)
typedef GraphBLAS::Matrix<A_TYPE> AMatrixT;
#elif defined(A_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<A_TYPE>> AMatrixT;
#elif defined(A_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<A_TYPE>> AMatrixT;
#elif defined(U_VECTOR)
typedef GraphBLAS::Vector<U_TYPE> UVectorT;
#elif defined(U_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<U_TYPE>> UVectorT;
#elif defined(A_VALUE)
typedef A_TYPE AValueT;
#endif

// right type
#if defined(B_MATRIX)
typedef GraphBLAS::Matrix<B_TYPE> BMatrixT;
#elif defined(B_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<B_TYPE>> BMatrixT;
#elif defined(B_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<B_TYPE>> BMatrixT;
#elif defined(V_VECTOR)
typedef GraphBLAS::Vector<V_TYPE> VVectorT;
#elif defined(V_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<V_TYPE>> VVectorT;
#endif

// out type
#if defined(C_MATRIX)
typedef GraphBLAS::Matrix<C_TYPE> CMatrixT;
#elif defined(W_VECTOR)
typedef GraphBLAS::Vector<W_TYPE> WVectorT;
#elif defined(S_VALUE)
typedef S_TYPE SValueT;
#endif

// mask type
#if defined(M_MATRIX)
typedef GraphBLAS::Matrix<M_TYPE> MMatrixT;
#elif defined(M_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<M_TYPE>> MMatrixT;
#elif defined(M_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<M_TYPE>> MMatrixT;
#elif defined(M_VECTOR)
typedef GraphBLAS::Vector<M_TYPE> MVectorT;
#elif defined(M_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<M_TYPE>> MVectorT;
#elif defined(M_NOMASK)
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask MVectorT;
#endif

// for assign and extract operations
#if defined(ROW_INDICES)
typedef std::vector<ROW_INDICES_TYPE> RSequenceT;
#elif defined(ROW_INDEX_VALUE)
typedef ROW_INDEX_TYPE RIndexT;
#endif

#if defined(COL_INDICES)
typedef std::vector<COL_INDICES_TYPE> CSequenceT;
#elif defined(COL_INDEX_VALUE)
typedef COL_INDEX_TYPE CIndexT;
#endif

#if defined(INDICES)
typedef std::vector<INDICES_TYPE> SequenceT;
#endif

// TODO check functionality
#if defined(ALL_INDICES)
typedef GraphBLAS::AllIndices SequenceT;
#endif
#if defined(ALL_ROW_INDICES)
typedef GraphBLAS::AllIndices RSequenceT;
#endif
#if defined(ALL_COL_INDICES)
typedef GraphBLAS::AllIndices CSequenceT;
#endif

#if defined(NO_ACCUM)
typedef GraphBLAS::NoAccumulate AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<C_TYPE> AccumT;
#endif

#if defined(MIN_IDENTITY)
#define A_IDENTITY std::numeric_limits<C_TYPE>::max()
#elif defined(MONOID) || defined(SEMIRING)
#define A_IDENTITY ADD_IDENTITY
#endif

#if (defined(APPLYMATRIX) || defined(APPLYVECTOR)) && defined(BOUND_CONST)
typedef GraphBLAS::BinaryOp_Bind2nd<A_TYPE, GraphBLAS::UNARY_OP<A_TYPE, C_TYPE>> ApplyT;
#elif defined(APPLYMATRIX) || defined(APPLYVECTOR)
#define BOUND_CONST
typedef GraphBLAS::UNARY_OP<A_TYPE, C_TYPE> ApplyT;
#endif

#if defined(U_TYPE)
#define A_TYPE U_TYPE
#endif
#if defined(V_TYPE)
#define B_TYPE V_TYPE
#endif
#if defined(W_TYPE)
#define C_TYPE W_TYPE
#endif

// TODO ensure that correct types compile these operators
// operations that need monoid
#if defined(BINARY_OP) 
typedef GraphBLAS::BINARY_OP<A_TYPE> BinaryOp;
#elif defined(MONOID) || defined(SEMIRING)
typedef GraphBLAS::ADD_BINARY_OP<A_TYPE> BinaryOp;
GEN_GRAPHBLAS_MONOID(Monoid, GraphBLAS::ADD_BINARY_OP, A_IDENTITY)
typedef Monoid<C_TYPE> MonoidT;
#endif
#if defined(SEMIRING)
GEN_GRAPHBLAS_SEMIRING(Semiring, Monoid, GraphBLAS::MULT_BINARY_OP)
typedef Semiring<A_TYPE, B_TYPE, C_TYPE> SemiringT;
#endif

#if defined(MXM)
void mxm(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        BMatrixT const &B,
        bool            replace_flag
    )
{ GraphBLAS::mxm(C, M, AccumT(), SemiringT(), A, B, replace_flag); }

#elif defined(MXV)
void mxv(
        WVectorT       &w,
        MVectorT const &M,
        AMatrixT const &A,
        VVectorT const &v,
        bool            replace_flag
    )
{ GraphBLAS::mxv(w, M, AccumT(), SemiringT(), A, v, replace_flag); }

#elif defined(VXM)
void vxm(
        WVectorT       &w,
        MVectorT const &M,
        UVectorT const &u,
        BMatrixT const &B,
        bool            replace_flag
    )
{ GraphBLAS::vxm(w, M, AccumT(), SemiringT(), u, B, replace_flag); }

#elif defined(EWISEADDMATRIX)
void eWiseAddMatrix(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        BMatrixT const &B,
        bool            replace_flag
    )
{ GraphBLAS::eWiseAdd(C, M, AccumT(), BinaryOp(), A, B, replace_flag); }

#elif defined(EWISEADDVECTOR)
void eWiseAddVector(
        WVectorT       &w,
        MVectorT const &M,
        UVectorT const &u,
        VVectorT const &v,
        bool            replace_flag
    )
{ GraphBLAS::eWiseAdd(w, M, AccumT(), BinaryOp(), u, v, replace_flag); }

#elif defined(EWISEMULTMATRIX)
void eWiseMultMatrix(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        BMatrixT const &B,
        bool            replace_flag
    )
{ GraphBLAS::eWiseMult(C, M, AccumT(), BinaryOp(), A, B, replace_flag); }

#elif defined(EWISEMULTVECTOR)
void eWiseMultVector(
        WVectorT       &w,
        MVectorT const &M,
        UVectorT const &u,
        VVectorT const &v,
        bool            replace_flag
    )
{ GraphBLAS::eWiseMult(w, M, AccumT(), BinaryOp(), u, v, replace_flag); }

#elif defined(APPLYMATRIX)
void applyMatrix(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        bool            replace_flag
    )
{ GraphBLAS::apply(C, M, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }

#elif defined(APPLYVECTOR)
void applyVector(
        WVectorT       &w,
        MVectorT const &m,
        UVectorT const &u, 
        bool            replace_flag
    )
{ GraphBLAS::apply(w, m, AccumT(), ApplyT(BOUND_CONST), u, replace_flag); }

#elif defined(REDUCEMATRIXVECTOR)
void reduceMatrixVector(
        WVectorT       &w, 
        MVectorT const &m, 
        AMatrixT const &A,
        bool            replace_flag
    ) 
{ reduce(w, m, AccumT(), MonoidT(), A, replace_flag); }

#elif defined(REDUCEMATRIX)
SValueT reduceMatrix(
        SValueT        &s, 
        AMatrixT const &A
    ) {
    reduce(s, AccumT(), MonoidT(), A);
    return s;
} 
 
#elif defined(REDUCEVECTOR)
SValueT reduceVector(
        SValueT        &s, 
        UVectorT const &u
    ) {
    reduce(s, AccumT(), MonoidT(), u);
    return s;
}

#elif defined(EXTRACTSUBMATRIX)
void extractSubmatrix(
        CMatrixT         &C,
        MMatrixT   const &M,
        AMatrixT   const &A,
        RSequenceT const &row_indices,
        CSequenceT const &col_indices,
        bool              replace_flag
    )
{ extract(C, M, AccumT(), A, row_indices, col_indices, replace_flag); }

#elif defined(EXTRACTMATRIXCOL)
void extractMatrixCol(
        WVectorT         &w,
        MVectorT   const &m,
        AMatrixT   const &A,
        RSequenceT const &row_indices,
        CIndexT    const &col_index,
        bool              replace_flag
    )
{ extract(w, m, AccumT(), A, row_indices, col_index, replace_flag); }

#elif defined(EXTRACTSUBVECTOR)
void extractSubvector(
        WVectorT        &w,
        MVectorT  const &m,
        UVectorT  const &u,
        SequenceT const &indices,
        bool             replace_flag
    )
{ extract(w, m, AccumT(), u, indices, replace_flag); }

#elif defined(ASSIGNSUBMATRIX)
void assignSubmatrix(
        CMatrixT         &C,
        MMatrixT   const &M,
        AMatrixT   const &A,
        RSequenceT const &row_indices,
        CSequenceT const &col_indices,
        bool              replace_flag
    )
{ assign(C, M, AccumT(), A, row_indices, col_indices, replace_flag); }

#elif defined(ASSIGNMATRIXCOL)
void assignMatrixCol(
        CMatrixT         &C,
        MMatrixT   const &M,
        UVectorT   const &u,
        RSequenceT const &row_indices,
        CIndexT    const &col_index,
        bool              replace_flag
    )
{ assign(C, M, AccumT(), u, row_indices, col_index, replace_flag); }

#elif defined(ASSIGNMATRIXROW)
void assignMatrixRow(
        CMatrixT         &C,
        MMatrixT   const &M,
        UVectorT   const &u,
        RIndexT    const &row_index,
        CSequenceT const &col_indices,
        bool              replace_flag
    )
{ assign(C, M, AccumT(), u, row_index, col_indices, replace_flag); }

#elif defined(ASSIGNMATRIXCONST)
void assignMatrixConst(
        CMatrixT          &C,
        MMatrixT    const &M,
        ValueT     const &val,
        RSequenceT  const &row_indices,
        CSequenceT  const &col_indices,
        bool               replace_flag
    )
{ assign(C, M, AccumT(), val, row_indices, col_indices, replace_flag); }

#elif defined(ASSIGNSUBVECTOR)
void assignSubvector(
        WVectorT        &w,
        MVectorT  const &m,
        UVectorT  const &u,
        SequenceT const &indices,
        bool             replace_flag
    )
{ assign(w, m, AccumT(), u, indices, replace_flag); }

#elif defined(ASSIGNVECTORCONST)
void assignVectorConst(
        WVectorT        &w,
        MVectorT  const &m,
        ValueT   const &val,
        SequenceT const &indices,
        bool             replace_flag
    )
{ assign(w, m, AccumT(), val, indices, replace_flag); }

#elif defined(TRANSPOSE)
void transpose(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        bool            replace_flag
    )
{ transpose(C, M, AccumT(), A, replace_flag); }

#endif

PYBIND11_MODULE(MODULE, m) {

#if defined(MONOID) || defined(SEMIRING)
    py::class_<MonoidT>(m, "Monoid", py::module_local())
        .def(py::init<>())
        .def("identity", &MonoidT::identity);
#endif
#if defined(SEMIRING)
    py::class_<SemiringT>(m, "Semiring", py::module_local())
        .def(py::init<>())
        .def("add", &SemiringT::add)
        .def("mult", &SemiringT::mult)
        .def("zero", &SemiringT::zero);
#endif

#if defined(MXM)
    m.def("mxm", &mxm, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(MXV)
    m.def("mxv", &mxv, "w"_a, "m"_a, "A"_a, "v"_a, "replace_flag"_a);
#elif defined(VXM)
    m.def("vxm", &vxm, "w"_a, "m"_a, "u"_a, "B"_a, "replace_flag"_a);

#elif defined(EWISEADDMATRIX)
    m.def("eWiseAddMatrix", &eWiseAddMatrix, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(EWISEADDVECTOR)
    m.def("eWiseAddVector", &eWiseAddVector, "w"_a, "m"_a, "u"_a, "v"_a, "replace_flag"_a);
#elif defined(EWISEMULTMATRIX)
    m.def("eWiseMultMatrix", &eWiseMultMatrix, "C"_a, "M"_a, "A"_a, "B"_a, "replace_flag"_a);
#elif defined(EWISEMULTVECTOR)
    m.def("eWiseMultVector", &eWiseMultVector, "w"_a, "m"_a, "u"_a, "v"_a, "replace_flag"_a);

#elif defined(APPLYMATRIX)
    m.def("applyMatrix", &applyMatrix, "C"_a, "M"_a, "A"_a, "replace_flag"_a);
#elif defined(APPLYVECTOR)
    m.def("applyVector", &applyVector, "w"_a, "m"_a, "u"_a, "replace_flag"_a);

#elif defined(REDUCEMATRIXVECTOR)
    m.def("reduceMatrixVector", &reduceMatrixVector, "w"_a, "m"_a, "A"_a, "replace_flag"_a);
#elif defined(REDUCEMATRIX)
    m.def("reduceMatrix", &reduceMatrix, "s"_a, "A"_a);
#elif defined(REDUCEVECTOR)
    m.def("reduceVector", &reduceVector, "s"_a, "u"_a);

#elif defined(EXTRACTSUBMATRIX)
    m.def("extractSubmatrix", &extractSubmatrix, "C"_a, "M"_a, "A"_a, "row_indices"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(EXTRACTMATRIXCOL)
    m.def("extractMatrixCol", &extractMatrixCol, "w"_a, "m"_a, "A"_a, "row_indices"_a, "col_index"_a, "replace_flag"_a);
#elif defined(EXTRACTSUBVECTOR)
    m.def("extractSubvector", &extractSubvector, "w"_a, "m"_a, "u"_a, "indices"_a, "replace_flag"_a);

#elif defined(ASSIGNSUBMATRIX)
    m.def("assignSubmatrix", &assignSubmatrix, "C"_a, "M"_a, "A"_a, "row_indices"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(ASSIGNMATRIXCOL)
    m.def("assignMatrixCol", &assignMatrixCol, "C"_a, "M"_a, "u"_a, "row_indices"_a, "col_index"_a, "replace_flag"_a);
#elif defined(ASSIGNMATRIXROW)
    m.def("assignMatrixRow", &assignMatrixRow, "C"_a, "M"_a, "u"_a, "row_index"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(ASSIGNMATRIXCONST)
    m.def("assignMatrixConst", &assignMatrixConst, "C"_a, "M"_a, "val"_a, "row_indices"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(ASSIGNSUBVECTOR)
    m.def("assignSubvector", &assignSubvector, "w"_a, "m"_a, "u"_a, "indices"_a, "replace_flag"_a);
#elif defined(ASSIGNVECTORCONST)
    m.def("assignVectorConst", &assignVectorConst, "w"_a, "m"_a, "val"_a, "indices"_a, "replace_flag"_a);
#elif defined(TRANSPOSE)
    m.def("transpose", &transpose, "C"_a, "M"_a, "A"_a, "replace_flag"_a);


#endif
}
