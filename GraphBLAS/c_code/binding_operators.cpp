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
#elif defined(A_MATRIXCOMPLEMENT)
typedef GraphBLAS::MatrixComplementView<GraphBLAS::Matrix<A_TYPE>> AMatrixT;
#elif defined(A_MATRIXTRANSPOSE)
typedef GraphBLAS::TransposeView<GraphBLAS::Matrix<A_TYPE>> AMatrixT;
#elif defined(A_VECTOR)
typedef GraphBLAS::Vector<A_TYPE> UVectorT;
#elif defined(A_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<A_TYPE>> UVectorT;
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
#elif defined(B_VECTOR)
typedef GraphBLAS::Vector<B_TYPE> VVectorT;
#elif defined(B_VECTORCOMPLEMENT)
typedef GraphBLAS::VectorComplementView<GraphBLAS::Vector<B_TYPE>> VVectorT;
#endif

// out type
#if defined(C_MATRIX)
typedef GraphBLAS::Matrix<C_TYPE> CMatrixT;
#elif defined(C_VECTOR)
typedef GraphBLAS::Vector<C_TYPE> WVectorT;
#elif defined(C_VALUE)
typedef C_TYPE CValueT;
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
#elif defined(M__NOMASK)
typedef GraphBLAS::NoMask MMatrixT;
typedef GraphBLAS::NoMask MVectorT;
#endif

// for assign and extract operations
#if defined(ROW_INDICES)
typedef std::vector<ROW_INDICES_TYPE> RSequenceT;
#elif defined(ROW_INDEX)
typedef ROW_INDEX_TYPE RIndexT
#endif

#if defined(COL_INDICES)
typedef std::vector<COL_INDICES_TYPE> CSequenceT;
#elif defined(COL_INDEX)
typedef COL_INDEX_TYPE CIndexT
#endif

#if defined(INDICES)
typedef std::vector<INDICES_TYPE> SequenceT;
#endif

#if defined(NO_ACCUM)
typedef GraphBLAS::NoAccumulate AccumT;
#else
typedef GraphBLAS::ACCUM_BINARYOP<C_TYPE> AccumT;
#endif

#if defined(MIN_IDENTITY)
#define A_IDENTITY std::numeric_limits<C_TYPE>::max()
#elif defined(SEMIRING) || defined(REDUCE)
#define A_IDENTITY IDENTITY
#endif

#if defined(APPLY) \
 && defined(BOUND_CONST)
typedef GraphBLAS::BinaryOp_Bind2nd<A_TYPE, GraphBLAS::UNARY_OP<A_TYPE, C_TYPE>> ApplyT;
#elif defined(APPLY)
#define BOUND_CONST
typedef GraphBLAS::UNARY_OP<A_TYPE, C_TYPE> ApplyT;
#endif

// TODO ensure that correct types compile these operators
// operations that need monoid
#if defined(A_BINARY_OP) \
 || defined(MONOID) \
 || defined(SEMIRING) \
 || defined(REDUCE)
typedef GraphBLAS::A_BINARY_OP<A_TYPE> AddBinaryOp;
#endif
#if defined(MONOID) \
 || defined(SEMIRING)
GEN_GRAPHBLAS_MONOID(Monoid, GraphBLAS::A_BINARY_OP, A_IDENTITY)
typedef Monoid<C_TYPE> MonoidT;
#endif
#if defined(SEMIRING)
typedef GraphBLAS::M_BINARY_OP<A_TYPE, B_TYPE, C_TYPE> MultBinaryOp;
GEN_GRAPHBLAS_SEMIRING(Semiring, Monoid, GraphBLAS::M_BINARY_OP)
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
        WVectorT       &C,
        MVectorT const &M,
        AMatrixT const &A,
        VVectorT const &B,
        bool            replace_flag
    )
{ GraphBLAS::mxv(C, M, AccumT(), SemiringT(), A, B, replace_flag); }

#elif defined(VXM)
void vxm(
        WVectorT       &C,
        MVectorT const &M,
        UVectorT const &A,
        BMatrixT const &B,
        bool            replace_flag
    )
{ GraphBLAS::vxm(C, M, AccumT(), SemiringT(), A, B, replace_flag); }

#elif defined(EWISEADDMATRIX)
void eWiseAddMatrix(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        BMatrixT const &B,
        bool            replace_flag
    )
{ GraphBLAS::eWiseAdd(C, M, AccumT(), AddBinaryOp(), A, B, replace_flag); }

#elif defined(EWISEADDVECTOR)
void eWiseAddVector(
        WVectorT       &C,
        MVectorT const &M,
        UVectorT const &A,
        VVectorT const &B,
        bool            replace_flag
    )
{ GraphBLAS::eWiseAdd(C, M, AccumT(), AddBinaryOp(), A, B, replace_flag); }

// TODO decide if MultBinaryOp is necessary or if any binary op is OK
#elif defined(EWISEMULTMATRIX)
void eWiseMultMatrix(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        BMatrixT const &B,
        bool            replace_flag
    )
{ GraphBLAS::eWiseMult(C, M, AccumT(), MultBinaryOp(), A, B, replace_flag); }

#elif defined(EWISEMULTVECTOR)
void eWiseMultVector(
        WVectorT       &C,
        MVectorT const &M,
        UVectorT const &A,
        VVectorT const &B,
        bool            replace_flag
    )
{ GraphBLAS::eWiseMult(C, M, AccumT(), MultBinaryOp(), A, B, replace_flag); }

#elif defined(APPLY) \
   && defined(C_MATRIX)
void applyMatrix(
        CMatrixT       &C,
        MMatrixT const &M,
        AMatrixT const &A,
        bool            replace_flag
    )
{ GraphBLAS::apply(C, M, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }

#elif defined(APPLY) \
   && defined(C_VECTOR)
void applyVector(
        WVectorT       &C,
        MVectorT const &M,
        UVectorT const &A, 
        bool            replace_flag
    )
{ GraphBLAS::apply(C, M, AccumT(), ApplyT(BOUND_CONST), A, replace_flag); }

#elif defined(REDUCE) \
   && defined(A_MATRIX) \
   && defined(C_VECTOR)
void reduceMatrixVector(
        WVectorT       &C, 
        MVectorT const &M, 
        AMatrixT const &A,
        bool            replace_flag
    ) 
{ reduce(C, M, AccumT(), MonoidT(), A, replace_flag); }

#elif defined(REDUCE) \
   && defined(A_MATRIX)
CValueT reduceMatrix(
        CValueT        &C, 
        AMatrixT const &A
    ) {
    reduce(C, AccumT(), MonoidT(), A);
    return C;
} 
 
#elif defined(REDUCE) \
   && defined(A_VECTOR)
CValueT reduceVector(
        CValueT        &C, 
        UVectorT const &A
    ) {
    reduce(C, AccumT(), MonoidT(), A);
    return C;
}

#elif defined(EXTRACT) \
   && defined(C_MATRIX) \
   && defined(ROW_INDICES) \
   && defined(COL_INDICES)
void extractMatrix(
        CMatrixT         &C,
        MMatrixT   const &M,
        AMatrixT   const &A,
        RSequenceT const &row_indices,
        CSequenceT const &col_indices,
        bool              replace_flag
    )
{ extract(C, M, AccumT(), A, row_indices, col_indices, replace_flag); }

#elif defined(EXTRACT) \
   && defined(C_MATRIX) \
   && defined(ROW_INDICES) \
   && defined(COL_INDEX)
void extractMatrixCol(
        WVectorT         &C,
        MVectorT   const &M,
        AMatrixT   const &A,
        RSequenceT const &row_indices,
        CIndexT    const &col_index,
        bool              replace_flag
    )
{ extract(C, M, AccumT(), A, row_indices, col_index, replace_flag); }

#elif defined(EXTRACT) \
   && defined(C_VECTOR) \
   && defined(INDICES)
void extractVector(
        WVectorT        &C,
        MVectorT  const &M,
        UVectorT  const &A,
        SequenceT const &indices,
        bool             replace_flag
    )
{ extract(C, M, AccumT(), A, indices, replace_flag); }

#elif defined(ASSIGN) \
   && defined(C_MATRIX) \
   && defined(A_MATRIX) \
   && defined(ROW_INDICES) \
   && defined(COL_INDICES)
void assignMatrix(
        CMatrixT         &C,
        MMatrixT   const &M,
        AMatrixT   const &A,
        RSequenceT const &row_indices,
        CSequenceT const &col_indices,
        bool              replace_flag
    )
{ assign(C, M, AccumT(), A, row_indices, col_indices, replace_flag); }

#elif defined(ASSIGN) \
   && defined(C_MATRIX) \
   && defined(A_MATRIX) \
   && defined(ROW_INDICES) \
   && defined(COL_INDEX)
void assignMatrixCol(
        CMatrixT         &C,
        MMatrixT   const &M,
        AMatrixT   const &A,
        RSequenceT const &row_indices,
        CIndexT    const &col_index,
        bool              replace_flag
    )
{ assign(C, M, AccumT(), A, row_indices, col_index, replace_flag); }

#elif defined(ASSIGN) \
   && defined(C_MATRIX) \
   && defined(A_MATRIX) \
   && defined(ROW_INDEX) \
   && defined(COL_INDICES)
void assignMatrixRow(
        CMatrixT         &C,
        MMatrixT   const &M,
        AMatrixT   const &A,
        RIndexT    const &row_index,
        CSequenceT const &col_indices,
        bool              replace_flag
    )
{ assign(C, M, AccumT(), A, row_index, col_indices, replace_flag); }

#elif defined(ASSIGN) \
   && defined(C_MATRIX) \
   && defined(A_VALUE)
void assignMatrixConst(
        CMatrixT          &C,
        MMatrixT    const &M,
        AValueT     const &A,
        RSequenceT  const &row_indices,
        CSequenceT  const &col_indices,
        bool               replace_flag
    )
{ assign(C, M, AccumT(), A, row_indices, col_indices, replace_flag); }

#elif defined(ASSIGN) \
   && defined(C_VECTOR) \
   && defined(A_VECTOR) \
   && defined(INDICES)
void assignVector(
        WVectorT        &C,
        MVectorT  const &M,
        UVectorT  const &A,
        SequenceT const &indices,
        bool             replace_flag
    )
{ assign(C, M, AccumT(), A, indices, replace_flag); }

#elif defined(ASSIGN) \
   && defined(C_VECTOR) \
   && defined(A_VALUE)
void assignVectorConst(
        WVectorT        &C,
        MVectorT  const &M,
        AValueT   const &A,
        SequenceT const &indices,
        bool             replace_flag
    )
{ assign(C, M, AccumT(), A, indices, replace_flag); }

#elif defined(TRANSPOSE)
void transpose(
        CMatrixT       &C,
        MMatrixT const &Mask,
        AMatrixT const &A,
        bool            replace_flag
    )
{ transposed(C, M, AccumT(), A, replace_flag); }

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

#elif defined(APPLYMATRIX)
    m.def("applyMatrix", &applyMatrix, "C"_a, "M"_a, "A"_a, "replace_flag"_a);
#elif defined(APPLYVECTOR)
    m.def("applyVector", &applyVector, "C"_a, "M"_a, "A"_a, "replace_flag"_a);

#elif defined(REDUCEMATRIXVECTOR)
    m.def("reduceMatrixVector", &reduceMatrixVector, "C"_a, "M"_a, "A"_a, "replace_flag"_a);
#elif defined(REDUCEMATRIX)
    m.def("reduceMatrix", &reduceMatrix, "C"_a, "A"_a);
#elif defined(REDUCEVECTOR)
    m.def("reduceVector", &reduceVector, "C"_a, "A"_a);

#elif defined(EXTRACTSUBMATRIX)
    m.def("extractSubmatrix", &extractMatrix, "C"_a, "M"_a, "A"_a, "row_indices"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(EXTRACTMATRIXCOL)
    m.def("extractMatrixCol", &extractMatrixCol, "C"_a, "M"_a, "A"_a, "row_indices"_a, "col_index"_a, "replace_flag"_a);
#elif defined(EXTRACTVECTOR)
    m.def("extractVector", &extractVector, "C"_a, "M"_a, "A"_a, "indices"_a, "replace_flag"_a);

#elif defined(ASSIGN) && defined(C_MATRIX) && defined(A_MATRIX) && defined(ROW_INDICES) && defined(COL_INDICES)
    m.def("assign", &assignMatrix, "C"_a, "M"_a, "A"_a, "row_indices"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(ASSIGN) && defined(C_MATRIX) && defined(A_MATRIX) && defined(ROW_INDICES) && defined(COL_INDEX)
    m.def("assign", &assignMatrixCol, "C"_a, "M"_a, "A"_a, "row_indices"_a, "col_index"_a, "replace_flag"_a);
#elif defined(ASSIGN) && defined(C_MATRIX) && defined(A_MATRIX) && defined(ROW_INDEX) && defined(COL_INDICES)
    m.def("assign", &assignMatrixRow, "C"_a, "M"_a, "A"_a, "row_index"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(ASSIGN) && defined(C_MATRIX) && defined(A_VALUE)
    m.def("assign", &assignMatrixConst, "C"_a, "M"_a, "A"_a, "row_indices"_a, "col_indices"_a, "replace_flag"_a);
#elif defined(ASSIGN) && defined(C_VECTOR) && defined(A_VECTOR) && defined(INDICES)
    m.def("assign", &assignVector, "C"_a, "M"_a, "A"_a, "indices"_a, "replace_flag"_a);
#elif defined(ASSIGN) && defined(C_VECTOR) && defined(A_VALUE)
    m.def("assign", &assignVectorConst, "C"_a, "M"_a, "A"_a, "indices"_a, "replace_flag"_a);
#elif defined(TRANSPOSE)
    m.def("transpose", &transpose, "C"_a, "M"_a, "A"_a, "replace_flag"_a);


#endif
}
