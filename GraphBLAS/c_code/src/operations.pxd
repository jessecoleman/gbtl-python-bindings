from libcpp cimport bool
from libcpp.vector cimport vector
from container cimport Matrix


cdef extern from "graphblas/graphblas.hpp" namespace "GraphBLAS":
 
    cdef cppclass NoAccumulate:
        pass
    
    cdef cppclass ArithmeticSemiring[T]:
        pass

    cdef cppclass PlusMonoid[T]:
        pass

    cdef void mxm[
        CMatrixT,
        MMatrixT,
        NoAccumulate,
        SemiringT,
        AMatrixT,
        BMatrixT](
            CMatrixT &C,
            const MMatrixT &M,
            NoAccumulate accum,
            SemiringT op,
            const AMatrixT &A,
            const BMatrixT &B,
            bool replace_flag)

    cdef void eWiseAdd[
        C,
        MMatrixT,
        NoAccumulate,
        BinaryOpT,
        AMatrixT,
        BMatrixT](
            Matrix[C] &C,
            const MMatrixT &M,
            NoAccumulate accum,
            BinaryOpT op,
            const AMatrixT &A,
            const BMatrixT &B,
            bool replace_flag)

