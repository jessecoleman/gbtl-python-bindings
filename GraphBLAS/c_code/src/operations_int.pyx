from libcpp cimport bool
cimport operations as c_operations
cimport container as c_container
cimport container_int as c
import container_int as c


cdef class NoAccumulate:
    cdef c_operations.NoAccumulate *cobj

cdef class ArithmeticSemiring:
    cdef c_operations.ArithmeticSemiring[int] *cobj

cdef class PlusMonoid:
    cdef c_operations.PlusMonoid[int] *cobj

cdef mxm(
        c.Matrix C,
        c.Matrix M,
        NoAccumulate accum,
        ArithmeticSemiring op,
        c.Matrix A,
        c.Matrix B,
        bool replace_flag):

    c_operations.mxm[
        c_container.Matrix[int],
        c_container.Matrix[int],
        c_operations.NoAccumulate,
        c_operations.ArithmeticSemiring[int],
        c_container.Matrix[int],
        c_container.Matrix[int]
    ](
            C.cobj[0], 
            M.cobj[0], 
            accum.cobj[0], 
            op.cobj[0], 
            A.cobj[0], 
            B.cobj[0], 
            replace_flag)

cdef eWiseAddMatrix(
        c.Matrix C,
        c.Matrix M,
        NoAccumulate accum,
        PlusMonoid op,
        c.Matrix A,
        c.Matrix B,
        bool replace_flag):

     c_operations.eWiseAdd[
        int,
        c_container.Matrix[int],
        c_operations.NoAccumulate,
        c_operations.PlusMonoid[int],
        c_container.Matrix[int],
        c_container.Matrix[int]
    ](
            C.cobj[0], 
            M.cobj[0], 
            accum.cobj[0], 
            op.cobj[0], 
            A.cobj[0], 
            B.cobj[0], 
            replace_flag)

