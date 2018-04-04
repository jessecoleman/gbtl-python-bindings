cimport container as c


cdef class Matrix:
    
    cdef c.Matrix[int] *cobj


