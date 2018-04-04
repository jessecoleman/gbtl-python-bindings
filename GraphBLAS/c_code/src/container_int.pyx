cimport container as c
from libcpp.vector cimport vector
from libcpp cimport bool

cdef class Matrix:
    
    def __cinit__(self, 
            vector[int] rows, 
            vector[int] cols, 
            vector[int] vals, 
            tuple shape):

        self.cobj = new c.Matrix[int](shape[0], shape[1])
        if self.cobj == NULL:
            raise MemoryError("Not enough memory.")

        self.cobj.build(rows.begin(), cols.begin(), vals.begin(), vals.size())

    def __dealloc__(self):
        del self.cobj

    def __getitem__(self, index):
        return self.cobj.extractElement(index[0], index[1])

    def __setitem__(self, index, value):
        self.cobj.setElement(index[0], index[1], value)

    @property
    def nvals(self):
        return self.cobj.nvals()

    @property
    def shape(self):
        return (self.cobj.nrows(), self.cobj.ncols())


cdef class Vector:
    
    cdef c.Vector[int] *cobj

    def __cinit__(self, 
            vector[int] indices,
            vector[int] vals, 
            int shape):

        self.cobj = new c.Vector[int](shape)
        if self.cobj == NULL:
            raise MemoryError("Not enough memory.")

        self.cobj.build(indices.begin(), vals.begin(), vals.size())

    def __dealloc__(self):
        del self.cobj

    def __getitem__(self, index):
        return self.cobj.extractElement(index)

    def __setitem__(self, index, value):
        self.cobj.setElement(index, value)

    @property
    def nvals(self):
        return self.cobj.nvals()

    @property
    def shape(self):
        return (self.cobj.size(),)


