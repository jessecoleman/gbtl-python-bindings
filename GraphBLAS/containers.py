from scipy import sparse
from GraphBLAS import compile_c as c
from GraphBLAS.semirings import ArithmeticSemiring

class Matrix(ArithmeticSemiring._MatrixOps):
    def __init__(self, m):
        # get C++ module with declaration for Matrix class
        self.dtype = c._get_type(m)
        a = c._get_module(self.dtype)

        # construct from scipy sparse matrix
        if (sparse.issparse(m)):
            d = m.tocoo(copy=False)
            self.mat = a.init_sparse_matrix(d.shape[0], d.shape[1], d.row, d.col, d.data)
        # construct from tuple of arrays
        else:
            self.mat = a.init_sparse_matrix(max(m[1]) + 1, max(m[2]) + 1, m[1], m[2], m[0])

    def __str__(self):
        return self.mat.__str__()

    #def __matmul__(self, o):
    #    return self.mat @ o.mat

    def init_ring(self):
        c._build_semiring("INT", "TEST")

class Vector(ArithmeticSemiring._VectorOps):
    def __init__(self, v):
        self.dtype = c._get_type(v)
        a = c._get_module(self.dtype)
        self.vec = a.Vector(v)

    def __str__(self):
        return self.vec.__str__()
