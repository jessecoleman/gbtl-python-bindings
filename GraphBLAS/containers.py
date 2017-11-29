from scipy import sparse
from GraphBLAS import compile_c as c

class _MatrixOps(object):
    def __add__(self, other):
        return self._sr("eWiseAdd", self, other)

    def __mul__(self, other):
        return self._sr("eWiseMult", self, other) 

    def __matmul__(self, other):
        return self._sr("mxm", self, other)

    def __rmatmul__(self, other):
        return self._sr("mxm", other, self)

class _VectorOps(object):
    def __add__(self, other):
        return self._sr("eWiseAdd", self.vec, other.vec)

    def __mul__(self, other):
        return self._sr("eWiseMult", self.vec, other.vec)

    def __matmul__(self, other):
        return self._sr("vxm", self.vec, other.mat)

    def __rmatmul__(self, other):
        return self._sr("mxv", other.mat, self.vec)

class Matrix(_MatrixOps):

    def __init__(self, m=None, shape=None, dtype=None):
        if m is None and (shape is None or dtype is None):
            raise ValueError("Please provide matrix or shape and dtype")

        self._mask = None

        # copy constructor
        if isinstance(m, Matrix):
            self.shape = m.shape
            self.dtype = m.dtype
            self.mat = m.mat
            return

        else:
            # get C++ module with declaration for Matrix class
            self.dtype = dtype if dtype is not None else c._get_type(m)
            module = c.get_container(self.dtype)

        # construct from scipy sparse matrix
        if (sparse.issparse(m)):
            d = m.tocoo(copy=False)
            self.shape = d.shape
            self.mat = module.init_sparse_matrix(
                    self.shape[0], self.shape[1], 
                    d.row, d.col, d.data
            )

        # construct from tuple of arrays
        elif isinstance(m, tuple) and len(m) == 3:
            self.shape = shape if shape is not None else (max(m[-1])+1, max(m[2])+1)
            self.mat = module.init_sparse_matrix(
                    self.shape[0], self.shape[1], 
                    m[1], m[2], m[0]
            )

        # construct empty matrix
        else:
            self.shape = shape
            self.mat = module.init_sparse_matrix(
                    self.shape[0], self.shape[1],
                    [], [], []
            )

    def __str__(self):
        return str(self.mat)

    def __getitem__(self, item):
        # TODO typecheck for boolean matrix
        if isinstance(item, Matrix):
            self._mask = item.mat
        return self

    # self[item] += assign
    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    def __setitem__(self, item, assign):
        print("set item")
        print(item._mask)
        if isinstance(item, Matrix):
            self._mask = item.mat
        elif isinstance(item, tuple):
            self._mask = item.mat
            self._repl = True

    # TODO double check that ~ copies instead of referencing
    def __invert__(self):
        mat = self.mat
        self.mat = ~self.mat
        inverted = Matrix(self)
        self.mat = mat
        return inverted

class Vector(_VectorOps):

    def __init__(self, v, dtype=None, shape=None):
        self.dtype = dtype if dtype is not None else c._get_type(v)
        self.shape = shape if shape is not None else max(v) + 1
        module = c.get_container(self.dtype)
        self.vec = module.Vector(v)

    def __str__(self):
        return str(self)

    
