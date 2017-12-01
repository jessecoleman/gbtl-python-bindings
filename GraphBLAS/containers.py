import numpy as np
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
        return self._sr("eWiseAdd", self, other)

    def __mul__(self, other):
        return self._sr("eWiseMult", self, other)

    def __matmul__(self, other):
        return self._sr("vxm", self, other)

    def __rmatmul__(self, other):
        return self._sr("mxv", other, self)

class Matrix(_MatrixOps):

    def __init__(self, m=None, shape=None, dtype=None):
        # require matrix or shape and type
        if m is None and (shape is None or dtype is None):
            raise ValueError("Please provide matrix or shape and dtype")

        self._mask = None
        self._ac = None

        # copy constructor, no compilation needed
        # TODO mat is currently copied by reference, not value
        if isinstance(m, tuple) and len(m) == 3:
            self.mat, self.shape, self.dtype = m
            return

        # get C++ module with declaration for Matrix class
        if dtype is not None: self.dtype = dtype 
        else: self.dtype = c._get_type(m)
        module = c.get_container(self.dtype)

        # construct from scipy sparse matrix
        if (sparse.issparse(m)):
            d = m.tocoo(copy=False)
            self.shape = d.shape
            self.mat = module.init_sparse_matrix(
                    self.shape[0], self.shape[1], 
                    d.row, d.col, d.data
            )

        # construct from tuple of arrays (data, (rows, cols))
        elif isinstance(m, tuple) and len(m) == 2:
            data, idx = m
            row_idx, col_idx = idx
            if shape is not None: self.shape = shape
            else: self.shape = (max(row_idx) + 1, max(col_idx) + 1)
            self.mat = module.init_sparse_matrix(
                    self.shape[0], self.shape[1], 
                    row_idx, col_idx, data
            )

        # construct empty matrix from shape and dtype
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
            if item.dtype == bool: 
                self._mask = item.mat
            else:
                raise TypeError("Mask must be applied with boolean Matrix")
        # if replace flag is set
        elif isinstance(item, tuple) and len(item) == 2:
            if item[1].dtype == bool: 
                self._mask, self._replace = item
            else:
                raise TypeError("Mask must be applied with boolean Matrix")
        # denote self as masked object
        return self

    # self[item] += assign
    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    def __setitem__(self, item, assign):
        if item == slice(None, None, None):
            print("NoMask")
        elif isinstance(item, tuple) and len(item) == 2\
                and isinstance(item[0], slice)\
                and isinstance(item[1], slice):
            row_idx, col_idx, vals = [], [], []
            for i in range(*item[0].indices(self.shape[0])):
                for j in range(*item[1].indices(self.shape[1])):
                    row_idx.append(i)
                    col_idx.append(j)
                    vals.append(True)
            print(Matrix((vals, (row_idx, col_idx)), shape=self.shape, dtype=bool))


    # TODO double check that ~ copies instead of referencing
    def __invert__(self):
        return Vector((~self.mat, self.shape, self.dtype))

    @property
    def nvals(self):
        return self.vec.nvals()

class Vector(_VectorOps):

    def __init__(self, v=None, shape=None, dtype=None):
        # require vector or shape and type
        if v is None and (shape is None or dtype is None):
            raise ValueError("Please provide vector or shape and dtype")

        self._mask = None
        self._ac = None

        # copy constructor, no compilation needed
        if isinstance(v, tuple) and len(v) == 3:
            self.vec, self.shape, self.dtype = v
            # TODO temporary
            self.mat = self.vec
            return

        # get C++ module with declaration for Matrix class
        if dtype is not None: self.dtype = dtype 
        else: self.dtype = c._get_type(v)
        module = c.get_container(self.dtype)

        if isinstance(v, list):
            self.shape = (len(v),)
            self.vec = module.Vector(v)

        elif isinstance(v, np.ndarray):
            self.shape = v.shape
            self.vec = module.Vector(self)

        # construct from tuple of arrays (data, vals)
        elif isinstance(v, tuple) and len(v) == 2:
            data, idx = v
            if shape is not None: self.shape = shape
            else: self.shape = (max(idx) + 1,)
            self.vec = module.Vector(self.shape[0], idx, data)

        else:
            self.shape = shape
            self.vec = module.init_sparse_vector(
                    self.shape[0], [], []
            )

        self.mat = self.vec

    def __setitem__(self, item, assign):
        if item == slice(None, None, None):
            print("NoMask")
            ...
        elif isinstance(item, slice): 
            idx, vals = [], []
            # TODO optimize building matrix
            for i in range(*item.indices(self.shape[0])):
                    idx.append(i)
                    vals.append(True)
            self._mask = Vector(
                    (vals, idx), 
                    shape=self.shape, 
                    dtype=bool
                )
            return self

    def __invert__(self):
        return Vector((~self.vec, self.shape, self.dtype))

    def __getitem__(self, item):
        return self

    def __str__(self):
        return str(self.vec)

    @property
    def nvals(self):
        return self.vec.nvals()

    def __len__(self):
        return self.vec.size()
    
