import numpy as np
from scipy import sparse
from GraphBLAS import compile_c as c
import GraphBLAS.operators as ops

no_mask = c.get_container(bool).NoMask()

class Matrix(object):

    def __init__(self, m=None, shape=None, dtype=None):
        # require matrix or shape and type
        if m is None and (shape is None or dtype is None):
            raise ValueError("Please provide matrix or shape and dtype")

        self._mask = no_mask
        self._repl = False

        # copy constructor, no compilation needed
        # TODO mat is currently copied by reference, not value
        if isinstance(m, tuple) and len(m) == 3:
            self.mat, self.shape, self.dtype = m
            return

        # get C++ module with declaration for Matrix class
        if dtype is not None: self.dtype = dtype 
        else: self.dtype = c.get_type(m)
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

    def __add__(self, other):
        return ops.semring.eWiseAdd(self, other)

    def __mul__(self, other):
        return ops.semiring.eWiseMult(self, other) 

    def __matmul__(self, other):
        return ops.semiring.mxm(self, other)

    def __rmatmul__(self, other):
        return ops.semiring.mxm(other, self)

    # create callable object to accumulate
    def __iadd__(self, expr):
        print(signature(expr))
        # if already callable, add accumulator and output params
        if callable(expr):
            return lambda: expr(self, accum=ops.accumulator)
        # else turn into identity apply operator first
        elif isinstance(expr, Matrix):
            # lazily evaluates Apply object
            return lambda: ops.Identity(expr, self, accum=ops.accumulator)
        else: raise TypeError("Evaluation was not deferred")

    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    # applies mask stored in item and returns self
    def __getitem__(self, item):
        print("MASK", item)

        # if replace flag is set
        if isinstance(item, bool):
            self._repl = item
            return self

        # strip replace off end if it exists
        elif isinstance(item, tuple):
            if isinstance(item[-1], bool):
                *item, self._repl = item

        # no mask
        if item == slice(None, None, None):
            pass

        # masking with slice
        elif isinstance(item, tuple) and len(item) == 2\
                and all(isinstance(s, slice) for s in item):
            row_idx, col_idx, vals = [], [], []
            for i in range(*item[0].indices(self.shape[0])):
                for j in range(*item[1].indices(self.shape[1])):
                    row_idx.append(i)
                    col_idx.append(j)
                    vals.append(True)

            # mask self
            self._mask = Matrix(
                    (vals, 
                    (row_idx, col_idx)), 
                    shape=self.shape, 
                    dtype=bool
            ).mat

        # masking with boolean Matrix
        elif isinstance(item, Matrix):
            if item.dtype == bool: 
                self._mask = item.mat
            else:
                raise TypeError("Mask must be applied with boolean Matrix")

        else:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        return self

    # self[item] += assign
    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    def __setitem__(self, item, assign):
        assign()
        self._mask = no_mask
        self._repl = False

    # TODO double check that ~ copies instead of referencing
    def __invert__(self):
        return Matrix((~self.mat, self.shape, self.dtype))

    @property
    def nvals(self):
        return self.vec.nvals()

    def _combine(self, other):
        if isinstance(other, Matrix): 
            return Matrix((other.shape[0], self.shape[1]), ctype)
        elif isinstance(other, Vector): 
            return Vector((self.shape[1],), ctype)

class Vector(object):

    def __init__(self, v=None, shape=None, dtype=None):
        # require vector or shape and type
        if v is None and (shape is None or dtype is None):
            raise ValueError("Please provide vector or shape and dtype")

        self._mask = no_mask
        self._repl = False

        # copy constructor, no compilation needed
        if isinstance(v, tuple) and len(v) == 3:
            self.vec, self.shape, self.dtype = v
            # TODO temporary
            self.mat = self.vec
            return

        # get C++ module with declaration for Matrix class
        if dtype is not None: self.dtype = dtype 
        else: self.dtype = c.get_type(v)
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
            self.vec = module.init_sparse_vector(
                    self.shape[0], idx, data
            )

        else:
            self.shape = shape
            self.vec = module.init_sparse_vector(
                    self.shape[0], [], []
            )

        self.mat = self.vec

    def __add__(self, other):
        return ops.semiring.eWiseAdd(self, other)

    def __mul__(self, other):
        return ops.semiring.eWiseMult(self, other)

    def __matmul__(self, other):
        return ops.semiring.vxm(self, other)

    def __rmatmul__(self, other):
        return ops.semiring.mxv(other, self)

    def __iadd__(self, expr):
        print(self, expr)
        print(expr.args, expr.kwargs)
        if callable(expr):
            return lambda: expr(self, accum=ops.accumulator)
        elif isinstance(expr, Vector):
            return lambda: ops.Identity(expr, self, accum=ops.accumulator)
        else: raise TypeError("Evaluation was not deferred")

    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    def __getitem__(self, item):
        print("MASK", item)
        # if replace flag is set
        if isinstance(item, bool):
            self._repl = item
            return self

        # strip replace off end if it exists
        elif isinstance(item, tuple):
            if isinstance(item[-1], bool):
                *item, self._repl = item

        # no mask
        if item == slice(None, None, None):
            pass

        # masking with slice
        elif isinstance(item, slice):
            idx, vals = [], []
            for i in range(*item.indices(self.shape[0])):
                idx.append(i)
                vals.append(True)

            # mask self
            self._mask = Vector(
                    (vals, idx), 
                    shape=self.shape, 
                    dtype=bool
            ).mat

        # masking with boolean Matrix
        elif isinstance(item, Vector):
            if item.dtype == bool: self._mask = item.vec
            else: raise TypeError("Mask must be applied with boolean Vector")

        else:
            raise TypeError("Mask must be boolean Vector or slice with optional replace flag")

        return self

    def __setitem__(self, item, assign):
        if callable(assign):
            assign()
        print(item, assign)
        assign()
        self._mask = no_mask
        self._repl = False

    def __invert__(self):
        return Vector((~self.vec, self.shape, self.dtype))

    def __str__(self):
        return str(self.vec)

    @property
    def nvals(self):
        return self.vec.nvals()

    def __len__(self):
        return self.vec.size()

    def _combine(self, other):
        if isinstance(other, Matrix): 
            return Vector((other.shape[0],), ctype)
        elif isinstance(other, Vector): 
            return Vector((self.shape[1],), ctype)

    
