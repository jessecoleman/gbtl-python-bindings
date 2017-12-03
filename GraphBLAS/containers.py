import numpy as np
from scipy import sparse
from GraphBLAS import compile_c as c
import GraphBLAS.operators as ops


class Matrix(object):

    no_mask = c.get_container(bool).NoMask()

    def __init__(self, m=None, shape=None, dtype=None):
        # require matrix or shape and type
        if m is None and (shape is None or dtype is None):
            raise ValueError("Please provide matrix or shape and dtype")

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

    def __radd__(self, other):
        if isinstance(other, tuple):
            return ops.Identity(self, other, ops.accumulator)

        else:
            return ops.semring.eWiseAdd(other, self)

    def __mul__(self, other):
        return ops.semiring.eWiseMult(self, other) 

    def __rmul__(self, other):
        return ops.semiring.eWiseMult(other, self) 

    def __matmul__(self, other):
        return ops.semiring.mxm(self, other)

    def __rmatmul__(self, other):
        return ops.semiring.mxm(other, self)

    # TODO check if iadd is necessary
    # create callable object from self and current accumulator
    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
#    def __iadd__(self, expr):
#        print(self, expr)
#
#        # if already callable
#        if callable(expr):
#            return lambda: expr(
#                    self, 
#                    accum=ops.accumulator
#            )
#
#        # else convert to apply with Identity
#        elif isinstance(expr, Matrix):
#            # lazily evaluates Apply object
#            return lambda: ops.Identity(
#                    expr, 
#                    self, 
#                    accum=ops.accumulator
#            )
#
#        else: 
#            raise TypeError("Evaluation was not deferred")

    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    # applies mask stored in item and returns self
    def __getitem__(self, item):
    
        if isinstance(item, tuple)\
                and all(isinstance(i, int) for i in item):
            if self.hasElement(*item):
                return self.extractElement(*item)
            else:
                return ops.semiring.add_identity

        mask = Matrix.no_mask
        replace_flag = False

        # no mask
        if item == slice(None, None, None):
            pass

        # if replace flag is set
        if isinstance(item, bool):
            replace_flag = item

        # strip replace off end if it exists
        elif isinstance(item, tuple):
            # if single element access
            if all(isinstance(i, int) for i in item):
                if self.mat.hasElement(*item):
                    return self.mat.extractElement(*item)
                else: 
                    return semiring._ops.add_identity

            if isinstance(item[-1], bool):
                *item, replace_flag = item
                item = tuple(item)

        # masking with slice
        if isinstance(item, tuple) and len(item) == 2\
                and all(isinstance(s, slice) for s in item):
            row_idx, col_idx, vals = [], [], []
            for i in range(*item[0].indices(self.shape[0])):
                for j in range(*item[1].indices(self.shape[1])):
                    row_idx.append(i)
                    col_idx.append(j)
                    vals.append(True)

            # mask self
            mask = Matrix(
                    (vals, 
                    (row_idx, col_idx)), 
                    shape=self.shape, 
                    dtype=bool
            ).mat

        # masking with boolean Matrix
        elif isinstance(item, Matrix):
            mask = item.mat

        else:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        return self, mask, replace_flag

    # self[item] += assign
    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    def __setitem__(self, item, assign):

        if isinstance(item, tuple)\
                and all(isinstance(i) for i in item):
            self.mat.setElement(item, assign)

        elif callable(assign):
            self = assign(self[item])

        elif isinstance(assign, Matrix):
            self = assign

        else: 
            raise TypeError("Matrix can be assigned to with integer indices or masks")

        return self

    # TODO double check that ~ copies instead of referencing
    def __invert__(self):
        return Matrix((~self.mat, self.shape, self.dtype))

    @property
    def nvals(self):
        return self.vec.nvals()

    @property
    def T(self):
        return self.mat.T()

    # returns a new container with the correct output dimensions
    def _get_out_shape(self, other=None):
        if other is None:
            return Matrix(
                    shape=self.shape,
                    dtype=self.dtype
            )
            
        ctype = c.upcast(self.dtype, other.dtype)
        if isinstance(other, Matrix): 
            return Matrix(
                    shape=(other.shape[0], self.shape[1]), 
                    dtype=ctype
            )
        elif isinstance(other, Vector): 
            return Vector(
                    shape=(self.shape[1],), 
                    dtype=ctype
            )


class Vector(object):

    no_mask = c.get_container(bool).NoMask()

    def __init__(self, v=None, shape=None, dtype=None):
        # require vector or shape and type
        if v is None and (shape is None or dtype is None):
            raise ValueError("Please provide vector or shape and dtype")

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

    # if trying to add masked object with self
    def __radd__(self, other):
        # if assignment
        if isinstance(other, tuple):
            return ops.Identity(self, other, ops.accumulator)

        return ops.semiring.eWiseAdd(other, self)

    def __mul__(self, other):
        return ops.semiring.eWiseMult(self, other)

    def __rmul__(self, other):
        return ops.semiring.eWiseMult(other, self)

    def __matmul__(self, other):
        return ops.semiring.vxm(self, other)

    def __rmatmul__(self, other):
        return ops.semiring.mxv(other, self)

    def __iadd__(self, expr):

        # if already callable
        if callable(expr):
            return expr(self, accum=ops.accumulator)

        # else convert to apply with Identity
        elif isinstance(expr, Vector):
            return ops.Identity(
                    expr, 
                    self, 
                    accum=ops.accumulator
            )

        else: 
            raise TypeError("Evaluation was not deferred")

    # self.__setitem__(self.__getitem__(item).__iadd__(assign))
    def __getitem__(self, item):

        if isinstance(item, int):
            if self.hasElement(item):
                return self.extractElement(item)
            else:
                return semiring.add_identity

        mask = Vector.no_mask
        replace_flag = False

        # if replace flag is set
        if isinstance(item, bool):
            replace_flag = item

        # strip replace off end if it exists
        elif isinstance(item, tuple):
            if isinstance(item[-1], bool):
                item, replace_flag = item

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
            mask = Vector(
                    (vals, idx), 
                    shape=self.shape, 
                    dtype=bool
            ).vec

        # masking with boolean Matrix
        elif isinstance(item, Vector):
            mask = item.vec

        else:
            raise TypeError("Mask must have type Vector or slice")

        return self, mask, replace_flag

    def __setitem__(self, item, assign):

        if isinstance(item, int):
            self.vec.setElement(item, assign)

        elif callable(assign):
            self = assign(self[item])
        
        elif isinstance(assign, Vector):
            self = assign

        else:
            raise TypeError("Vectors can be assigned to with integer indices or masks")
        
        return self

    def __invert__(self):
        return Vector((~self.vec, self.shape, self.dtype))

    def __str__(self):
        return str(self.vec)

    @property
    def nvals(self):
        return self.vec.nvals()

    @property
    def T(self):
        return self.vec.T()

    def __len__(self):
        return self.vec.size()

    # returns a new container with the correct output dimensions
    def _get_out_shape(self, other=None):
        if other is None:
            return Vector(
                    shape=self.shape,
                    dtype=self.dtype
            )

        ctype = c.upcast(self.dtype, other.dtype)
        if isinstance(other, Matrix):
            return Vector(
                    shape=(other.shape[0],), 
                    dtype=ctype
            )
        elif isinstance(other, Vector): 
            return Vector(
                    shape=self.shape, 
                    dtype=ctype
            )
