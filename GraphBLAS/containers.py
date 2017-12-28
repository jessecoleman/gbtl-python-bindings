from attr import attrs, attrib
import numpy as np
from scipy import sparse
from .boundinnerclass import BoundInnerClass
from . import c_functions as c
from . import operators as ops


class Matrix(object):

    def __init__(self, m=None, shape=None, dtype=None, copy=True):

        # require matrix or shape and type
        if m is None and (shape is None or dtype is None):
            raise ValueError("Please provide matrix or shape and dtype")

        # copy constructor, no compilation needed
        if isinstance(m, Matrix):

            self.dtype = m.dtype
            self.shape = m.shape
            module = c.container(self.dtype)

            if copy == True:
                self.container = module.Matrix(m.container)
            else:
                self.container = m.container
            return

        else:
            # get C++ module with declaration for Matrix class
            if dtype is not None: 
                self.dtype = dtype
            else: 
                self.dtype = c.get_type(m)
            module = c.container(self.dtype)

        # construct from scipy sparse matrix
        if (sparse.issparse(m)):
            d = m.tocoo(copy=False)
            self.shape = d.shape
            self.container = module.init_sparse_matrix(
                    *self.shape, d.row, d.col, d.data
            )

        # construct from tuple of arrays (data, (rows, cols))
        elif isinstance(m, tuple) and len(m) == 2:
            data, idx = m
            row_idx, col_idx = idx
            if shape is not None: 
                self.shape = shape
            else: 
                self.shape = (max(row_idx) + 1, max(col_idx) + 1)

            self.container = module.init_sparse_matrix(
                    *self.shape, row_idx, col_idx, data
            )

        # construct empty matrix from shape and dtype
        else:
            self.shape = shape
            self.container = module.init_sparse_matrix(
                    *self.shape, [], [], []
            )

    def __repr__(self):
        #return "<{}x{} {} object with {} values>".format(*self.shape, type(self).__name__, self.nvals)
        return str(self.container)

    def __eq__(self, other):
        if isinstance(other, Matrix):
            return self.container == other.container
        else:
            return False

    def __neq__(self, other):
        if isinstance(other, Matrix):
            return self.container != other.container
        else:
            return False

    @property
    def nvals(self):
        return self.container.nvals()

    def __iadd__(self, expr):
        raise Exception("use {}[:] notation to assign into container".format(type(self)))

    def __add__(self, other):
        return ops.eWiseAdd(None, self, other)

    def __radd__(self, other):
        return ops.eWiseAdd(None, other, self)

    def __mul__(self, other):
        return ops.eWiseMult(None, self, other)

    def __rmul__(self, other):
        return ops.eWiseMult(None, other, self)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return ops.mxm(None, self, other)
        elif isinstance(other, Vector):
            return ops.mxv(None, self, other)

    def __rmatmul__(self, other):
        if isinstance(other, Matrix):
            return ops.mxm(None, other, self)
        elif isinstance(other, Vector):
            return ops.vxm(None, other, self)

    @property
    def T(self):
        return MatrixTranspose(self)

    def __invert__(self):
        return MatrixComplement(self)

    def __neg__(self):
        return ops.AdditiveInverse(self)

    def __getitem__(self, item):

        if type(item) is not tuple:
            item = (item,)

        return ops.MaskedExpression(self, *item)

   # NOTE if accum is expected, that gets handled in semiring or assign partial expression
    def __setitem__(self, item, value):

        if isinstance(value, ops._Expression):
            value.eval(self[item])

        else:
            self[item].assign(value)

        return self

    def __iter__(self):
        i, j, v = self.container.extractTuples()
        return zip(i, j, v).__iter__()

    # returns a new container with the correct output dimensions
    def _out_container(self, other=None):

        # output from apply
        if other is None:
            return Matrix(
                    shape=self.shape,
                    dtype=self.dtype
            )

        # output from semiring operation
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


class MatrixTranspose(Matrix):

    def __init__(self, matrix):
        self.source = matrix
        self.container = matrix.container.T()
        self.shape = (matrix.shape[1], matrix.shape[0])
        self.dtype = matrix.dtype

    @property
    def T(self):
        return self.source


class MatrixComplement(Matrix):

    def __init__(self, matrix):
        self.source = matrix
        self.container = ~matrix.container
        self.shape = matrix.shape
        self.dtype = matrix.dtype

    def __invert(self):
        return self.source


class Vector(object):

    def __init__(self, v=None, shape=None, dtype=None, copy=True):
        # require vector or shape and type
        if v is None and (shape is None or dtype is None):
            raise ValueError("Please provide vector or shape and dtype")

        # copy constructor, no compilation needed
        if isinstance(v, Vector):

            self.dtype = v.dtype
            self.shape = v.shape
            module = c.container(self.dtype)

            if copy == True:
                self.container = module.Vector(v.container)
            else:
                self.container = v.container

            return

        else:
            # get C++ module with declaration for Matrix class
            if dtype is not None: 
                self.dtype = dtype
            else: 
                self.dtype = c.get_type(v)

            module = c.container(self.dtype)


        if isinstance(v, list):
            self.shape = (len(v),)
            self.container = module.Vector(v)

        elif isinstance(v, np.ndarray):
            self.shape = v.shape
            self.container = module.Vector(self)

        # construct from tuple of arrays (data, vals)
        elif isinstance(v, tuple) and len(v) == 2:
            data, idx = v
            if shape is not None: self.shape = shape
            else: self.shape = (max(idx) + 1,)
            self.container = module.init_sparse_vector(
                    *self.shape, idx, data
            )

        else:
            self.shape = shape
            self.container = module.init_sparse_vector(
                    *self.shape, [], []
            )

    def __repr__(self):
        return str(self.container)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.container == other.container
        else:
            return False

    def __neq__(self, other):
        if isinstance(other, Vector):
            return self.container != other.container
        else:
            return False

    @property
    def nvals(self):
        return self.container.nvals()

    def __len__(self):
        return self.container.size()

    def __iadd__(self, expr):
        raise Exception("use {}[:] notation to assign into container".format(type(self)))

    def __add__(self, other):
        return ops.eWiseAdd(None, self, other)

    def __radd__(self, other):
        return ops.eWiseAdd(None, other, self)

    def __mul__(self, other):
        return ops.eWiseMult(None, self, other)

    def __rmul__(self, other):
        return ops.eWiseMult(None, other, self)

    def __matmul__(self, other):
        return ops.vxm(None, self, other)

    def __rmatmul__(self, other):
        return ops.mxv(None, other, self)

    def __invert__(self):
        return VectorComplement(self)

    def __neg__(self):
        return ops.AdditiveInverse(self)

    def __getitem__(self, item):

        if type(item) is not tuple:
            item = (item,)

        return ops.MaskedExpression(self, *item)

    def __setitem__(self, item, value):

        if isinstance(value, ops._Expression):
            value.eval(self[item])

        else:
            self[item].assign(value)

        return self

    def __iter__(self):
        i, v = self.container.extractTuples()
        return zip(i, v).__iter__()

    # returns a new container with the correct output dimensions
    def _out_container(self, other=None):

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


class VectorComplement(Vector):

    def __init__(self, vector):
        self.source = vector
        self.container = ~vector.container
        self.shape = vector.shape
        self.dtype = vector.dtype

    def __invert(self):
        return self.source

