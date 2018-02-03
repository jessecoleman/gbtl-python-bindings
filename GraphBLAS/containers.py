import numpy as np
from scipy import sparse
from . import c_functions as c


class Matrix(object):

    def __init__(self, m=None, shape=None, dtype=None, copy=True):

        # require matrix or shape and type
        if m is None and (shape is None or dtype is None):
            raise ValueError("Please provide matrix or shape and dtype")

        # copy constructor
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

    def __iadd__(self, const):
        if isinstance(const, self.dtype):
            self[:] += const
            return self

        else:
            raise TypeError("{} must be of type {}".format(const, self.dtype))

    #def __imul__(self, other):
    #    if type(other) in ops.

    def __add__(self, other):
        from .operators import eWiseAdd
        return eWiseAdd(None, self, other)

    def __radd__(self, other):
        from .operators import eWiseAdd
        return eWiseAdd(None, other, self)

    def __mul__(self, other):
        from .operators import eWiseMult
        return eWiseMult(None, self, other)

    def __rmul__(self, other):
        from .operators import eWiseMult
        return eWiseMult(None, other, self)

    def __matmul__(self, other):
        from .operators import mxm, mxv
        if isinstance(other, Matrix):
            return mxm(None, self, other)
        elif isinstance(other, Vector):
            return mxv(None, self, other)

    def __rmatmul__(self, other):
        from .operators import mxm, vxm
        if isinstance(other, Matrix):
            return mxm(None, other, self)
        elif isinstance(other, Vector):
            return vxm(None, other, self)

    @property
    def T(self):
        # TODO
        return MatrixTranspose(self)
        from .expressions import Transpose
        return Transpose(self)

    def __invert__(self):
        return MatrixComplement(self)

    def __neg__(self):
        from .operators import apply, AdditiveInverse
        return apply(AdditiveInverse, self)

    def __getitem__(self, item):
        from .expressions import IndexedMatrix, MaskedMatrix, AllIndices, NoMask

        if item is None:
            return MaskedMatrix(self, NoMask())

        elif isinstance(item, (Matrix, NoMask)):
            return MaskedMatrix(self, item)

        elif item == slice(None):
            return IndexedMatrix(self, (item,) * 2)

        elif type(item) == tuple and len(item) == 2:

            # assign/extract expression
            if all(isinstance(i, (int, slice, list, np.ndarray)) for i in item):
                return IndexedMatrix(self, item)

            elif all(isinstance(i, int) for i in item):
                return self.container.extractElement(*item)
    
            else:
                raise TypeError("Mask must be a boolean matrix or [:] slice")

    def __setitem__(self, item, value):

        if type(item) == tuple and len(item) == 2 and all(isinstance(i, int) for i in item):
            self.container.setElement(*item, value)

        # TODO handle other improper input
        else:
            self[item].assign(value)
            return self

    def __iter__(self):
        i, j, v = self.container.extractTuples()
        return iter(zip(i, j, v))


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
            if isinstance(shape, int):
                self.shape = (shape,)
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

    def __iadd__(self, const):

        if isinstance(const, self.dtype):
            self[:] += const
            return self

        else:
            raise TypeError("{} must be of type {}".format(const, self.dtype))


    def __add__(self, other):
        from .operators import eWiseAdd
        return eWiseAdd(None, self, other)

    def __radd__(self, other):
        from .operators import eWiseAdd
        return eWiseAdd(None, other, self)

    def __mul__(self, other):
        from .operators import eWiseMult
        return eWiseMult(None, self, other)

    def __rmul__(self, other):
        from .operators import eWiseMult
        return eWiseMult(None, other, self)

    def __matmul__(self, other):
        from .operators import vxm
        return vxm(None, self, other)

    def __rmatmul__(self, other):
        from .operators import mxv
        return mxv(None, other, self)

    def __invert__(self):
        return VectorComplement(self)

    def __neg__(self):
        from .operators import apply, AdditiveInverse
        return apply(AdditiveInverse, self)

    def __getitem__(self, item):
        from .expressions import MaskedVector, IndexedVector, NoMask, AllIndices

        if isinstance(item, int):
            return self.container.extractElement(item)

        if item is None:
            return MaskedVector(self, NoMask())

        if isinstance(item, (Vector, NoMask)):
            return MaskedVector(self, item)

        elif item == slice(None) or isinstance(item, (slice, list, np.ndarray, AllIndices)):
            return IndexedVector(self, item)

        else:
            raise TypeError("Mask must be a boolean vector or [:] slice")

    def __setitem__(self, item, value):

        if isinstance(item, int):
            self.container.setElement(item, value)

        # TODO check for invalid input
        else:
            self[item].assign(value)
            return self

    def __iter__(self):

        i, v = self.container.extractTuples()
        return iter(zip(i, v))


class VectorComplement(Vector):

    def __init__(self, vector):

        self.source = vector
        self.container = ~vector.container
        self.shape = vector.shape
        self.dtype = vector.dtype

    def __invert(self):

        return self.source

