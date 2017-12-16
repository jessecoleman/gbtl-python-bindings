import numpy as np
from scipy import sparse
from .boundinnerclass import BoundInnerClass
from . import compile_c as c
from . import operators as ops


class Matrix(object):

    def __init__(self, m=None, shape=None, dtype=None):
        # require matrix or shape and type
        if m is None and (shape is None or dtype is None):
            raise ValueError("Please provide matrix or shape and dtype")

        # get C++ module with declaration for Matrix class
        if dtype is not None: self.dtype = dtype 
        else: self.dtype = c.get_type(m)
        module = c.get_container(self.dtype)

        # copy constructor, no compilation needed
        if isinstance(m, Matrix):
            self.mat = module.Matrix(m.mat)

        # construct from scipy sparse matrix
        elif (sparse.issparse(m)):
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

    def __eq__(self, other):
        return self.mat == other.mat

    def __neq__(self, other):
        return self.mat != other.mat

    def __add__(self, other):
        return ops.semiring.eWiseAdd(self, other)

    def __radd__(self, other):
        return ops.semring.eWiseAdd(other, self)

    def __mul__(self, other):
        return ops.semiring.eWiseMult(self, other) 

    def __rmul__(self, other):
        return ops.semiring.eWiseMult(other, self) 

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return ops.semiring.mxm(self, other)
        elif isinstance(other, Vector):
            return ops.semiring.mxv(self, other)

    def __rmatmul__(self, other):
        if isinstance(other, Matrix):
            return ops.semiring.mxm(other, self)
        elif isinstance(other, Vector):
            return ops.semiring.vxm(other, self)

    def __iadd__(self, expr):
        raise Exception("use Matrix[:] notation to assign into matrix")

    @BoundInnerClass
    class masked(object):

        def __init__(self, matrix, mask=None, replace_flag=False):

            self.container = matrix
            self.replace_flag = replace_flag

            if mask is None:
                self.mask = c.no_mask()
                self.mtype = (None, None)

            elif isinstance(mask, Matrix):
                self.mask = mask.mat
                if isinstance(mask, Complement):
                    self.mtype = (mask.dtype, "complement")
                elif isintance(mask, Transpose):
                    self.mtype = (mask.dtype, "transpose")
                else:
                    self.mtype = (mask.dtype, "mask")

            else:
                raise TypeError("Incorrect type for mask parameter")

        def __iadd__(self, other):
            if isinstance(other, ops.Expression):
                return other.eval(self, ops.accumulator)
            
            else:
                return ops.Identity(other).eval(self, ops.accumulator)

    # applies mask stored in item and returns self
    def __getitem__(self, item):

        # index into matrix
        if all(isinstance(i, int) for i in item):
            if self.mat.hasElement(*item):
                return self.mat.extractElement(*item)
            else:
                return ops.semiring.add_identity

        mask = None
        replace_flag = False
    
        if isinstance(item, tuple):
            # must be 2 slices and bool
            # self[0:N,0:M,True]
            if len(item) == 3 and isinstance(item[2], bool):
                    *item, replace_flag = item

            # 2D index or slices or mask and bool
            # self[1,1] or self[0:N,0:M] or self[M,True]
            if len(item) == 2:
                if all(isinstance(s, slice) for s in item):
                    s_i, s_j = item
                    item = None
                    row_idx, col_idx, vals = [], [], []
                    for i in range(*s_i.indices(self.shape[0])):
                        for j in range(*s_j.indices(self.shape[0])):
                            row_idx.append(i)
                            col_idx.append(j)
                            vals.append(True)
                
                    # build mask from slice data
                    mask = Matrix(
                            (vals,
                            (row_idx, col_idx)), 
                            shape=self.shape, 
                            dtype=bool
                    )
                
                elif isinstance(item[1], bool):
                    item, replace_flag = item

        if isinstance(item, bool):
            replace_flag = item

        elif isinstance(item, Matrix):
            mask = item

        elif item == slice(None, None, None):
            mask = None

        elif item is not None:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        return self.masked(mask, replace_flag)

    # NOTE if accum is expected, that gets handled in semiring or assign partial expression
    # self[item] = assign
    def __setitem__(self, item, assign):

        if (isinstance(item, tuple)
                and all(isinstance(i, int) for i in item)
                and len(item) == 2):
            self.mat.setElement(item, assign)

        elif hasattr(assign, "eval"):
            self = assign.eval(self[item])

        # TODO copy constructor
        elif isinstance(assign, Matrix):
            self = Matrix(assign)

        else: 
            raise TypeError("Matrix can be assigned to with integer indices or masks")

        return self

    def __str__(self):
        return str(self.mat)

    @property
    def nvals(self):
        return self.vec.nvals()

    @property
    def T(self):
        return MatrixTranspose(self)

    # TODO double check that ~ copies instead of referencing
    def __invert__(self):
        return MatrixComplement(self)

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
        self.mat = matrix.mat.T()
        self.shape = (matrix.shape[1], matrix.shape[0])
        self.dtype = matrix.dtype


class MatrixComplement(Matrix):

    def __init__(self, matrix):
        self.mat = ~matrix.mat
        self.shape = matrix.shape
        self.dtype = matrix.dtype


class Vector(object):

    def __init__(self, v=None, shape=None, dtype=None):
        # require vector or shape and type
        if v is None and (shape is None or dtype is None):
            raise ValueError("Please provide vector or shape and dtype")

        # get C++ module with declaration for Matrix class
        if dtype is not None: self.dtype = dtype 
        else: self.dtype = c.get_type(v)
        module = c.get_container(self.dtype)

        # copy constructor, no compilation needed
        if isinstance(v, Vector):
            self.vec = module.Vector(v.vec)

        elif isinstance(v, list):
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

    def __eq__(self, other):
        return self.vec == other.vec

    def __neq__(self, other):
        return self.vec != other.vec

    def __add__(self, other):
        return ops.semiring.eWiseAdd(self, other)

    def __radd__(self, other):
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
        raise Exception("use Vector[:] notation to assign into vector")

    @BoundInnerClass
    class masked(object):
        "object returned by call to __getitem__ on vector"

        def __init__(self, vector, mask=None, replace_flag=False):

            self.container = vector
            self.replace_flag = replace_flag

            if mask is None:
                self.mask = c.utilities().NoMask()
                self.mtype = (None, None)

            elif isinstance(mask, Vector):
                self.mask = mask.vec
                if isinstance(mask, VectorComplement):
                    self.mtype = (mask.dtype, "complement")
                else:
                    self.mtype = (mask.dtype, "mask")

            else:
                raise TypeError("Incorrect type for mask parameter")

        def __iadd__(self, other):

            if isinstance(other, ops.Expression):
                return other.eval(self, ops.accumulator)
            
            else:
                return ops.Identity(other).eval(self, ops.accumulator)

    def __getitem__(self, item):

        if isinstance(item, int) and not isinstance(item, bool):
            if self.vec.hasElement(item):
                return self.vec.extractElement(item)
            else:
                return ops.semiring.add_identity

        mask = None
        replace_flag = False
    
        if isinstance(item, tuple):
            # self[0:N,True]
            if len(item) == 2 and isinstance(item[1], bool):
                *item, replace_flag = item

            if len(item) == 1:
                item = item[0]

        if item == slice(None, None, None):
            mask = None

        elif isinstance(item, slice):
            idx, vals = [], []
            for i in range(*item.indices(self.shape[0])):
                idx.append(i)
                vals.append(True)

            # build mask from slice data
            mask = Vector(
                    (vals, idx), 
                    shape=self.shape, 
                    dtype=bool
            )

        elif isinstance(item, bool):
            replace_flag = item
       
        elif isinstance(item, Vector):
            mask = item

        elif item is not None:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        return self.masked(mask, replace_flag)

    def __setitem__(self, item, assign):

        # if vector[1] = int
        if isinstance(assign, int):
            self.vec.setElement(item, assign)

        # if vector[:] = expr
        elif isinstance(assign, ops.Expression):
            self = assign.eval(self[item])
        
        # call copy constructor
        elif isinstance(assign, Vector):
            self = Vector(assign)

        else:
            raise TypeError("Vectors can be assigned to with integer indices or masks")

        return self

    def __invert__(self):
        return VectorComplement(self)

    def __str__(self):
        return str(self.vec)

    @property
    def nvals(self):
        return self.vec.nvals()

    def __len__(self):
        return self.vec.size()

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
        self.vec = ~vector.vec
        self.shape = vector.shape
        self.dtype = vector.dtype

