import numpy as np
from scipy import sparse
from GraphBLAS import compile_c as c
import GraphBLAS.operators as ops


class Matrix(object):

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
        return ops.semring.eWiseAdd(other, self)

    def __mul__(self, other):
        return ops.semiring.eWiseMult(self, other) 

    def __rmul__(self, other):
        return ops.semiring.eWiseMult(other, self) 

    def __matmul__(self, other):
        return ops.semiring.mxm(self, other)

    def __rmatmul__(self, other):
        return ops.semiring.mxm(other, self)

    def __iadd__(self, expr):
        raise Exception("use Matrix[:] notation to assign into matrix")

    # applies mask stored in item and returns self
    def __getitem__(self, item):

        mask = ops.no_mask
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
                    ).mat
                
                # index into matrix
                elif all(isinstance(i, int) for i in item):
                    if self.mat.hasElement(*item):
                        return self.mat.extractElement(*item)
                    else:
                        return ops.semiring.add_identity
                    item = None

                elif isinstance(item[1], bool):
                    item, replace_flag = item

        if isinstance(item, bool):
            replace_flag = item

        elif isinstance(item, Matrix):
            mask = item.mat

        elif item == slice(None, None, None):
            mask = ops.no_mask

        elif item is not None:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        return ops.Masked(self, mask, replace_flag)

    # NOTE if accum is expected, that gets handled in semiring or assign partial expression
    # self[item] = assign
    def __setitem__(self, item, assign):

        if isinstance(item, tuple)\
                and all(isinstance(i, int) for i in item):
            self.mat.setElement(item, assign)

        elif hasattr(assign, "eval"):
            self = assign.eval(self[item])

        # TODO copy constructor
        elif isinstance(assign, Matrix):
            self.mat = assign.mat

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


class Vector(object):

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

    # other = vector, mask, replace
    # other + self
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

    def __getitem__(self, item):

        mask = ops.no_mask
        replace_flag = False
    
        if isinstance(item, tuple):
            # self[0:N,True]
            if len(item) == 2 and isinstance(item[1], bool):
                *item, replace_flag = item

            if len(item) == 1:
                item = item[0]

        if item == slice(None, None, None):
            mask = ops.no_mask

        elif isinstance(item, slice):
            idx, vals = [], []
            for i in range(*item.indices(self.shape[0])):
                row_idx.append(i)
                col_idx.append(j)
                vals.append(True)

            # build mask from slice data
            mask = Vector(
                    (vals, idx), 
                    shape=self.shape, 
                    dtype=bool
            ).vec

        elif isinstance(item, bool):
            replace_flag = item
       
        elif isinstance(item, int):
            if self.vec.hasElement(item):
                return self.vec.extractElement(item)
            else:
                return ops.semiring.add_identity

        elif isinstance(item, Vector):
            mask = item.vec

        elif item is not None:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        return ops.Masked(self, mask, replace_flag)

    def __setitem__(self, item, assign):

        if isinstance(assign, int):
            self.vec.setElement(item, assign)

        elif hasattr(assign, "eval"):
            self = assign.eval(self[item], None)
        
        # TODO copy
        elif isinstance(assign, Vector):
            self.vec = assign.vec

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
