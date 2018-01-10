from abc import ABC, abstractmethod
import numpy as np
from functools import wraps

from . import c_functions as c_func
from . import containers

__all__ = [
        "ApplyExpression",
        "ReduceExpression",
        "BinaryExpression",
        "MaskedExpression"
]


###############################################################################
####  Expression definitions, provide partial evaluation and memoization  #####
###############################################################################

# decorator for expression.eval() to memoize results
def memoize(func):

    @wraps(func)
    def new_func(self, *args, **kwargs):

        C = func(self, *args, **kwargs)
        self._eval = C
        return C

    return new_func

def convert_mask(func):

    @wraps(func)
    def new_func(self, C=None, M=no_mask, accum=None, replace_flag=False):

        if type(C) == MaskedMatrix or type(C) == MaskedVector:
            return func(self, C.C, C.M, C.accum, C.replace_flag)
        
        else:
            return func(self, C, M, accum, replace_flag)

    return new_func

class _NoMask(object):

    def __init__(self):
        self.container = c_func.no_mask()
        self.dtype = None


no_mask = _NoMask()


# TODO
# indexing into MaskedContainer returns indexed expression
class MaskedMatrix(object):

    def __init__(self, C, M=no_mask, accum=None, replace_flag=False):

        self.C = C
        self.M = M
        self.accum = accum
        self.replace_flag = replace_flag

    # self.__setitem__(item, self.__getitem(item).__iadd__(value))
    def __iadd__(self, A):
        from .operators import get_accum

        self.accum = get_accum()

        if isinstance(A, _Expression):
            # TODO avoid double execution of assign
            return A.eval(self)

        else:
            return NotImplemented

    # masked can be converted to indexed
    def __getitem__(self, indices):
        if indices == slice(None, None, None) or len(indices) == 2:
            return IndexedMatrix(self, indices)

    def assign(self, A):

        # evaluate expression into self
        if isinstance(A, _Expression):
            return A.eval(self)

        # copy constructor
        elif isinstance(A, containers.Vector):
            return IndexedVector(A, slice(None, None, None)).eval(self) 

        else:
            return NotImplemented


class MaskedVector(object):

    def __init__(self, C, M=no_mask, accum=None, replace_flag=False):

        self.C = C
        self.M = M
        self.accum = accum
        self.replace_flag = replace_flag

    # self.__setitem__(item, self.__getitem(item).__iadd__(value))
    def __iadd__(self, A):
        from .operators import get_accum

        self.accum = get_accum()
        # TODO avoid double execution of assign
        return A.eval(self)

    # TODO masked can be converted to indexed
    def __getitem__(self, indices):
        if len(indices) == 1:
            return IndexedVector(self, indices)

    def assign(self, A):

        # evaluate expression into self
        if isinstance(A, _Expression):
            return A.eval(self)

        # copy constructor
        elif isinstance(A, containers.Vector):
            return IndexedVector(A, slice(None, None, None)).eval(self) 

        else:
            return NotImplemented


class _Expression(ABC):
    
    @abstractmethod
    def eval(self, C=None, M=no_mask, accum=None, replace_flag=False): pass

    @property
    def evaluated(self):
        if not hasattr(self, "_eval"):
            self._eval = self.eval()
        return self._eval

    @property
    def container(self):
        return self.evaluated.container

    @property
    def dtype(self):
        return self.evaluated.dtype

    @property
    def shape(self):
        return self.evaluated.shape

    def __neg__(self):
        return -self.evaluated

    def __invert__(self):
        return ~self.evaluated

    def __repr__(self):
        return str(self.evaluated)

    def __iter__(self):
        return iter(self.evaluated)

    def __getitem__(self, item):
        return self.evaluated[item]

    def __setitem__(self, item, value):
        self.evaluated[item] = value
        return self


class _BinaryExpression(_Expression):

    def __init__(self, op, A, B, C):

        self.op = op
        self.A = A
        self.B = B

        if C is not None:
            return self.eval(C)


class EWiseAddMatrix(_BinaryExpression):

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        assert(self.A.shape == self.B.shape)

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            C = containers.Matrix(
                    shape=self.A.shape,
                    dtype=c_type
            )

        # TODO convert indexed to masked
        elif isinstance(C, IndexedMatrix):
            pass

        elif not isinstance(C, containers.Matrix):
            return NotImplemented

        c_func.operator(
            function        = "eWiseAddMatrix",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            B               = self.B,
            C               = C,
            M               = M
        )

        return C


class EWiseAddVector(_BinaryExpression):

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        assert(self.A.shape == self.B.shape)

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            C = containers.Vector(
                    shape=self.A.shape,
                    dtype=c_type
            )

        # TODO convert indexed to masked
        elif isinstance(C, IndexedVector):
            pass

        elif not isinstance(C, containers.Vector):
            return NotImplemented

        c_func.operator(
            function        = "eWiseAddVector",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            B               = self.B,
            C               = C,
            M               = M
        )

        return C


class EWiseMultMatrix(_BinaryExpression):

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        assert(self.A.shape == self.B.shape)

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            C = containers.Matrix(
                    shape=self.A.shape,
                    dtype=c_type
            )

        # TODO convert indexed to masked
        elif isinstance(C, IndexedMatrix):
            pass

        elif not isinstance(C, containers.Matrix):
            return NotImplemented

        c_func.operator(
            function        = "eWiseMultMatrix",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            B               = self.B,
            C               = C,
            M               = M
        )

        return C


class EWiseMultVector(_BinaryExpression):

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        assert(self.A.shape == self.B.shape)

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            C = containers.Vector(
                    shape=self.A.shape,
                    dtype=c_type
            )

        # TODO convert indexed to masked
        elif isinstance(C, IndexedVector):
            pass

        elif not isinstance(C, containers.Vector):
            return NotImplemented

        c_func.operator(
            function        = "eWiseMultVector",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            B               = self.B,
            C               = C,
            M               = M
        )

        return C


class MXM(_BinaryExpression):

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            assert(self.A.shape[0] == self.B.shape[1])
            C = containers.Matrix(
                    shape=(self.B.shape[0], self.A.shape[1]),
                    dtype=c_type
            )

        # TODO convert indexed to masked
        elif isinstance(C, IndexedMatrix):
            pass

        elif not isinstance(C, containers.Matrix):
            return NotImplemented

        c_func.operator(
            function        = "mxm",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            B               = self.B,
            C               = C,
            M               = M
        )

        return C


class MXV(_BinaryExpression):

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            assert(self.A.shape[1] == self.B.shape[0])
            C = containers.Vector(
                    shape=(self.A.shape[0],),
                    dtype=c_type
            )

        # TODO convert indexed to masked?
        elif isinstance(C, IndexedVector):
            pass

        elif not isinstance(C, containers.Vector):
            return NotImplemented

        c_func.operator(
            function        = "mxv",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            B               = self.B,
            C               = C,
            M               = M
        )

        return C


class VXM(_BinaryExpression):

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            assert(self.A.shape[0] == self.B.shape[0])
            C = containers.Vector(
                    shape=(self.A.shape[0],),
                    dtype=c_type
            )

        # TODO convert indexed to masked
        elif isinstance(C, IndexedVector):
            pass

        elif not isinstance(C, containers.Vector):
            return NotImplemented

        c_func.operator(
            function        = "vxm",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            B               = self.B,
            C               = C,
            M               = M
        )

        return C


class ApplyMatrix(_Expression):

    def __init__(self, op, A, C=None):

        self.op = op
        self.A = A

        if C is not None:
            return self.eval(C)

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        if C is None:

            C = containers.Matrix(
                    shape=self.A.shape,
                    dtype=self.A.dtype
            )

        c_func.operator(
            function        = "applyMatrix",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            C               = C,
            M               = M
        )

        return C


class ApplyVector(_Expression):

    def __init__(self, op, A, C=None):

        self.op = op
        self.A = A

        if C is not None:
            return self.eval(C)

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        if C is None:

            C = containers.Vector(
                    shape=self.A.shape,
                    dtype=self.A.dtype
            )

        c_func.operator(
            function        = "applyVector",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            C               = C,
            M               = M
        )

        return C


class ReduceMatrix(_Expression):
    
    def __init__(self, reduce, A, C=None):

        self.reduce = reduce
        self.A = A

        if C is not None:
            return self.eval(C)

    @memoize
    @convert_mask
    def eval(self, C=None, M=no_mask, accum=None, replace_flag=False):

        function = "reduceMatrix"
        kwargs = {"A": self.A}

        # reduce to a scalar
        if C is None:
            kwargs["C"] = self.reduce.identity

        # reduce to a scalar with initial value
        elif isinstance(C, self.A.dtype):
            kwargs["C"] = C

        # reduce to a vector
        elif isinstance(C, containers.Vector):
            function = "reduceMatrixVector"
            kwargs["C"] = C
            kwargs["M"] = M
            kwargs["accum"] = accum
            kwargs["replace_flag"] = replace_flag

        else:
            raise TypeError("Can't reduce to {}".format(type(C)))

        result = c_func.operator(
                function        = function,
                operation       = self.reduce,
                **kwargs
        )

        return result


class ReduceVector(_Expression):
    
    def __init__(self, reduce, A, C=None):

        self.reduce = reduce
        self.A = A

        if C is not None:
            return self.eval(C)

    @memoize
    def eval(self, C=None, M=no_mask, accum=None, replace_flag=False):

        # reduce to a scalar
        if C is None:
            C = self.reduce.identity

        elif not isinstance(C, self.A.dtype):
            raise TypeError("Can't reduce to {}".format(type(C)))

        result = c_func.operator(
                function        = "reduceVector",
                operation       = self.reduce,
                accum           = accum,
                replace_flag    = replace_flag,
                A               = self.A,
                C               = C
        )

        return result


# acts as LHS if performing assign, RHS if performing extract
class IndexedMatrix(_Expression):

    def __init__(self, A, indices):

        self.A = A
        self.idx = dict()

        # convert 1D index to 2D
        if len(indices) == 1 and indices[0] == slice(None, None, None):
            indices = (*indices, *indices)

        for i, s, dim in zip(indices, A.shape, ("row", "col")):

            if isinstance(i, slice): 
                self.idx[dim + "_indices"] = range(*i.indices(s))

            elif isinstance(i, (list, np.ndarray)):
                self.idx[dim + "_indices"] = i

            elif isinstance(i, int):
                self.idx[dim + "_index"] = i

            else:
                raise TypeError("Mask indices can be slice, list or int")

    # accum expression will be evaluated by __setitem__ of underlying container
    # self.__setitem__(item, self.__getitem__(item).__iadd__(value))
    def __iadd__(self, A):
        from .operators import get_accum

        self.accum = get_accum()
        # TODO avoid double execution
        return A.eval(self)

    # TODO can be masked if dim(M) == dim(idx)
    def __getitem__(self, item):

        if item == slice(None, None, None):
            return self

        else:
            return NotImplemented

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        # construct container of correct shape and size to extract to
        if C is None:

            # extract row
            if "row_index" in self.idx:
                C = containers.Vector(
                        shape=(len(self.idx["col_indices"]),),
                        dtype = self.A.dtype
                )

            # extract column
            elif "col_index" in self.idx:
                C = containers.Vector(
                        shape=(len(self.idx["row_indices"]),),
                        dtype = self.A.dtype
                )

            # extract submatrix
            else:
                C = containers.Matrix(
                        shape=(
                            len(self.idx["row_indices"]), 
                            len(self.idx["col_indices"])
                        ),
                        dtype=self.A.dtype
                )

        if "row_index" in self.idx:
            function = "extractMatrixRow"

        elif "col_index" in self.idx:
            function = "extractMatrixCol"

        else:
            function = "extractSubmatrix"

        result = c_func.operator(
                function        = function,
                accum           = accum,
                replace_flag    = replace_flag,
                C               = C,
                M               = M,
                A               = self.A,
                **self.idx
        )

        return C
                    
    # LHS expression evaluation
    @memoize
    def assign(self, A):

        if isinstance(A, _Expression):
            A = A.eval()
   
        # constant assignment to indices
        if isinstance(A, self.A.dtype):

            function = "assignMatrixConst"

            if "row_index" in self.idx:
                self.idx["row_indices"] = [self.idx["row_index"]]
                del self.idx["row_index"]

            if "col_index" in self.idx:
                self.idx["col_indices"] = [self.idx["col_index"]]
                del self.idx["col_index"]

        # TODO assign from expression behavior not supported
        elif isinstance(A, containers.Matrix):
            function = "assignSubmatrix"

        elif isinstance(A, containers.Vector):
            
            if "row_index" in self.idx:
                function = "assignMatrixRow"

            elif "col_index" in self.idx:
                function = "assignMatrixCol"
                
        else:
            raise TypeError("Can't assign from non-matrix type")

        # TODO get params from somewhere
        c_func.operator(
                function        = function,
                replace_flag    = False,
                accum           = None,
                C               = self.A,
                M               = no_mask,
                A               = A,
                **self.idx
        )

        return self.A


class IndexedVector(_Expression):

    def __init__(self, A, index):

        self.A = A
        self.idx = dict()

        if isinstance(index, slice): 
            self.idx["indices"] = range(*index.indices(self.A.shape[0]))

        elif isinstance(index, (list, np.ndarray)):
            self.idx["indices"] = i

        else:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

    # TODO fix broken shit
    # accum expression will be evaluated by __setitem__ of underlying container
    # self.__setitem__(item, self.__getitem__(item).__iadd__(value))
    def __iadd__(self, A):
        from .operators import get_accum

        self.accum = get_accum()
        return A.eval(self)

    # TODO decide if self[i, j][M] is desirable interface
    def __getitem__(self, item):

        if item == slice(None, None, None):
            return self
        else:
            return NotImplemented

    @memoize
    @convert_mask
    def eval(self, C, M, accum, replace_flag):

        if C is None:

            C = containers.Vector(
                    shape=(len(self.idx["indices"]),),
                    dtype=self.A.dtype
            )

        result = c_func.operator(
                function        = "extractSubvector",
                accum           = accum,
                replace_flag    = replace_flag,
                C               = C,
                M               = M,
                A               = self.A,
                **self.idx
        )

        return C

    @memoize
    def assign(self, A):

        # TODO default replace flag?
        if isinstance(A, _Expression):
            A = A.eval()

        if isinstance(A, self.A.dtype):
            function = "assignVectorConst"

        elif isinstance(A, containers.Vector):
            function = "assignSubvector"

        else:
            raise TypeError("Can't assign from non-matrix type")

        c_func.operator(
                function        = function,
                replace_flag    = False,
                accum           = None,
                C               = self.A,
                M               = no_mask,
                A               = A,
                **self.idx
        )

        return self.A
