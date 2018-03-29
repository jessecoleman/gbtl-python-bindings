from abc import ABC, abstractmethod
import numpy as np
from functools import wraps

from . import c_functions as c_func
from . import containers

__all__ = [
        "ApplyExpression",
        "ReduceExpression",
        "MaskedExpression"
]


###############################################################################
####  Expression definitions, provide partial evaluation and memoization  #####
###############################################################################

# decorator for expression.eval() to memoize results
def memoize(eval):

    @wraps(eval)
    def memoized_eval(expr, *args, **kwargs):

        expr._eval = eval(expr, *args, **kwargs)
        return expr._eval

    return memoized_eval


def capture_operators(eval):

    @wraps(eval)
    def captured_eval(expr, C=None, M=None, accum=False, replace_flag=False):
        from .operators import get_accum

        if M is None:
            M = no_mask

        if accum:
            accum = get_accum()

        if isinstance(C, MaskedMatrix):
            return eval(expr, C.C, C.M, accum, replace_flag)

        elif isinstance(C, MaskedVector):
            return eval(expr, C.w, C.m, accum, replace_flag)

        else:
            return eval(expr, C, M, accum, replace_flag)

    return captured_eval


class NoMask(object):

    def __init__(self):
        self.container = c_func.no_mask()
        self.dtype = None


class AllIndices(object):

    def __init__(self):
        self.container = c_func.all_indices()


no_mask = NoMask()
all_indices = AllIndices()


# TODO
# indexing into MaskedContainer returns indexed expression
class MaskedMatrix(object):

    def __init__(self, C, M, replace_flag=False):
        self.dtype = C.dtype
        self.C = C
        self.M = M
        self.replace_flag = replace_flag

    # self.__setitem__(item, self.__getitem(item).__iadd__(value))
    def __iadd__(self, A):
        # TODO avoid double execution
        return self.assign(A, accum=True)

    # TODO debug
    def __getitem__(self, indices):

        if indices == slice(None):
            return IndexedMatrix(self, (indices,) * 2)

        elif len(indices) == 2:
            return IndexedMatrix(self, indices)

        else:
            raise TypeError("Index must be length 2")

    # masked can be converted to indexed
    def __setitem__(self, indices, value):

        return self[indices].assign(value)

    def assign(self, A, accum=False):
        from .operators import apply, Identity

        # evaluate expression into self
        if isinstance(A, _Expression):
            return A.eval(
                    self, 
                    accum=accum, 
                    replace_flag=self.replace_flag
            )

        # copy constructor with cast
        elif isinstance(A, containers.Vector):
            return apply(Identity, A).eval(
                    self, 
                    accum=accum, 
                    replace_flag=self.replace_flag
            )

        else:
            return NotImplemented


class MaskedVector(object):

    def __init__(self, w, m, replace_flag=False):
        self.dtype = w.dtype
        self.w = w
        self.m = m
        self.accum = False
        from .operators import get_replace
        self.replace_flag = get_replace() #replace_flag

    # self.__setitem__(item, self.__getitem(item).__iadd__(value))
    def __iadd__(self, u):
        return self.assign(u, accum=True)

    def __getitem__(self, indices):
        return IndexedVector(self, indices)

    def __setitem__(self, indices, value):
        return self[indices].assign(value)

    def assign(self, A, accum=False):
        from .operators import apply, Identity

        # evaluate expression into self
        if isinstance(A, _Expression):
            return A.eval(
                    self, 
                    accum=accum, 
                    replace_flag=self.replace_flag
            )

        elif isinstance(A, containers.Vector):
            return apply(Identity, A).eval(
                    self, 
                    accum=accum, 
                    replace_flag=self.replace_flag
            )

        # TODO ???
        else:
            return NotImplemented


class _Expression(ABC):

    @abstractmethod
    def eval(self, C=None, M=no_mask, accum=None, replace_flag=False): 
        pass

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

    def __mul__(self, other):
        return self.evaluated * other


class EWiseAddMatrix(_Expression):

    def __init__(self, op, A, B, C):

        self.op = op
        self.A = A
        self.B = B

        if C is not None:
            return self.eval(C)

    @memoize
    @capture_operators
    def eval(self, C, M, accum, replace_flag):

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            C = containers.Matrix(
                    shape=self.A.shape,
                    dtype=c_type
            )

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


class EWiseAddVector(_Expression):

    def __init__(self, op, u, v, w):

        self.op = op
        self.u = u
        self.v = v

        if w is not None:
            return self.eval(w)

    @memoize
    @capture_operators
    def eval(self, w, m, accum, replace_flag):

        # construct appropriate container
        if w is None:

            w_type = c_func.upcast(self.u.dtype, self.v.dtype)

            w = containers.Vector(
                    shape=self.u.shape,
                    dtype=w_type
            )

        elif not isinstance(w, containers.Vector):
            return NotImplemented

        c_func.operator(
            function        = "eWiseAddVector",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            u               = self.u,
            v               = self.v,
            w               = w,
            m               = m
        )

        return w


class EWiseMultMatrix(_Expression):

    def __init__(self, op, A, B, C):

        self.op = op
        self.A = A
        self.B = B

        if C is not None:
            return self.eval(C)

    @memoize
    @capture_operators
    def eval(self, C, M, accum, replace_flag):

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            C = containers.Matrix(
                    shape=self.A.shape,
                    dtype=c_type
            )

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


class EWiseMultVector(_Expression):

    def __init__(self, op, u, v, w):

        self.op = op
        self.u = u
        self.v = v

        if w is not None:
            return self.eval(w)

    @memoize
    @capture_operators
    def eval(self, w, m, accum, replace_flag):

        # construct appropriate container
        if w is None:

            w_type = c_func.upcast(self.u.dtype, self.v.dtype)

            w = containers.Vector(
                    shape=self.u.shape,
                    dtype=w_type
            )

        elif not isinstance(w, containers.Vector):
            return NotImplemented

        result = c_func.operator(
            function        = "eWiseMultVector",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            u               = self.u,
            v               = self.v,
            w               = w,
            m               = m
        )

        return w


class MXM(_Expression):

    def __init__(self, op, A, B, C):

        self.op = op
        self.A = A
        self.B = B

        if C is not None:
            return self.eval(C)

    @memoize
    @capture_operators
    def eval(self, C, M, accum, replace_flag):

        # construct appropriate container
        if C is None:

            c_type = c_func.upcast(self.A.dtype, self.B.dtype)

            C = containers.Matrix(
                    shape=(self.B.shape[0], self.A.shape[1]),
                    dtype=c_type
            )

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


class MXV(_Expression):

    def __init__(self, op, A, v, w):

        self.op = op
        self.A = A
        self.v = v

        if w is not None:
            return self.eval(w)

    @memoize
    @capture_operators
    def eval(self, w, m, accum, replace_flag):

        # construct appropriate container
        if w is None:

            w_type = c_func.upcast(self.A.dtype, self.v.dtype)

            w = containers.Vector(
                    shape=(self.A.shape[0],),
                    dtype=w_type
            )

        elif not isinstance(w, containers.Vector):
            return NotImplemented

        c_func.operator(
            function        = "mxv",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            A               = self.A,
            v               = self.v,
            w               = w,
            m               = m
        )

        return w


class VXM(_Expression):

    def __init__(self, op, u, B, w):

        self.op = op
        self.u = u
        self.B = B

        if w is not None:
            return self.eval(w)

    @memoize
    @capture_operators
    def eval(self, w, m, accum, replace_flag):

        # construct appropriate container
        if w is None:

            w_type = c_func.upcast(self.u.dtype, self.B.dtype)

            w = containers.Vector(
                    shape=(self.u.shape[0],),
                    dtype=w_type
            )

        elif not isinstance(w, containers.Vector):
            return NotImplemented

        c_func.operator(
            function        = "vxm",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            u               = self.u,
            B               = self.B,
            w               = w,
            m               = m
        )

        return w


class ApplyMatrix(_Expression):

    def __init__(self, op, A, C=None):

        self.op = op
        self.A = A

        if C is not None:
            return self.eval(C)

    @memoize
    @capture_operators
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

    def __init__(self, op, u, w=None):

        self.op = op
        self.u = u

        if w is not None:
            return self.eval(w)

    @memoize
    @capture_operators
    def eval(self, w, m, accum, replace_flag):

        if w is None:

            w = containers.Vector(
                    shape=self.u.shape,
                    dtype=self.u.dtype
            )

        c_func.operator(
            function        = "applyVector",
            operation       = self.op,
            accum           = accum,
            replace_flag    = replace_flag,
            u               = self.u,
            w               = w,
            m               = m
        )

        return w


class ReduceMatrix(_Expression):

    def __init__(self, op, A, s=None):

        self.op = op
        self.A = A

        if s is not None:
            return self.eval(s)

    @memoize
    @capture_operators
    def eval(self, x, m, accum, replace_flag):

        kwargs = {"A": self.A}

        # reduce to a scalar
        if x is None:
            function = "reduceMatrix"
            kwargs["s"] = self.reduce.identity

        # reduce to a scalar with initial value
        elif (list(c_func.types.keys()).index(self.A.dtype) <
                list(c_func.types.keys()).index(type(x))):
            function = "reduceMatrix"
            kwargs["s"] = x

        # reduce to a vector
        elif isinstance(x, containers.Vector):
            function = "reduceMatrixVector"
            kwargs["w"] = x
            kwargs["m"] = m
            kwargs["accum"] = accum
            kwargs["replace_flag"] = replace_flag

        else:
            raise TypeError("Can't reduce to {}".format(type(x)))

        result = c_func.operator(
                function        = function,
                operation       = self.op,
                **kwargs
        )

        return result


class ReduceVector(_Expression):

    def __init__(self, op, u, s=None):

        self.op = op
        self.u = u

        if s is not None:
            return self.eval(s)

    @memoize
    @capture_operators
    def eval(self, s, m, accum, replace_flag):

        # reduce to a scalar
        if s is None:
            s = self.reduce.identity

        elif (list(c_func.types.keys()).index(self.u.dtype) >
                list(c_func.types.keys()).index(type(s))):
            raise TypeError("Can't reduce to {}".format(type(s)))

        result = c_func.operator(
                function        = "reduceVector",
                operation       = self.op,
                accum           = accum,
                u               = self.u,
                s               = s
        )

        return result

    # TODO
    def __div__(self, other):
        return self.evaluated / other


# acts as LHS if performing assign, RHS if performing extract
class IndexedMatrix(_Expression):

    def __init__(self, X, indices):

        # forces LHS evaluation
        if isinstance(X, MaskedMatrix):
            self.LHS = True
            self.C = X.C
            self.M = X.M
            self.replace_flag = X.replace_flag

        # can be used as LHS or RHS
        else:
            self.LHS = False
            self.A = X
            self.C = X
            self.M = no_mask
            # TODO figure out this bs
            self.replace_flag = False

        self.idx = dict()

        for i, s, dim in zip(indices, self.C.shape, ("row", "col")):

            if i == slice(None):
                self.idx[dim + "_indices"] = all_indices

            elif isinstance(i, slice):
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
        return self.assign(A, accum=True)

    # extract operation
    @memoize
    @capture_operators
    def eval(self, C, M, accum, replace_flag):

        if self.LHS:
            raise TypeError("Mask can only be applied to LHS")

        # construct container of correct shape and size to extract to
        if C is None:

            # extract row
            if "row_index" in self.idx:
                C = containers.Vector(
                        shape=(len(self.idx["col_indices"]),),
                        dtype=self.A.dtype
                )

            # extract column
            elif "col_index" in self.idx:
                C = containers.Vector(
                        shape=(len(self.idx["row_indices"]),),
                        dtype=self.A.dtype
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

        if "row_index" in self.idx and isinstance(C, containers.Vector):
            function = "extractMatrixRow"

        elif "col_index" in self.idx and isinstance(C, containers.Vector):
            function = "extractMatrixCol"

        elif isinstance(C, containers.Matrix):
            function = "extractSubmatrix"

        else:
            raise TypeError("Incorrect parameter types")

        c_func.operator(
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
    def assign(self, A, accum=False):

        if isinstance(A, _Expression):
            A = A.eval()

        kwargs = {}

        # constant assignment to indices
        if isinstance(A, self.C.dtype):

            function = "assignMatrixConst"
            kwargs["val"] = A

            if "row_index" in self.idx:
                self.idx["row_indices"] = [self.idx["row_index"]]
                del self.idx["row_index"]

            if "col_index" in self.idx:
                self.idx["col_indices"] = [self.idx["col_index"]]
                del self.idx["col_index"]

        # TODO assign from expression behavior not supported
        elif isinstance(A, containers.Matrix):
            function = "assignSubmatrix"
            kwargs["A"] = A

        elif isinstance(A, containers.Vector):

            kwargs["u"] = A
            if "row_index" in self.idx:
                function = "assignMatrixRow"

            elif "col_index" in self.idx:
                function = "assignMatrixCol"

        else:
            raise TypeError("Can't assign from non-matrix type")

        # TODO get params from somewhere
        c_func.operator(
                function        = function,
                replace_flag    = self.replace_flag,
                accum           = accum,
                C               = self.C,
                M               = self.M,
                **self.idx,
                **kwargs
        )

        return self.C


class IndexedVector(_Expression):

    def __init__(self, x, index):

        # forces LHS evaluation
        if isinstance(x, MaskedVector):
            self.LHS = True
            self.w = x.w
            self.m = x.m
            self.replace_flag = x.replace_flag

        # can be used as LHS or RHS
        else:
            self.LHS = False
            self.u = x
            self.w = x
            self.m = no_mask
            self.replace_flag = False

        self.idx = dict()

        if index == slice(None):
            self.idx["indices"] = all_indices

        elif isinstance(index, slice):
            self.idx["indices"] = range(*index.indices(self.w.shape[0]))

        elif isinstance(index, (list, np.ndarray)):
            self.idx["indices"] = index

        else:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

    # accum expression will be evaluated by __setitem__ of underlying container
    # self.__setitem__(item, self.__getitem__(item).__iadd__(value))
    def __iadd__(self, u):
        return self.assign(u, accum=True)

    @memoize
    @capture_operators
    def eval(self, w, m, accum, replace_flag):

        if self.LHS:
            raise TypeError("Mask can only be applied to LHS")

        if w is None:

            w = containers.Vector(
                    shape=(len(self.idx["indices"]),),
                    dtype=self.w.dtype
            )

        c_func.operator(
                function        = "extractSubvector",
                accum           = accum,
                replace_flag    = replace_flag,
                w               = w,
                m               = m,
                u               = self.u,
                **self.idx
        )

        return w

    @memoize
    def assign(self, u, accum=False):

        if isinstance(u, _Expression):
            u = u.eval()

        kwargs = {}

        if isinstance(u, self.w.dtype):
            function = "assignVectorConst"
            kwargs["val"] = u

        elif isinstance(u, containers.Vector):
            function = "assignSubvector"
            kwargs["u"] = u

        else:
            raise TypeError("Can't assign from non-matrix type")

        c_func.operator(
                function        = function,
                replace_flag    = self.replace_flag,
                accum           = accum,
                w               = self.w,
                m               = self.m,
                **self.idx,
                **kwargs
        )

        return self.w


# TODO don't eval transpose unless necessary
# Try to use transpose view where possible
class Transpose(_Expression):

    def __init__(self, A):
        self.A = A

    @memoize
    @capture_operators
    def eval(self, C, M, accum, replace_flag):

        if C is None:

            C = containers.Matrix(
                    shape=self.A.shape,
                    dtype=self.A.dtype
            )

        c_func.operator(
                function        = "transpose",
                accum           = accum,
                replace_flag    = replace_flag,
                C               = C,
                M               = M,
                A               = self.A,
        )

        return C
