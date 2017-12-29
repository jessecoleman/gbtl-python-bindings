from abc import ABC, abstractmethod
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
def lazy_eval(func):

    @wraps(func)
    def new_func(self, *args, **kwargs):

        C = func(self, *args, **kwargs)
        self._eval = C
        return C

    return new_func


class _Expression(ABC):
    
    @abstractmethod
    def eval(self, other=None, accum=None): pass

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


class BinaryExpression(_Expression):

    def __init__(self, f, op, A, B, out_shape):
        self.f = f
        self.op = op
        self.A = A
        self.B = B
        self.out_shape = out_shape

    @lazy_eval
    def eval(self, out=None, accum=None, replace_flag=None):

        # TODO fix container construction
        if out is None:
            out = self.A._out_container(self.B)

        # if out is not masked, apply NoMask
        try: out = out[:]
        except TypeError: pass

        c_func.operator(
            function        = self.f,
            operation       = self.op,
            accum           = accum,
            replace_flag    = out.replace_flag,
            A               = self.A,
            B               = self.B,
            C               = out.C,
            M               = out.M
        )

        return out.C


class ApplyExpression(_Expression):

    def __init__(self, f, op, A, out_shape):
        self.f = f
        self.op = op
        self.A = A
        self.out_shape = out_shape

    @lazy_eval
    def eval(self, out=None, accum=None, replace_flag=None):

        # if evaluated with symbol notation
        if replace_flag is None:
            if out is None:
                replace_flag = False
                # TODO improve creation of output
                out = self.A._out_container()

            else:
                replace_flag = out.replace_flag

        # if out is not masked, apply NoMask
        try: out = out[:]
        except TypeError: pass

        c_func.operator(
            function        = self.f,
            operation       = self.op,
            accum           = accum,
            replace_flag    = out.replace_flag,
            A               = self.A,
            C               = out.C,
            M               = out.M
        )

        return out.C


class ReduceExpression(_Expression):
    
    def __init__(self, reduce, A):
        self.reduce = reduce
        self.A = A

    @lazy_eval
    def eval(self, out=None, accum=None, replace_flag=None):

        # if out is not masked, apply NoMask
        try: out = out[:]
        except TypeError: pass

        containers = {"A": self.A}

        # reduce to a scalar
        if out is None:
            containers["C"] = self.reduce.identity
            replace_flag = None

        elif isinstance(out, int):
            containers["C"] = out

        # reduce to a vector
        else:
            containers["C"] = out.C
            containers["M"] = out.M
            replace_flag = out.replace_flag

        result = c_func.operator(
                function        = "reduce",
                operation       = self.reduce,
                accum           = accum,
                replace_flag    = replace_flag,
                **containers
        )

        return result


class _NoMask(object):

    def __init__(self):
        self.container = c_func.no_mask()
        self.dtype = None


class MaskedMatrix(_Expression):

    def __init__(self, C, *mask):

        # if assigning into matrix, use M
        # if extracting from matrix, use idx

        self.C = C
        self.idx = dict()
        self.replace_flag = False

        # TODO only allow replace with mask, not slice

        # replace flag
        if len(mask) > 0 and type(mask[-1]) is bool:
            *mask, self.replace_flag = mask

        # container mask (LHS)
        if len(mask) == 1 and isinstance(mask, containers.Matrix):
            self.M = mask

        # NoMask (LHS)
        if all(m == slice(None, None, None) for m in mask):
            self.M = _NoMask()

            # convert 1D index to 2D
            if len(mask) == 1:
                mask = (*mask, *mask)

        # slice or number index
        if len(mask) == 2:

            # element accessor
            if all(isinstance(i, int) for i in mask):
                self.idx["index"] = mask

            # Matrix index
            else:
                for i, s, dim in zip(mask, C.shape, ("row", "col")):
                    if isinstance(i, slice): 
                        self.idx[dim + "_indices"] = range(*i.indices(s))

                    elif isinstance(i, (list, np.array)):
                        self.idx[dim + "_indices"] = i

                    elif isinstance(i, int):
                        self.idx[dim + "_index"] = i
                    
                    else:
                        raise TypeError("Mask indices can be slice, list or int")

        # TODO
        elif len(mask) > 0:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

    # converts row/col indices into matrix mask
    @property
    def M(self):
        # if mask is set
        if hasattr(self, "_M"):
            return self._M
        
        # if indices are set
        if "row_indices" in self.idx:
            rows = self.idx["row_indices"]
        elif "row_index" in self.idx:
            rows = [self.idx["row_index"]]
        else:
            self._M = NoMask()
            return self.M

        if "col_indices" in self.idx:
            cols = self.idx["col_indices"]
        elif "col_index" in self.idx:
            cols = [self.idx["col_index"]]
        else:
            self._M = NoMask()
            return self.M

        i, j = zip(*product(rows, cols))

        self._M = containers.Matrix(
                ([True] * len(i), (i, j)), 
                shape=self.C.shape, 
                dtype=bool
        )

        return self._M

    @M.setter
    def M(self, M):
        self._M = M

    # accum expression will be evaluated by __setitem__ of underlying container
    def __iadd__(self, other):
        return AccumExpression(other)

    def eval(self, out=None, accum=None, replace_flag=None):
        return self.extract(out, accum, replace_flag)

    @lazy_eval
    def extract(self, out, accum=None, replace_flag=None):

        if "index" in self.idx:
            return self.C.container.extractElement(*self.idx["index"])
    
        # evaluate expression from 
        if replace_flag is None:
            if out is None:
                replace_flag = False
            else:
                replace_flag = out.replace_flag

        # construct container of correct shape and size to extract to
        if out is None:

            if "row_index" in self.idx:
                out = containers.Vector(
                        shape=(len(self.idx["col_indices"]),),
                        dtype = self.C.dtype
                )

            elif "col_index" in self.idx:
                out = containers.Vector(
                        shape=(len(self.idx["row_indices"]),),
                        dtype = self.C.dtype
                )

            else:
                out = containers.Matrix(
                        shape=(
                            len(self.idx["row_indices"]), 
                            len(self.idx["col_indices"])
                        ),
                        dtype=self.C.dtype
                )[:]

        result = c_func.operator(
                function        = "extract",
                accum           = accum,
                replace_flag    = self.replace_flag,
                C               = out.C,
                M               = out.M,
                A               = self.C,
                **self.idx
        )

        return out.C
                    
    @lazy_eval
    def assign(self, assign, accum=None, replace_flag=None):

        if isinstance(assign, self.C.dtype):

            # element setter
            if "index" in self.idx:
                self.C.container.setElement(*self.idx["index"], assign)
                return

            # constant assignment to indices
            else:
                A = assign
                # full index
                idx = self.C[:].idx

        else:
            # if assigning un-indexed container
            if isinstance(assign, Matrix):
                assign = assign[:]

            A = assign.C
            idx = assign.idx

        c_func.operator(
                function        = "assign",
                replace_flag    = self.replace_flag,
                accum           = accum,
                C               = self.C,
                M               = self.M,
                A               = A,
                **idx
        )

        return self.C


class MaskedVector(_Expression):

    def __init__(self, C, *mask):

        # if assigning into matrix, use M
        # if extracting from matrix, use idx

        self.C = C
        self.idx = dict()
        self.replace_flag = False

        # TODO only allow replace with mask, not slice

        # replace flag
        if len(mask) > 0 and type(mask[-1]) is bool:
            *mask, self.replace_flag = mask

        # container mask (LHS)
        if len(mask) == 1 and isinstance(mask, containers.Vector):
            self.M = mask

        # NoMask (LHS)
        if mask == (slice(None, None, None),):
            self.M = _NoMask()

        # slice or number index
        if len(mask) == 1:

            i = mask[0]

            # element accessor
            if isinstance(i, int):
                self.idx["index"] = mask

            # Vector index
            elif isinstance(i, slice):
                self.idx["indices"] = range(*i.indices(*C.shape))

            elif isinstance(i, (list, np.array)):
                self.idx["indices"] = i

            else:
                raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        elif len(mask) > 0:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

    # converts row/col indices into matrix mask
    @property
    def M(self):
        # if mask is set
        if hasattr(self, "_M"):
            return self._M
        
        # if indices are set
        if "indices" in self.idx:

            i = self.idx["indices"]
            self._M = containers.Vector(
                    ([True] * len(i), i),
                    shape=self.C.shape,
                    dtype=bool
            )

        else:
            self._M = _NoMask()

        return self._M

    @M.setter
    def M(self, M):
        self._M = M

    # TODO fix broken shit
    # accum expression will be evaluated by __setitem__ of underlying container
    def __iadd__(self, other):
        return AccumExpression(other)

    def eval(self, out=None, accum=None, replace_flag=None):
        return self.extract(out, accum, replace_flag)

    @lazy_eval
    def extract(self, out, accum=None, replace_flag=None):

        if "index" in self.idx:
            return self.C.container.extractElement(*self.idx["index"])
    
        # evaluate expression from operator notation
        if replace_flag is None:
            if out is None:
                replace_flag = False
            else:
                replace_flag = out.replace_flag

        # construct container to extract to
        if out is None:

            out = containers.Vector(
                    shape=(len(self.idx["indices"]),), 
                    dtype=self.C.dtype
            )[:]

        result = c_func.operator(
                function        = "extract",
                accum           = accum,
                replace_flag    = self.replace_flag,
                C               = out.C,
                M               = out.M,
                A               = self.C,
                **self.idx
        )

        return out.C
                    
    @lazy_eval
    def assign(self, assign, accum=None, replace_flag=None):

        # TODO default replace flag?
        if isinstance(assign, self.C.dtype):

            # element setter
            if "index" in self.idx:
                self.C.container.setElement(*self.idx["index"], assign)
                return

            # TODO figure out masking vs indexing
            # constant assignment to indices
            else:
                A = assign
                # full index
                idx = self.C[:].idx

        else:
            # if assigning un-indexed container
            if isinstance(assign, Vector):
                assign = assign[:]

            A = assign.C
            idx = assign.idx

        c_func.operator(
                function        = "assign",
                replace_flag    = self.replace_flag,
                accum           = accum,
                C               = self.C,
                M               = self.M,
                A               = A,
                **idx
        )

        return self.C

# TODO better interface
class AccumExpression(_Expression):

    def __init__(self, expr):
        self.expr = expr

