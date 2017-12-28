from abc import ABC, abstractmethod
from attr import attrs, attrib
from contextlib import ContextDecorator
from functools import wraps
from itertools import product
import numpy as np
from . import c_functions as c

__all__ = [
    "Accumulator",
    "BinaryOp",
    "Monoid",
    "Semiring",
    "apply",
    "reduce",
    "extract",
    "assign",
    "BooleanAccumulate",
    "ArithmeticAccumulate",
    "Identity",
    "AdditiveInverse",
    "MultiplicativeInverse",
    "ArithmeticSemiring",
    "MinPlusSemiring",
    "MaxTimesSemiring",
    "LogicalSemiring",
    "MinSelect2ndSemiring",
    "MaxSelect2ndSemiring",
    "MinSelect1stSemiring",
    "MaxSelect1stSemiring",
    "unary_ops",
    "binary_ops",
    "identities"
]

_accum = [None]
_ops = []


class _Op(ABC):

    def __enter__(self):
        global _ops
        _ops.append(self)
        return self
        
    def __exit__(self, *errors):
        global _ops
        # TODO pop vs. remove
        _ops.remove(self)
        return False


class UnaryOp(_Op, ContextDecorator):

    def __init__(self, unary_op, bound_const=None):
        self.unary_op = unary_op
        if bound_const is not None:
            self.bound_const = bound_const


class BinaryOp(_Op, ContextDecorator):

    def __init__(a_binary_op):
        self.a_binary_op = a_binary_op


class Monoid(BinaryOp, ContextDecorator):

    def __init__(self, a_binary_op, identity):
        self.a_binary_op = a_binary_op
        self.identity = identity


class Semiring(Monoid, ContextDecorator):

    def __init__(self, a_binary_op, identity, m_binary_op):
        self.a_binary_op = a_binary_op
        self.identity = identity
        self.m_binary_op = m_binary_op


class Accumulator(BinaryOp, ContextDecorator):

    def __init__(self, binary_op):
        self.binary_op = binary_op

    def __enter__(self):
        global _accum
        _accum.append(self)
        return self

    def __exit__(self, *errors):
        global _accum
        _accum.pop()
        return False

NoAccumulate = Accumulator("NoAccumulate")

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
    def eval(): pass

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


# reduce, apply
class UnaryExpression(_Expression):

    def __init__(self, f, op, A, out_shape):
        self.f = f
        self.op = op
        self.A = A
        self.out_shape = out_shape

    @lazy_eval
    def eval(self, out=None, accum=NoAccumulate):

        if out is None:
            out = self.A._out_container()

        # if out is not masked, apply NoMask
        try: out = out[:]
        except TypeError: pass

        c.operator(
            function        = self.f,
            operation       = self.op,
            accum           = accum.binary_op,
            replace_flag    = out.replace_flag,
            A               = self.A,
            C               = out.C,
            M               = out.M
        )

        return out.C


class BinaryExpression(_Expression):

    def __init__(self, f, op, A, B, out_shape):
        self.f = f
        self.op = op
        self.A = A
        self.B = B
        self.out_shape = out_shape

    @lazy_eval
    def eval(self, out=None, accum=NoAccumulate):

        # TODO fix container construction
        if out is None:
            out = self.A._out_container(self.B)

        # if out is not masked, apply NoMask
        try: out = out[:]
        except TypeError: pass

        c.operator(
            function        = self.f,
            operation       = self.op,
            accum           = accum.binary_op,
            replace_flag    = out.replace_flag,
            A               = self.A,
            B               = self.B,
            C               = out.C,
            M               = out.M
        )

        return out.C


class ReduceExpression(_Expression):
    
    def __init__(self, reduce, A):
        self.reduce = reduce
        self.A = A

    @lazy_eval
    def eval(self, out, accum=NoAccumulate):

        # if out is not masked, apply NoMask
        try: out = out[:]
        except TypeError: pass

        containers = {"A": self.A}

        # reduce to a scalar
        if out is None:
            containers["C"] = self.reduce.add_identity
            replace_flag = None

        elif isinstance(out, int):
            containers["C"] = out

        # reduce to a vector
        else:
            containers["C"] = out.C
            containers["M"] = out.M
            replace_flag = out.replace_flag

        result = c.operator(
                function        = "reduce",
                operation       = self.reduce,
                accum           = accum.binary_op,
                replace_flag    = replace_flag,
                **containers
        )

        return result


class MaskedExpression(_Expression):
    
    class NoMask(object):

        def __init__(self):
            self.container = c.no_mask()
            self.dtype = None


    def __init__(self, C, *mask):

        self.idx = dict()
        self.C = C
        self.M = self.NoMask()

        self.replace_flag = False

        # TODO only allow replace with mask, not slice

        # replace flag
        if len(mask) > 0 and type(mask[-1]) is bool:
            *mask, self.replace_flag = mask

        # container mask
        if len(mask) == 1 and hasattr(mask[0], "container"):
            self.M = mask

        elif mask == (slice(None, None, None),):
            self.M = self.NoMask()

        # slice or number index
        elif len(mask) == len(C.shape):

            if all(isinstance(i, int) for i in mask):
                self.idx["index"] = mask

            elif len(C.shape) == 1:

                i = mask[0]
                if isinstance(i, slice):
                    self.idx["indices"] = range(*i.indices(*C.shape))
                elif isinstance(i, (list, np.array)):
                    self.idx["indices"] = i

            elif len(C.shape) == 2:

                for i, s, d in zip(mask, C.shape, ("row", "col")):
                    if isinstance(i, slice): 
                        self.idx[d + "_indices"] = list(range(*i.indices(s)))

                    elif isinstance(i, (list, np.array)):
                        self.idx[d + "_indices"] = i

                    elif isinstance(i, int):
                        self.idx[d + "_index"] = i

            else:
                raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

        elif len(mask) > 0:
            raise TypeError("Mask must be boolean Matrix or 2D slice with optional replace flag")

    def applyMask(self, rows, cols):
        i, j = (zip(*product(rows, cols)))

        # TODO
        self.idx["M"] = type(self.C)(
                ([True] * len(i), (i, j)), 
                shape=self.C.shape, 
                dtype=bool
        )

    # TODO fix broken shit
    def __iadd__(self, other):

        if isinstance(other, _Expression):
            return other.eval(self, _accum[-1])

        else:
            return apply(Identity, other).eval(self, _accum[-1])

    def eval(self):
        return self.extract()

    @lazy_eval
    def extract(self, A=None, accum=NoAccumulate):

        if "index" in self.idx:
            return self.C.container.extractElement(*self.idx["index"])

        if A is None:
            shape = None

            if len(self.C.shape) == 1:
                shape = (len(self.idx["indices"]),)

            elif len(self.C.shape) == 2:

                if "row_index" in self.idx:
                    shape = (len(self.idx["col_indices"]),)
                elif "col_index" in self.idx:
                    shape = (len(self.idx["row_indices"]),)
                else:
                    shape = (
                            len(self.idx["row_indices"]), 
                            len(self.idx["col_indices"])
                    )

            # TODO construct appropriate container
            A = type(self.C)(shape=tuple(shape), dtype=self.C.dtype)

        result = c.operator(
                function        = "extract",
                accum           = accum.binary_op,
                replace_flag    = self.replace_flag,
                C               = A,
                M               = self.M,
                A               = self.C,
                **self.idx
        )

        return A
                    
    @lazy_eval
    def assign(self, A, accum=NoAccumulate):

        if isinstance(A, self.C.dtype) and "index" in self.idx:
            self.C.container.setElement(*self.idx["index"])
            return

        c.operator(
                "assign",
                replace_flag    = self.replace_flag,
                accum           = accum.binary_op,
                C               = self.C,
                M               = self.M,
                A               = A,
                **self.idx
        )

        return self.C


# function decorator to fill in operator from context if not provided
def operator_type(op_type):
    
    def wrapper(function):

        @wraps(function)
        def new_func(*args):

            operator, *args = args

            if operator is None:
                for op in reversed(_ops):
                    if isinstance(op, op_type):
                        operator = op
                        break

            elif not isinstance(operator, op_type):
                raise Exception("operator must be of type {}".format(op_type))

            return function(operator, *args)
        
        return new_func

    return wrapper
            
@operator_type(Semiring)
def mxm(semiring, A, B):
    out_shape = (B.shape[0], A.shape[1])
    return BinaryExpression("mxm", semiring, A, B, out_shape)

@operator_type(Semiring)
def vxm(semiring, A, B):
    out_shape = (B.shape[0],)
    return BinaryExpression("vxm", semiring, A, B, out_shape)

@operator_type(Semiring)
def mxv(semiring, A, B):
    out_shape = (A.shape[1],)
    return BinaryExpression("mxv", semiring, A, B, out_shape)

@operator_type(BinaryOp)
def eWiseMult(binary_op, A, B):
    if len(A.shape) == 2 and len(B.shape) == 2:
        return BinaryExpression("eWiseMultMatrix", binary_op, A, B, A.shape)
    elif len(A.shape) == 1 and len(B.shape) == 1:
        return BinaryExpression("eWiseMultVector", binary_op, A, B, A.shape)

@operator_type(BinaryOp)
def eWiseAdd(binary_op, A, B):
    if len(A.shape) == 2 and len(B.shape) == 2:
        return BinaryExpression("eWiseAddMatrix", binary_op, A, B, A.shape)
    elif len(A.shape) == 1 and len(B.shape) == 1:
        return BinaryExpression("eWiseAddVector", binary_op, A, B, A.shape)
    else:
        raise Error("A and B must have the same dimension")

@operator_type(UnaryOp)
def apply(unary_op, A):

    return UnaryExpression("apply", unary_op, A, A.shape)

@operator_type(Monoid)
def reduce(monoid, A):

    return ReduceExpression("reduce", monoid, A)


# TODO
def extract(A, *indices):
    pass

def assign(A, *indices):
    pass


# dictionary of values to build operators
class OperatorMap(dict):
    def __init__(self, keys):
        super(OperatorMap, self).__init__(**keys)
        for key, val in keys.items():
            self[key] = val

    def __getattr__(self, attr):
        return self.get(attr)

    # TODO turn this into field for operator functions
    def __get__(self, instance, owner):
        pass


unary_ops = OperatorMap({
    "identity": "Identity",
    "logical_not": "LogicalNot",
    "additive_inverse": "AdditiveInverse",
    "multiplicative_inverse": "MultiplicativeInverse"
})

binary_ops = OperatorMap({
    "plus": "Plus",
    "times": "Times",
    "logical_or": "LogicalOr",
    "logical_and": "LogicalAnd",
    "minimum": "Min",
    "maximum": "Max",
    "first": "First",
    "second": "Second"
})

identities = OperatorMap({
    "additive": 0,
    "boolean": "false",
    "minimum": "MinIdentity"
})


# default accumulators
ArithmeticAccumulate = Accumulator(binary_ops.plus)
BooleanAccumulate = Accumulator(binary_ops.logical_and)

# default unary operators
Identity = UnaryOp(unary_ops.identity)
AdditiveInverse = UnaryOp(unary_ops.additive_inverse)
MultiplicativeInverse = UnaryOp(unary_ops.multiplicative_inverse)

# default semirings
ArithmeticSemiring = Semiring(binary_ops.plus, identities.additive, binary_ops.times)
LogicalSemiring = Semiring(binary_ops.logical_or, identities.boolean, binary_ops.logical_and)
MinPlusSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.plus)
MaxTimesSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.times)
MinSelect2ndSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.second)
MaxSelect2ndSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.second)
MinSelect1stSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.first)
MaxSelect1stSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.first)

