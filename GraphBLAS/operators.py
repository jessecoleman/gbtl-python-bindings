from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from functools import wraps

from . import expressions as expr

__all__ = [
    # operator classes
    "Accumulator",
    "UnaryOp",
    "BinaryOp",
    "Monoid",
    "Semiring",
    # functions
    "apply",
    "reduce",
    "extract",
    "assign",
    "Replace",
    "NoReplace",
    # operator instances
    "BooleanAccumulate",
    "ArithmeticAccumulate",
    "Identity",
    "AdditiveInverse",
    "MultiplicativeInverse",
    "PlusMonoid",
    "ArithmeticSemiring",
    "MinPlusSemiring",
    "MaxTimesSemiring",
    "LogicalSemiring",
    "MinSelect2ndSemiring",
    "MaxSelect2ndSemiring",
    "MinSelect1stSemiring",
    "MaxSelect1stSemiring",
]


###############################################################################
####   Operator definitions, provide context managers for expressions     #####
###############################################################################

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

    def __repr__(self):
        return ("<"
            + type(self).__name__
            + " object: {"
            + ", ".join(("{}: {}".format(k, v) 
                for k, v in self.__dict__.items()))
            + "}>")

class UnaryOp(_Op, ContextDecorator):

    identity                = "Identity"
    logical_not             = "LogicalNot"
    additive_inverse        = "AdditiveInverse"
    multiplicative_inverse  = "MultiplicativeInverse"

    def __init__(self, unary_op, bound_const=None):

        self.unary_op = unary_op
        if bound_const is not None:
            self.bound_const = bound_const


class BinaryOp(_Op, ContextDecorator):

    plus        = "Plus"
    times       = "Times"
    logical_or  = "LogicalOr"
    logical_and = "LogicalAnd"
    minimum     = "Min"
    maximum     = "Max"
    first       = "First"
    second      = "Second"

    def __init__(self, binary_op):
        self.binary_op = binary_op


class Monoid(BinaryOp, ContextDecorator):

    additive    = 0
    boolean     = "false"
    minimum     = "MinIdentity"

    def __init__(self, binary_op, identity):

        self.add_binary_op = binary_op
        self.add_identity = identity


class Semiring(Monoid, ContextDecorator):

    def __init__(self, monoid, binary_op):

        self.add_binary_op = monoid.add_binary_op
        self.add_identity = monoid.add_identity
        self.mult_binary_op = binary_op


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


class ReplaceFlag(ContextDecorator):

    def __init__(self, flag):

        self.flag = flag

    def __enter__(self):

        global _replace
        _replace.append(self)
        return self

    def __exit__(self, *args):

        global _replace
        _replace.pop()
        return False


# default accumulators
ArithmeticAccumulate = Accumulator(BinaryOp.plus)
BooleanAccumulate = Accumulator(BinaryOp.logical_and)

# replace flags
Replace = ReplaceFlag(True)
NoReplace = ReplaceFlag(False)

# default unary operators
Identity = UnaryOp(UnaryOp.identity)
AdditiveInverse = UnaryOp(UnaryOp.additive_inverse)
MultiplicativeInverse = UnaryOp(UnaryOp.multiplicative_inverse)

# default monoids
PlusMonoid = Monoid(BinaryOp.plus, Monoid.additive)
LogicalMonoid = Monoid(BinaryOp.logical_or, Monoid.boolean)
MinMonoid = Monoid(BinaryOp.minimum, Monoid.minimum)
MaxMonoid = Monoid(BinaryOp.maximum, Monoid.additive)

# default semirings
ArithmeticSemiring = Semiring(PlusMonoid, BinaryOp.times)
LogicalSemiring = Semiring(LogicalMonoid, BinaryOp.logical_and)
MinPlusSemiring = Semiring(MinMonoid, BinaryOp.plus)
MaxTimesSemiring = Semiring(MaxMonoid, BinaryOp.times)
MinSelect2ndSemiring = Semiring(MinMonoid, BinaryOp.second)
MaxSelect2ndSemiring = Semiring(MaxMonoid, BinaryOp.second)
MinSelect1stSemiring = Semiring(MinMonoid, BinaryOp.first)
MaxSelect1stSemiring = Semiring(MaxMonoid, BinaryOp.first)


###############################################################################
####                    Functions to use operators with                   #####
###############################################################################

_accum      = [None]
_ops        = [ArithmeticSemiring]
_replace    = [NoReplace]

def get_accum():
    return _accum[-1]

def get_replace():
    return _replace[-1].flag


# function decorator to fill in operator from context if not provided
def operator_type(op_type, mult=False):

    def wrapper(operation):

        @wraps(operation)
        def new_func(*args):

            args = list(args)

            if not isinstance(args[0], op_type):
                if not isinstance(args[0], _Op):
                    for op in reversed(_ops):
                        if isinstance(op, op_type):
                            # TODO
                            if mult and isinstance(op, Semiring):
                                args.insert(0, BinaryOp(op.mult_binary_op))
                            else:
                                args.insert(0, op)
                            break
                else:
                    raise Exception("operator must be {}".format(op_type))

            return operation(*args)

        return new_func

    return wrapper


# TODO consider lazy evaluation here
def eval_expressions(function):

    @wraps(function)
    def new_func(*args):

        args = list(args)

        for i, arg in enumerate(args):
            if isinstance(arg, expr._Expression):
                args[i] = arg.eval()

        return function(*args)

    return new_func


@eval_expressions
@operator_type(Semiring)
def mxm(semiring, A, B, C=None):

    if A.shape[0] == B.shape[1]:
        return expr.MXM(semiring, A, B, C)

    else:
        raise Exception("rows of A and columns of B must be equal")


@eval_expressions
@operator_type(Semiring)
def vxm(semiring, A, B, C=None):
#def vxm(C, M, accum, semiring, A, B, replace_flag):

    if A.shape[0] == B.shape[0]:
        return expr.VXM(semiring, A, B, C)

    else:
        raise Exception("length of A and columns of B must be equal")


@eval_expressions
@operator_type(Semiring)
def mxv(semiring, A, B, C=None):

    if A.shape[1] == B.shape[0]:
        return expr.MXV(semiring, A, B, C)

    else:
        raise Exception("rows of A and length of B must be equal")


@eval_expressions
@operator_type(BinaryOp, mult=True)
def eWiseMult(binary_op, A, B, C=None):

    if 1 == len(A.shape) == len(B.shape):
        return expr.EWiseMultVector(binary_op, A, B, C)

    elif 2 == len(A.shape) == len(B.shape):
        return expr.EWiseMultMatrix(binary_op, A, B, C)

    else:
        raise Exception("A and B must have the same dimension")


@eval_expressions
@operator_type(BinaryOp)
def eWiseAdd(binary_op, A, B, C=None):

    if 1 == len(A.shape) == len(B.shape):
        return expr.EWiseAddVector(binary_op, A, B, C)

    elif 2 == len(A.shape) == len(B.shape):
        return expr.EWiseAddMatrix(binary_op, A, B, C)

    else:
        raise Exception("A and B must have the same dimension")


@eval_expressions
@operator_type(UnaryOp)
def apply(unary_op, A, C=None):

    if 2 == len(A.shape):
        return expr.ApplyMatrix(unary_op, A, C)

    elif 1 == len(A.shape):
        return expr.ApplyVector(unary_op, A, C)


# TODO allow no value for monoid
@eval_expressions
@operator_type(Monoid)
def reduce(monoid, A, C=None):

    if 2 == len(A.shape):
        return expr.ReduceMatrix(monoid, A, C)

    elif 1 == len(A.shape):
        return expr.ReduceVector(monoid, A, C)


# TODO
@eval_expressions
def extract(A, indices=None, C=None, M=None, accum=None, replace_flag=False):

    if len(A.shape) == 2:
        idx = expr.IndexedMatrix(A, indices)

    elif len(A.shape) == 1:
        idx = expr.IndexedVector(A, indices)

    if C is not None:
        return idx.eval(C, M, accum, replace_flag)

    else:
        return idx


@eval_expressions
def assign(A, indices=None, C=None, M=None, accum=None, replace_flag=False):

    if len(A.shape) == 2:
        idx = expr.IndexedMatrix(A, indices)

    elif len(A.shape) == 1:
        idx = expr.IndexedVector(A, indices)

    if C is not None:
        return idx.eval(C, M, accum, replace_flag)

    else:
        return idx


@eval_expressions
def transpose(A, C=None):

    return expr.TransposeExpression(A, C)


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

