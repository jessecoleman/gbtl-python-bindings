from abc import ABC, abstractmethod
from attr import attrs, attrib
from contextlib import ContextDecorator
from functools import wraps
import numpy as np

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
    # operator instances
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

    def __init__(a_binary_op):
        self.a_binary_op = a_binary_op


class Monoid(BinaryOp, ContextDecorator):

    additive    = 0
    boolean     = "false"
    minimum     = "MinIdentity"

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


class ReplaceFlag(ContextDecorator):
    
    def __init__(self, flag):
        
        self.flag = flag

    def __enter__(self):
        
        global _replace
        _replace.append(self)
        return self

    def __exit__(self):
        
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

# default semirings
ArithmeticSemiring = Semiring(BinaryOp.plus, Monoid.additive, BinaryOp.times)
LogicalSemiring = Semiring(BinaryOp.logical_or, Monoid.boolean, BinaryOp.logical_and)
MinPlusSemiring = Semiring(BinaryOp.minimum, Monoid.minimum, BinaryOp.plus)
MaxTimesSemiring = Semiring(BinaryOp.maximum, Monoid.additive, BinaryOp.times)
MinSelect2ndSemiring = Semiring(BinaryOp.minimum, Monoid.minimum, BinaryOp.second)
MaxSelect2ndSemiring = Semiring(BinaryOp.maximum, Monoid.additive, BinaryOp.second)
MinSelect1stSemiring = Semiring(BinaryOp.minimum, Monoid.minimum, BinaryOp.first)
MaxSelect1stSemiring = Semiring(BinaryOp.maximum, Monoid.additive, BinaryOp.first)


###############################################################################
####                    Functions to use operators with                   #####
###############################################################################

_accum      = [None]
_ops        = [ArithmeticSemiring]
_replace    = [NoReplace]

def get_accum():
    return _accum[-1]

def get_replace():
    return _replace[-1]

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
        raise Error("rows of A and columns of B must be equal")

@eval_expressions
@operator_type(Semiring)
def vxm(semiring, A, B, C=None):
#def vxm(C, M, accum, semiring, A, B, replace_flag):

    if A.shape[0] == B.shape[0]:
        return expr.VXM(semiring, A, B, C)

    else:
        raise Error("length of A and columns of B must be equal")

@eval_expressions
@operator_type(Semiring)
def mxv(semiring, A, B, C=None):

    if A.shape[1] == B.shape[0]:
        return expr.MXV(semiring, A, B, C)

    else:
        raise Error("rows of A and length of B must be equal")


@eval_expressions
@operator_type(BinaryOp)
def eWiseMult(binary_op, A, B, C=None):

    if 1 == len(A.shape) == len(B.shape):
        return expr.EWiseMultVector(binary_op, A, B, C)

    elif 2 == len(A.shape) == len(B.shape):
        return expr.EWiseMultMatrix(binary_op, A, B, C)

    else:
        raise Error("A and B must have the same dimension")


@eval_expressions
@operator_type(BinaryOp)
def eWiseAdd(binary_op, A, B, C=None):

    if 1 == len(A.shape) == len(B.shape):
        return expr.EWiseAddVector(binary_op, A, B, C)

    elif 2 == len(A.shape) == len(B.shape):
        return expr.EWiseAddMatrix(binary_op, A, B, C)

    else:
        raise Error("A and B must have the same dimension")

@eval_expressions
@operator_type(UnaryOp)
def apply(unary_op, A, C=None):

    if 2 == len(A.shape):
        return expr.ApplyMatrix(unary_op, A, C)

    elif 1 == len(A.shape):
        return expr.ApplyVector(unary_op, A, C)

@eval_expressions
@operator_type(Monoid)
def reduce(monoid, A, C=None):

    return expr.ReduceExpression(monoid, A, C)

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

