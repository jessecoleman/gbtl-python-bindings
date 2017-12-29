from abc import ABC, abstractmethod
from attr import attrs, attrib
from contextlib import ContextDecorator
from functools import wraps
from itertools import product
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

_accum = [None]
_ops = []

def get_accum():
    return _accum[-1]

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

# default accumulators
ArithmeticAccumulate = Accumulator(BinaryOp.plus)
BooleanAccumulate = Accumulator(BinaryOp.logical_and)

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
    return expr.BinaryExpression("mxm", semiring, A, B, out_shape)

@operator_type(Semiring)
def vxm(semiring, A, B):
    out_shape = (B.shape[0],)
    return expr.BinaryExpression("vxm", semiring, A, B, out_shape)

@operator_type(Semiring)
def mxv(semiring, A, B):
    out_shape = (A.shape[1],)
    return expr.BinaryExpression("mxv", semiring, A, B, out_shape)

@operator_type(BinaryOp)
def eWiseMult(binary_op, A, B):
    if len(A.shape) == 2 and len(B.shape) == 2:
        return expr.BinaryExpression("eWiseMultMatrix", binary_op, A, B, A.shape)
    elif len(A.shape) == 1 and len(B.shape) == 1:
        return expr.BinaryExpression("eWiseMultVector", binary_op, A, B, A.shape)

@operator_type(BinaryOp)
def eWiseAdd(binary_op, A, B):
    if len(A.shape) == 2 and len(B.shape) == 2:
        return expr.BinaryExpression("eWiseAddMatrix", binary_op, A, B, A.shape)
    elif len(A.shape) == 1 and len(B.shape) == 1:
        return expr.BinaryExpression("eWiseAddVector", binary_op, A, B, A.shape)
    else:
        raise Error("A and B must have the same dimension")

@operator_type(UnaryOp)
def apply(unary_op, A):

    return expr.ApplyExpression("apply", unary_op, A, A.shape)

@operator_type(Monoid)
def reduce(monoid, A, C=None):
    if hasattr(C, "shape"):
        out_shape = C.shape
    else:
        out_shape = (1,)
    return expr.ReduceExpression(monoid, A)

def extract(A, *indices):
    return expr.MaskedExpression(A, *indices)

def assign(A, *indices):
    return expr.MaskedExpression(A, *indices)

def transpose(A):
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

