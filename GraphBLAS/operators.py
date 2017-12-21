from abc import ABC, abstractmethod
from attr import attrs, attrib
from contextlib import ContextDecorator
from collections import OrderedDict
from .boundinnerclass import BoundInnerClass
from . import c_functions as c

cache_flag = False

__all__ = [
    "Accumulator",
    "Apply",
    "Semiring",
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

accumulator = None
semiring = None

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

class _Expression(ABC):
    
    @abstractmethod
    def eval(): pass

    @property
    def evaluated(self):
        if not hasattr(self, "_evaluated"):
            self._evaluated = self.eval()
        return self._evaluated

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


@attrs(cmp=False)
class Accumulator(ContextDecorator):

    # keep track of accum precedence
    stack = []
    binary_op = attrib()

    def __enter__(self):
        global accumulator
        Accumulator.stack.append(accumulator)
        accumulator = self
        return self

    def __exit__(self, *errors):
        global accumulator
        accumulator = Accumulator.stack.pop()
        return False


# default accumulators
NoAccumulate = Accumulator("NoAccumulate")
ArithmeticAccumulate = Accumulator(binary_ops.plus)
BooleanAccumulate = Accumulator(binary_ops.logical_and)


# make sure accum doesn't go out of scope before evaluating expressions
@attrs(cmp=False)
class Apply(object):

    unary_op    = attrib()
    bound_const = attrib(default=None)

    def __call__(self, A, C=None, accum=None):

        if C is None and accum is not None:
            raise Exception("if accum is defined, expression needs to be evaluated on the spot")

        if isinstance(A, _Expression): A = A.eval()

        expr = Apply.Expression(self, A)
        if C is None: return expr
        else: return expr.eval(C, accum)


    @attrs(cmp=False, repr=False)
    class Expression(_Expression):

        apply   = attrib()
        A       = attrib()

        def eval(self, out=None, accum=NoAccumulate):

            if out is None:
                out = self.A._out_container()

            if hasattr(out, "masked"): 
                out = out.masked()

            c.operator(
                self.apply,
                "apply",
                accum.binary_op,
                out.replace_flag,
                A = self.A,
                C = out.C,
                M = out.M
            )

            self._evaluated = out.C
            return self.evaluated


Identity = Apply(unary_ops.identity)
AdditiveInverse = Apply(unary_ops.additive_inverse)
MultiplicativeInverse = Apply(unary_ops.multiplicative_inverse)


@attrs(cmp=False)
class Semiring(ContextDecorator):

    # keep track of semiring precedence
    stack = []
    add_binaryop = attrib()
    add_identity = attrib()
    mult_binaryop = attrib()

    def __enter__(self):
        global semiring
        Semiring.stack.append(semiring)
        semiring = self
        return self

    def __exit__(self, *errors):
        global semiring
        semring = Semiring.stack.pop()
        return False

    def partial(self, op, A, B, C, accum):

        if C is None and accum is not None:
            raise Exception("if accum is defined, expression needs to be evaluated on the spot")

        if isinstance(A, _Expression): A = A.eval()
        if isinstance(B, _Expression): B = B.eval()

        expr = Semiring.Expression(self, op, A, B)
        if C is None: return expr
        else: return expr.eval(C, accum)

    # mask and replace are configured at evaluation by C param
    # accum is optionally configured at evaluation
    def eWiseAddMatrix(self, A, B, C=None, accum=None):
        return self.partial("eWiseAddMatrix", A, B, C, accum)

    def eWiseAddVector(self, A, B, C=None, accum=None):
        return self.partial("eWiseAddVector", A, B, C, accum)

    def dot(self, A, B, C=None, accum=None):
        pass

    def eWiseMultMatrix(self, A, B, C=None, accum=None):
        return self.partial("eWiseMultMatrix", A, B, C, accum)

    def eWiseMultVector(self, A, B, C=None, accum=None):
        return self.partial("eWiseMultVector", A, B, C, accum)

    def mxm(self, A, B, C=None, accum=None):
        return self.partial("mxm", A, B, C, accum)

    def mxv(self, A, B, C=None, accum=None):
        return self.partial("mxv", A, B, C, accum)

    def vxm(self, A, B, C=None, accum=None):
        return self.partial("vxm", A, B, C, accum)


    @attrs(cmp=False, repr=False)
    class Expression(_Expression):

        semiring    = attrib()
        function    = attrib()
        A           = attrib()
        B           = attrib()

        def eval(self, out=None, accum=NoAccumulate):

            if out is None:
                out = self.A._out_container(self.B)

            if hasattr(out, "masked"): 
                out = out.masked()

            c.operator(
                self.semiring,
                self.function,
                accum.binary_op,
                out.replace_flag,
                A = self.A,
                B = self.B,
                C = out.C,
                M = out.M
            )

            self._evaluated = out.C
            return self.evaluated


# default semirings
ArithmeticSemiring = Semiring(binary_ops.plus, identities.additive, binary_ops.times)
LogicalSemiring = Semiring(binary_ops.logical_or, identities.boolean, binary_ops.logical_and)
MinPlusSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.plus)
MaxTimesSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.times)
MinSelect2ndSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.second)
MaxSelect2ndSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.second)
MinSelect1stSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.first)
MaxSelect1stSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.first)
