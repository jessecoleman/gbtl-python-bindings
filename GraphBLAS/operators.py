from abc import ABCMeta
from contextlib import ContextDecorator
from collections import OrderedDict
from .boundinnerclass import BoundInnerClass
from . import c_functions as c

cache_flag = False

__all__ = [
    "Apply",
    "Accumulator",
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


class Expression(object, metaclass=ABCMeta):
    
    #@abstractmethod
    def eval():
        pass


class Accumulator(ContextDecorator):

    # keep track of accum precedence
    stack = []

    def __init__(self, op):
        self.binary_op = op

    def __enter__(self):
        global accumulator
        Accumulator.stack.append(accumulator)
        accumulator = self
        return self

    def __exit__(self, *errors):
        global accumulator
        accumulator = Accumulator.stack.pop()
        return False

    def __repr__(self):
        return self.binaryop


# default accumulators
NoAccumulate = Accumulator("NoAccumulate")
ArithmeticAccumulate = Accumulator(binary_ops.plus)
BooleanAccumulate = Accumulator(binary_ops.logical_and)


# make sure accum doesn't go out of scope before evaluating expressions
class Apply(object):

    def __init__(self, op, bound_const=None):
        self.unary_op = (op, bound_const)


    # partially applied
    @BoundInnerClass
    class expr(Expression):

        def __init__(self, apply, A):
            self.unary_op = apply.unary_op
            self.A = A

        def eval(self, C=None, accum=NoAccumulate):

            if C is None:
                C = self.A._out_container()

            if hasattr(C, "masked"): 
                out = C.masked()

            else: 
                out = C

            return c.apply(
                accum.binary_op,
                self.unary_op,
                out.replace_flag,
                A = self.A,
                C = out.container,
                M = out.mask
            )

        def __repr__(self):
            return str(self.eval())

    def __call__(self, A, C=None, accum=None):

        if C is None and accum is not None:
            raise Exception("if accum is defined, expression needs to be evaluated on the spot")

        if isinstance(A, Expression):
            A = A.eval()

        part = self.expr(A)
        if C is None: return part
        else: return part.eval(C, accum)


class Semiring(ContextDecorator):

    # keep track of semiring precedence
    stack = []

    def __init__(self, add_binop, add_idnty, mul_binop):
        self.binary_ops = (add_binop, add_idnty, mul_binop)

    def __enter__(self):
        global semiring
        Semiring.stack.append(semiring)
        semiring = self
        return self

    def __exit__(self, *errors):
        global semiring
        semring = Semiring.stack.pop()
        return False

    @BoundInnerClass
    class expr(Expression):

        def __init__(self, semiring, op, A, B):
            self.semiring = semiring.binary_ops
            self.binary_op = op
            self.A = A
            self.B = B

        def eval(self, C=None, accum=NoAccumulate):

            if C is None:
                C = self.A._out_container(self.B)

            # if C is not masked, mask it with NoMask
            if hasattr(C, "masked"): 
                out = C.masked()
            else: 
                out = C

            return c.semiring(
                self.binary_op,
                self.semiring,
                accum.binary_op,
                out.replace_flag,
                A = self.A,
                B = self.B,
                C = out.container,
                M = out.mask
            )

        def __repr__(self):
            return str(self.eval())

    def partial(self, op, A, B, C, accum):

        if C is None and accum is not None:
            raise Exception("if accum is defined, expression needs to be evaluated on the spot")

        if isinstance(A, Expression): A = A.eval()
        if isinstance(B, Expression): B = B.eval()

        part = self.expr(op, A, B)
        if C is None: return part
        else: return part.eval(C, accum)

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


# default binary operators
Identity = Apply(unary_ops.identity)
AdditiveInverse = Apply(unary_ops.additive_inverse)
MultiplicativeInverse = Apply(unary_ops.multiplicative_inverse)


# default semirings
ArithmeticSemiring = Semiring(binary_ops.plus, identities.additive, binary_ops.times)
LogicalSemiring = Semiring(binary_ops.logical_or, identities.boolean, binary_ops.logical_and)
MinPlusSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.plus)
MaxTimesSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.times)
MinSelect2ndSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.second)
MaxSelect2ndSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.second)
MinSelect1stSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.first)
MaxSelect1stSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.first)
