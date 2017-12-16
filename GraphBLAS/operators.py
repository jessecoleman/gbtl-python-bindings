from contextlib import ContextDecorator
from collections import namedtuple
from .boundinnerclass import BoundInnerClass
from . import compile_c as c

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


class Expression(object):

    def eval(self, C=None, accum=None):

        if C is None:
            C = self._out()
            out = C.masked()

        else:
            # if C is not masked, mask it with NoMask
            # TODO check that error is handled correctly here
            try: 
                out = C.masked()
            except: 
                out = C

        # cpp function call
        return self._call_cpp(out, accum)


class Accumulator(ContextDecorator):

    # keep track of accum precedence
    stack = []

    def __init__(self, op):
        self.binaryop = op

    def __enter__(self):
        global accumulator
        Accumulator.stack.append(accumulator)
        accumulator = self
        return self

    def __exit__(self, *errors):
        global accumulator
        accumulator = Accumulator.stack.pop()
        return False

    def __str__(self):
        return self.binaryop


# make sure accum doesn't go out of scope before evaluating expressions
class Apply(object):

    def __init__(self, op, bound_const=None):
        self.unaryop = op
        self.bound_const = bound_const  # if operator is binary, bind constant
        self._modules = dict()

    # partially applied
    @BoundInnerClass
    class expr(Expression):
        
        def __init__(self, apply, A):
            self.apply = apply
            self.A = A

        def _get_module(self, out, accum):

            # get module
            m_args = (
                    self.A.dtype, 
                    out.container.dtype, 
                    out.mtype, 
                    accum
            )

            if m_args not in self.apply._modules:
                module = c.get_apply(
                        self.apply.unaryop, 
                        self.apply.bound_const, 
                        *m_args
                )
                self.apply._modules[m_args] = module

            return self.apply._modules[m_args]

        def _out(self):
            return self.A._out_container()

        def _call_cpp(self, out, accum):
            m = self._get_module(out, accum)
            m.apply(
                out.container.mat,
                out.mask,
                self.A.mat,
                out.replace_flag
            )
            return out.container

        def __str__(self):
            return str(self.eval(None, None))

    def __call__(self, A, C=None, accum=None):

        if C is None and accum is not None:
            raise Exception("if accum is defined, expression needs to be evaluated on the spot")

        # evaluate A before performing apply
        if isinstance(A, Expression): 
            A = A.eval()

        part = self.expr(A)
        # return partial expression
        if C is None: return part
        # return evaluated expression
        else: return part.eval(C, accum)


class Semiring(ContextDecorator):

    # keep track of semiring precedence
    stack = []

    def __init__(self, add_binop, add_idnty, mul_binop):
        self._ops = (add_binop, add_idnty, mul_binop)
        self._modules = dict()

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
            self.semiring = semiring
            self.op = op
            self.A = A
            self.B = B

        def _get_module(self, out, accum):

            # m_args provide a key to the modules dictionary
            m_args = (
                    self.A.dtype,
                    self.B.dtype,
                    out.container.dtype,
                    out.mtype,
                    accum
            )

            if m_args not in self.semiring._modules:
                module = c.get_semiring(*self.semiring._ops, *m_args)
                self.semiring._modules[m_args] = module

            return self.semiring._modules[m_args]


        def _out(self):
            return self.A._out_container(self.B)
        
        def _call_cpp(self, out, accum):
            m = self._get_module(out, accum)
            getattr(m, self.op)(
                out.container.mat,
                out.mask,
                self.A.mat,
                self.B.mat,
                out.replace_flag
            )
            return out.container


        def __str__(self):
            return str(self.eval(None, None))

    # TODO decide how to pass accum in
    def partial(self, op, A, B, C, accum):
        
        if C is None and accum is not None:
            raise Exception("if accum is defined, expression needs to be evaluated on the spot")

        # if A or B need to be evaluated before continuing
        if isinstance(A, Expression): A = A.eval()
        if isinstance(B, Expression): B = B.eval()

        part = self.expr(op, A, B)
        if C is None: return part
        else: return part.eval(C, accum)

    # mask and replace are configured at evaluation by C param
    # accum is optionally configured at evaluation
    def eWiseAdd(self, A, B, C=None, accum=None):
        return self.partial("eWiseAdd", A, B, C, accum)

    def dot(self, A, B, C=None, accum=None):
        pass

    def eWiseMult(self, A, B, C=None, accum=None):
        return self.partial("eWiseMult", A, B, C, accum)

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


# default accumulators
NoAccumulate = Accumulator(None)
ArithmeticAccumulate = Accumulator(binary_ops.plus)
BooleanAccumulate = Accumulator(binary_ops.logical_and)


# default semirings
ArithmeticSemiring = Semiring(binary_ops.plus, identities.additive, binary_ops.times)
LogicalSemiring = Semiring(binary_ops.logical_or, identities.boolean, binary_ops.logical_and)
MinPlusSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.plus)
MaxTimesSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.times)
MinSelect2ndSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.second)
MaxSelect2ndSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.second)
MinSelect1stSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.first)
MaxSelect1stSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.first)
