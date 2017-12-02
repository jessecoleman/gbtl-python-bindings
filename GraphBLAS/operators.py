from functools import partial
from contextlib import ContextDecorator
from collections import namedtuple
from GraphBLAS import compile_c as c

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


class Accumulator(ContextDecorator):

    stack = []

    def __init__(self, accum_binary_op):
        self.accum_binary_op = accum_binary_op

    def __enter__(self):
        global accumulator
        # save accumulator precedence
        Accumulator.stack.append(accumulator)
        accumulator = self
        return self

    def __exit__(self, *errors):
        global accumulator
        # reset accumulator
        accumulator = Accumulator.stack.pop()
        return False

    def __str__(self):
        return self.accum_binary_op


# make sure accum doesn't go out of scope before evaluating expressions
class Apply(object):

    def __init__(self, app_op, bound_const=""):
        self._ap = app_op
        self._const = bound_const  # if operator is binary, bind constant
        self._modules = dict()

    def _get_module(self, A, C=None, accum=None):
        
        # if C is none, cast
        if C is None: ctype = A.dtype
        else: ctype = C.dtype

        # get module
        m_args = (A.dtype, ctype, accum)
        try:
            module = self._modules[m_args]
        except KeyError:
            # atype ctype op const accum
            print(self._ap, self._const)
            module = c.get_apply(self._ap, self._const, *m_args)
            self._modules[m_args] = module
        return module

    # accum is bound at the module level so don't need to pass into lambda
    def __call__(self, A, C=None, accum=None):

        # return partial function
        def part(C=None, accum=accum):
            if C is None: C = A._combine(B)
            m = self._get_module(A, C, accum)
            m.apply(
                C.mat,
                A.mat,
                C._mask,
                C._repl
            )
            return C

        if C is None: return part
        else: return part(C)


class Semiring(ContextDecorator):

    stack = []

    _ops = namedtuple("_ops", "add_binaryop add_identity mult_binaryop")
    _modules = dict()

    def __init__(self, add_binop, add_idnty, mul_binop):
        if add_binop is None or add_idnty is None or mul_binop is None:
            print("constructing Semiring")
            print(add_binop, add_idnty, mul_binop)
        self._ops = self._ops(add_binaryop=add_binop, 
                              add_identity=add_idnty, 
                              mult_binaryop=mul_binop)
        self._modules = dict()

    def _get_module(self, A, B, C=None, accum=None):
        
        # if C is none, upcast
        if C is None: ctype = C.upcast(A.dtype, B.dtype)
        else: ctype = C.dtype

        # m_args provide a key to the modules dictionary
        m_args = (A.dtype, B.dtype, ctype, accum)
        try:
            module = self._modules[m_args]
        except KeyError:
            module = c.get_semiring(self._ops, *m_args)
            self._modules[m_args] = module
        return module

    def eval(self, op, A, B, C, accum):

        # if A or B need to be evaluated before continuing
        if callable(A): A()
        if callable(B): B()

        def part(C=None, accum=accum):
            # get empty matrix with the correct output size
            if C is None: C = A._combine(B)
            m = self._get_module(A, B, C, accum)
            getattr(m, op)(
                    C.mat, 
                    A.mat, 
                    B.mat, 
                    C._mask, 
                    C._repl
            )
            return C

        if C is None: return part
        else: return part(C)

    # mask and replace are configured at evaluation by C param
    # accum is optionally configured at evaluation
    def eWiseAdd(self, A, B, C=None, accum=None):
        return self._partial("eWiseAdd", A, B, C, accum)

    def dot(self, A, B, C=None, accum=None):
        pass

    def eWiseMult(self, A, B, C=None, accum=None):
        return self._partial("eWiseMult", A, B, C, accum)

    def mxm(self, A, B, C=None, accum=None):
        return self._partial("mxm", A, B, C, accum)

    def mxv(self, A, B, C=None, accum=None):
        return self._partial("mxv", A, B, C, accum)

    def vxm(self, A, B, C=None, accum=None):
        return self._partial("vxm", A, B, C, accum)

    def __enter__(self):
        global semiring
        # save semiring precedence
        Semiring.stack.append(semiring)
        semiring = self
        return self

    def __exit__(self, *errors):
        global semiring
        # reset semiring
        semring = Semiring.stack.pop()
        return False


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
# TODO The following identity only works for unsigned domains
MaxTimesSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.times)
MinSelect2ndSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.second)
MaxSelect2ndSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.second)
MinSelect1stSemiring = Semiring(binary_ops.minimum, identities.minimum, binary_ops.first)
MaxSelect1stSemiring = Semiring(binary_ops.maximum, identities.additive, binary_ops.first)
