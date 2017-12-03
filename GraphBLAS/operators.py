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
no_mask = c.utilities().NoMask()


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

    # keep track of accum precedence
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

    def __init__(self, apply_op, bound_const=""):
        self.apply_unary_op = apply_op
        self.bound_const = bound_const  # if operator is binary, bind constant
        self._modules = dict()

    # partially applied
    class expr(object):
        
        def __init__(self, parent, A):
            self.parent = parent
            self.A = A

        def _get_module(self, C, accum):
            
            # get module
            m_args = (self.A.dtype, C.dtype, accum)

            try:
                module = self.parent._modules[m_args]
            except KeyError:
                module = c.get_apply(
                        self.parent.apply_unary_op, 
                        self.parent.bound_const, 
                        *m_args
                )
                self.parent._modules[m_args] = module

            return module

        def __radd__(self, other):
            if isinstance(other, tuple):
                return self(other, accumulator)
            elif isinstance(other, type(self.A)):
                raise Exception("type")
            raise Exception("type")

        def __call__(self, C=None, accum=None):

            mask = no_mask
            replace_flag = False

            if C is None:
                out = self.A._get_out_shape()

            elif isinstance(C, tuple):
                out, mask, replace_flag = C

            else: out = C

            # accum is bound at the module level
            m = self._get_module(out, accum)
            # cpp function call
            m.apply(
                out.mat,
                self.A.mat,
                mask,
                replace_flag
            )
            return out

    def __call__(self, A, C=None, accum=None):

        # evaluate A before performing apply
        if callable(A): 
            A = A()

        # return partial expression
        if C is None:
            return Apply.expr(self, A)

        # return evaluated expression
        else: 
            return Apply.expr(self, A)(C)


class Semiring(ContextDecorator):

    # keep track of semiring precedence
    stack = []

    _ops = namedtuple("_ops", "add_binaryop add_identity mult_binaryop")
    _modules = dict()

    def __init__(self, add_binop, add_idnty, mul_binop):
        self._ops = Semiring._ops(
                add_binaryop=add_binop, 
                add_identity=add_idnty, 
                mult_binaryop=mul_binop
        )
        self._modules = dict()

    class expr(object):
        
        def __init__(self, parent, A, B):
            self.parent = parent
            self.A = A
            self.B = B

        def _get_module(self, C, accum):
            
            # m_args provide a key to the modules dictionary
            m_args = (
                    self.A.dtype, 
                    self.B.dtype, 
                    C.dtype, 
                    accum
            )

            try:
                module = self.parent._modules[m_args]

            except KeyError:
                module = c.get_semiring(self.parent._ops, *m_args)
                self.parent._modules[m_args] = module

            return module

        def __radd__(self, other):
            # C[:] += A + B
            if isinstance(other, tuple):
                return self(other)
            raise Exception("type")

        def __call__(self, C=None, accum=None):

            mask = no_mask
            replace_flag = False

            if C is None:
                out = self.A._get_out_shape(self.B)

            elif isinstance(C, tuple):
                out, mask, replace_flag = C

            else: out = C

            # accum is bound at the module level
            m = self._get_module(out, accum)
            # cpp function call
            m.apply(
                out.mat,
                self.B.mat,
                self.A.mat,
                mask,
                replace_flag
            )
            return out

    def eval(self, op, A, B, C, accum):

        # if A or B need to be evaluated before continuing
        if callable(A): A = A()
        if callable(B): B = B()

        if C is not None:
            return Semiring.expr(self, A, B)(C)

        else: 
            return Semiring.expr(self, A, B)

    # mask and replace are configured at evaluation by C param
    # accum is optionally configured at evaluation
    def eWiseAdd(self, A, B, C=None, accum=None):
        return self.eval("eWiseAdd", A, B, C, accum)

    def dot(self, A, B, C=None, accum=None):
        pass

    def eWiseMult(self, A, B, C=None, accum=None):
        return self.eval("eWiseMult", A, B, C, accum)

    def mxm(self, A, B, C=None, accum=None):
        return self.eval("mxm", A, B, C, accum)

    def mxv(self, A, B, C=None, accum=None):
        return self.eval("mxv", A, B, C, accum)

    def vxm(self, A, B, C=None, accum=None):
        return self.eval("vxm", A, B, C, accum)

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
