from contextlib import ContextDecorator
from collections import namedtuple
from GraphBLAS import compile_c as c

__all__ = [
    'Apply', 
    'Accumulator', 
    'Semiring',
    'BooleanAccumulate',
    'ArithmeticAccumulate',
    'Identity',
    'AdditiveInverse',
    'MultiplicativeInverse',
    'ArithmeticSemiring', 
    'MinPlusSemiring', 
    'MaxTimesSemiring', 
    'LogicalSemiring',
    'MinSelect2ndSemiring', 
    'MaxSelect2ndSemiring',
    'MinSelect1stSemiring',
    'MaxSelect1stSemiring',
    'unary_ops',
    'binary_ops', 
    'identities'
]

accumulator = None
semiring = None

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

print(identities.additive)
print(binary_ops.logical_and)

class Accumulator(ContextDecorator):

    # TODO create stack to trace previous accumulators
    stack = []

    def __init__(self, acc_binop):
        self._ac = acc_binop

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

NoAccumulate = Accumulator(None)
ArithmeticAccumulate = Accumulator(binary_ops.plus)
BooleanAccumulate = Accumulator(binary_ops.logical_and)

# make sure accum doesn't go out of scope before evaluating expressions
class Apply(object):

    def __init__(self, app_op, bound_const=None):
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
            module = c.get_apply(self._ap, self._const, *m_args)
            self._modules[m_args] = module
        return module

    # accum is bound at the module level so don't need to pass into lambda
    def __call__(self, A, C=None, accum=None):
        partial = lambda C, accum=accum:\
                self._get_module(A, C, accum)\
                .apply(C.mat, A.mat, C._mask, C._repl)
        print("C", C)
        if C is None: return partial
        else: partial(C)
        return C

Identity = Apply(unary_ops.identity)
AdditiveInverse = Apply(unary_ops.additive_inverse)
MultiplicativeInverse = Apply(unary_ops.multiplicative_inverse)

class Semiring(ContextDecorator):

    stack = []

    _ops = namedtuple('_ops', 'add_binaryop add_identity mult_binaryop')
    _modules = dict()

    def __init__(self, add_binop, add_idnty, mul_binop):
        self._ops = self._ops(add_binaryop=add_binop, 
                              add_identity=add_idnty, 
                              mult_binaryop=mul_binop)
        self._modules = dict()

    def _get_module(self, A, B, C=None, accum=None):
        
        # if C is none, upcast
        if C is None: ctype = C.upcast(A.dtype, B.dtype)
        else: ctype = C.dtype

        m_args = (A.dtype, B.dtype, ctype, accum)
        try:
            module = self._modules[m_args]
        except KeyError:
            module = c.get_semiring(self._ops, *m_args)
            # cache module
            self._modules[m_args] = module
        return module

    # TODO decide how to allow user to override accum
    def eWiseAdd(self, A, B, C=None, accum=None):
        partial = lambda C, accum=accum:\
                self._get_module(A, B, C, accum)\
                .eWiseAdd(C.mat, A.mat, B.mat, C._mask, C._repl)

        if C is None: return partial
        else: partial(C)
        return C

    def dot(self, A, B, C=None, accum=None):
        pass

    def eWiseMult(self, A, B, C=None, accum=None):
        partial = lambda C, accum=accum:\
            self._get_module(A, B, C, accum)\
            .eWiseMult(C.mat, A.mat, B.mat, C._mask, C._repl)
        if C is None: return partial
        else: partial(C)
        return C

    def mxm(self, A, B, C=None, accum=None):
        partial = lambda C, accum=accum:\
            self._get_module(A, B, C, accum)\
            .mxm(C.mat, A.mat, B.mat, C._mask, C._repl)
        if C is None: return partial
        else: partial(C)
        return C

    def mxv(self, A, B, C=None, accum=None):
        partial = lambda C, accum=accum:\
            self._get_module(A, B, C, accum)\
            .mxv(C.mat, A.mat, B.mat, C._mask, C._repl)
        if C is None: return partial
        else: partial(C)
        return C

    def vxm(self, A, B, C=None, accum=None):
        partial = lambda C, accum=accum:\
            self._get_module(A, B, C, accum)\
            .vxm(C.mat, A.mat, B.mat, C._mask, C._repl)
        if C is None: return partial
        else: partial(C)
        return C

    def __enter__(self):
        global semiring
        # save semiring precedence
        Semiring.stack.append(semiring)
        semiring = self
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # reset semiring
        semring = Semiring.stack.pop()
        return False

ArithmeticSemiring = Semiring(binary_ops.plus, identities.additive, binary_ops.times)
LogicalSemiring = Semiring(binary_ops.logical_or, identities.false, binary_ops.logical_and)
MinPlusSemiring = Semiring(binary_ops.min, identities.min_identity, binary_ops.plus)
# TODO The following identity only works for unsigned domains
MaxTimesSemiring = Semiring(binary_ops.max, identities.additive, binary_ops.tmes)
MinSelect2ndSemiring = Semiring(binary_ops.min, identities.min_identity, binary_ops.second)
MaxSelect2ndSemiring = Semiring(binary_ops.max, identities.additive, binary_ops.second)
MinSelect1stSemiring = Semiring(binary_ops.min, identities.min_identity, binary_ops.first)
MaxSelect1stSemiring = Semiring(binary_ops.max, identities.additive, binary_ops.first)
