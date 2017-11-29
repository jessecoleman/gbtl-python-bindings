from contextlib import ContextDecorator
from enum import Enum
from collections import namedtuple
from GraphBLAS import compile_c as c
from GraphBLAS import Matrix, Vector

__all__ = ['Apply', 'Accumulator', 'Semiring',
           'BooleanAccumulate',
           'ArithmeticAccumulate',
           'Identity',
           'AdditiveInverse',
           'MultiplicativeInverse',
           'ArithmeticSemiring', 
           'MinPlusSemiring', 
           'MaxTimesSemiring', 
           'MinSelect2ndSemiring', 
           'MaxSelect2ndSemiring',
           'MinSelect1stSemiring',
           'MaxSelect1stSemiring',
           'BinaryOps', 'Identities']

class UnaryOps(Enum):
    identity = "Identity"
    logical_not = "LogicalNot"
    additive_inverse = "AdditiveInverse"
    multiplicative_inverse = "MultiplicativeInverse"

class BinaryOps(Enum):
    plus = "Plus"
    times = "Times"
    logical_or = "LogicalOr"
    logical_and = "LogicalAnd"
    minimum = "Min"
    maximum = "Max"
    first = "First"
    second = "Second"

class Identities(Enum):
    additive = 0
    boolean = "false"
    minimum = "MinIdentity"

class Accumulator(ContextDecorator):

    replace = True

    def __init__(self, acc_binop):
        self._ac = acc_binop

    def __enter__(self):
        # save previously active accumulator to restore on exit
        try: self._parent_ac = (Matrix._ac, Vector._ac)
        except: self._parent_ac = (None, None)
        self._prev_bases = (Matrix.__bases__, Vector.__bases__)
        # mixin accumulator operators
        Matrix._ac = Vector._ac = self._ac
        Matrix.__bases__ = (self._MatrixAccum,)
        Vector.__bases__ = (self._VectorAccum,)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # reset default operator mixins
        Matrix._ac, Vector._ac = self._parent_ac
        Matrix.__bases__, Vector.__bases__ = self._prev_bases
        return False

    # Matrix mixin to provide operator overloads
    class _MatrixAccum(object):

        expression = namedtuple("expression", "op left right")

        def __iadd__(self, expr):
            print("i add")
            # perform accumulation on expression
            if isinstance(expr, type(self).expression):
                # TODO check that types are correct in module
                return self._sr(expr.op, expr.left, expr.right, self, self)

        def __add__(self, other):
            return type(self).expression(op="eWiseAdd", left=self, right=other)

        def __mul__(self, other):
            return type(self).expression(op="eWiseMult", left=self, right=other)

        def __matmul__(self, other):
            return type(self).expression(op="mxm", left=self, right=other)

    # Vector mixin to provide operator overloads
    class _VectorAccum(object):

        def __add__(self, other):
            return self._sr("eWiseAdd", self, other)

        def __iadd__(self, other):
            return self._sr("mxv", other, self)

NoAccumulate = Accumulator(None)
ArithmeticAccumulate = Accumulator("Plus")
BooleanAccumulate = Accumulator("LogicalAnd")

class Apply(object):
    def __init__(self, app_op, bound_const=None):
        self._ap = app_op
        self._const = bound_const
        self._modules = dict()

    def __call__(self, A, accum=NoAccumulate):
        # get module
        try:
            module = self._modules[(A.dtype, accum._ac)]
        except KeyError:
            module = c.get_apply(A.dtype, self._ap, self._const, accum._ac)
            # cache module
            self._modules[(A.dtype, accum._ac)] = module
        if A._mask is not None:
            module.apply(A.mat, A._mask, A.mat, False)
        else:
            module.apply(A.mat, A.mat, False)
        return A

Identity = Apply("Identity")
AdditiveInverse = Apply("AdditiveInverse")
MultiplicativeInverse = Apply("MultiplicativeInverse")

class Semiring(ContextDecorator):

    _ops = namedtuple('_ops', 'add_binaryop add_identity mult_binaryop')
    _modules = dict()

    def __init__(self, add_binop, add_idnty, mul_binop):
        self._ops = self._ops(add_binaryop=add_binop, 
                              add_identity=add_idnty, 
                              mult_binaryop=mul_binop)
        self._modules = dict()

    def __call__(self, op, A, B, C=None, accum=NoAccumulate):
        # get module
        try:
            module = self._modules[(A.dtype, B.dtype, accum._ac)]
        except KeyError:
            module = c.get_semiring(
                    A.dtype, B.dtype,
                    semiring=self._ops,
                    accum=accum._ac
            )
            # cache module
            self._modules[(A.dtype, B.dtype, accum._ac)] = module
        # initialize C
        if C is None:
            # TODO test dimensions are correct for each operator
            if isinstance(A, Matrix) and isinstance(B, Matrix):
                C = Matrix(
                        shape=(B.shape[0], A.shape[1]), 
                        dtype=c.upcast(A.dtype, B.dtype)[0]
                )
            elif isinstance(A, Matrix) and isinstance(B, Vector):
                C = Vector(
                        shape=(A.shape[1]), 
                        dtype=c.upcast(A.dtype, B.dtype)[0]
                )
            elif isinstance(A, Vector) and isinstance(B, Matrix):
                C = Vector(
                        shape=(A.shape[0]), 
                        dtype=c.upcast(A.dtype, B.dtype)[0]
                )
            elif isinstance(A, Vector) and isinstance(B, Vector):
                C = Vector(
                        shape=(max(A.shape[0], B.shape[0])), 
                        dtype=c.upcast(A.dtype, B.dtype)[0]
                )
        # call operator on semiring
        getattr(module, op)(C.mat, A.mat, B.mat)
        return C

    def add(self, A, B):
        return self("eWiseAdd", A.mat, B.mat)

    def dot(self, A, B):
        pass

    def multiply(self, A, B):
        return self("eWiseMult", A.mat, B.mat)

    def matmul(self, A, B):
        return self("mxm", A.mat, B.mat)

    def __enter__(self):
        # tell Matrix and Vector objects where to get semiring operators
        try: self._parent_sr = (Matrix._sr, Vector._sr)
        except: self._parent_sr = (None, None)
        Matrix._sr = Vector._sr = self
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # reset default operator mixins
        Matrix._sr, Vector._sr = self._parent_sr
        return False

ArithmeticSemiring = Semiring("Plus", "0", "Times")
LogicalSemiring = Semiring("LogicalOr", "false", "LogicalAnd")
MinPlusSemiring = Semiring("Min", "MinIdentity", "Plus")
# TODO The following identity only works for unsigned domains
MaxTimesSemiring = Semiring("Max", "0", "Times")
MinSelect2ndSemiring = Semiring("Min", "MinIdentity", "Second")
MaxSelect2ndSemiring = Semiring("Max", "0", "Second")
MinSelect1stSemiring = Semiring("Min", "MinIdentity", "First")
MaxSelect1stSemiring = Semiring("Max", "0", "First")
