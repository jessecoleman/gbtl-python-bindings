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
           'AddBinaryOpws', 'Identities']

class UnaryOps(Enum):
    identity = "Identity"
    logical_not = "LogicalNot"
    additive_inverse = "AdditiveInverse"
    multiplicative_inverse = "MultiplicativeInverse"

class AddBinaryOpws(Enum):
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

class Expression(object):
    # semiring operator
    sr_expr = namedtuple("expression", "op left right")
    # apply operator
    ap_expr = namedtuple("expression", "op container")

    def __init__(self):
        pass

    def eval(self):
        pass

    def __iadd__(self, other):
        pass

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __matmul__(self, other):
        pass

    def __invert__(self):
        pass

class Accumulator(ContextDecorator):

    def __init__(self, acc_binop):
        self._ac = acc_binop

    def __enter__(self):
        # save previously active accumulator to restore on exit
        try: self._parent_ac = (Matrix._ac, Vector._ac)
        except: self._parent_ac = (None, None)
        self._prev_bases = (Matrix.__bases__, Vector.__bases__)
        # mixin accumulator operators
        Matrix._ac = self._ac
        Vector._ac = self._ac
        Matrix.__bases__ = (self._Accum,)
        Vector.__bases__ = (self._Accum,)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # reset default operator mixins
        Matrix._ac, Vector._ac = self._parent_ac
        Matrix.__bases__, Vector.__bases__ = self._prev_bases
        return False

    # Matrix mixin to provide operator overloads
    class _Accum(object):

        sr_expr = namedtuple("expression", "op left right")
        ap_expr = namedtuple("expression", "op container")

        # self[item] += assign
        # self.__setitem__(self.__getitem__(item).__iadd__(assign))
        # self.__setitem__(Masked(self).__iadd__(assign))

        # terminating expression
        def __iadd__(self, expr):
            print("expr", expr)
            # remove mask attribute and return
            def get_attr(self, attr):
                if getattr(self,attr) is None:
                    return None
                else:
                    a = getattr(self,attr)
                    setattr(self,attr,None)
                    return a

            # perform semiring operator
            if isinstance(expr, type(self).sr_expr):
                # TODO check that types are correct in module
                print("is sr_expr")
                return self._sr(
                        expr.op, 
                        expr.left, 
                        expr.right, 
                        self, 
                        self._ac,
                        get_attr(self, "_mask")
                )
            # perform apply operator
            elif isinstance(expr, type(self).ap_expr):
                print("is ap_expr")
                return get_attr(self, "_ap")(
                        expr.op,
                        expr.container,
                        get_attr(self, "_ac"),
                        get_attr(self, "_mask")
                )

        def __add__(self, other):
            return type(self).sr_expr(op="eWiseAdd", left=self, right=other)

        def __mul__(self, other):
            return type(self).sr_expr(op="eWiseMult", left=self, right=other)

        #def __matmul__(self, other):
        #    return type(self).sr_expr(op="mxm", left=self, right=other)

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
        self._const = bound_const  # if operator is binary, bind constant
        self._modules = dict()

    def __call__(self, A, output=None):
        # if accumulate is being used, defer apply
        print(A)
        print("A._ac", A._ac)
        if A._ac is not None:
            A._ap = self
            return type(A).ap_expr(op=self._ap, container=A)
        if output is None: output = A
        # get module
        try:
            module = self._modules[(A.dtype, A._ac)]
        except KeyError:
            module = c.get_apply(A.dtype, self._ap, self._const, A._ac)
            # cache module
            self._modules[(A.dtype, A._ac)] = module

        if A._mask is not None:
            module.apply(output.mat, A._mask, A.mat, False)
        else:
            module.apply(output.mat, A.mat, False)
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

    def __call__(self, op, A, B, C=None):
        # get module
        if C is not None:
            ctype = C.dtype, C.dtype 
        else:
            ctype = c.upcast(A.dtype, B.dtype)

        try:
            module = self._modules[(A.dtype, B.dtype, ctype, A._ac)]
        except KeyError:
            module = c.get_semiring(
                    self._ops,
                    A.dtype, 
                    B.dtype,
                    ctype,
                    accum=A._ac
            )
            # cache module
            self._modules[(A.dtype, B.dtype, ctype, A._ac)] = module
        # initialize C
        if C is None:
            # TODO test dimensions are correct for each operator
            if isinstance(A, Matrix) and isinstance(B, Matrix):
                C = Matrix(shape=(B.shape[0], A.shape[1]), dtype=ctype)
            elif isinstance(A, Matrix) and isinstance(B, Vector):
                C = Vector(shape=(A.shape[0],), dtype=ctype)
            elif isinstance(A, Vector) and isinstance(B, Matrix):
                C = Vector(shape=(A.shape[0],), dtype=ctype)
            elif isinstance(A, Vector) and isinstance(B, Vector):
                C = Vector(shape=(max(A.shape[0], B.shape[0]),), dtype=ctype)
                
        if C._mask is not None:
            cmask = C._mask
        else:
            cmask = module.NoMask()
        # call operator on semiring
        getattr(module, op)(C.mat, A.mat, B.mat, cmask)
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
