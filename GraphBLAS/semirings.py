import GraphBLAS as gb
from GraphBLAS import compile_c as c

__all__ = ['ArithmeticSemiring', 
           'MinPlusSemiring', 
           'MaxTimesSemiring', 
           'MinSelect2ndSemiring', 
           'MaxSelect2ndSemiring',
           'MinSelect1stSemiring',
           'MaxSelect1stSemiring']


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # SemiringBase is abstract
            if cls == SemiringBase: 
                def __abstract_init__(self): 
                    raise TypeError("Can't instantiate abstract class SemiringBase")
                cls.__init__ = __abstract_init__
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class SemiringBase(metaclass=Singleton):

    #TODO: cache modules at class-level
    __modules = {}

    @classmethod
    def __get_module(cls, A, B):
        s = c._get_module(A.dtype, semiring=(
            cls._add_binop, 
            cls._add_idnty, 
            cls._mul_binop)
        )
        return s

    @classmethod
    def add(cls, A, B):
        s = cls.__get_module(A,B)
        return s.eWiseAdd(A.mat, B.mat)

    @classmethod
    def dot(cls, A, B):
        pass

    @classmethod
    def multiply(cls, A, B):
        s = cls.__get_module(A,B)
        return s.eWiseMult(A.mat, B.mat)

    @classmethod
    def matmul(cls, A, B):
        s = cls.__get_module(A, B)
        return s.mxm(A.mat, B.mat)

    @classmethod
    def __enter__(cls):
        # provide semring module to operator mixins
        cls._MatrixOps.sr = cls.__get_module
        cls._VectorOps.sr = cls.__get_module
        # mix in operator classes
        cls.__mat_base = gb.Matrix.__bases__
        cls.__vec_base = gb.Vector.__bases__
        gb.Matrix.__bases__ = (cls._MatrixOps,)
        gb.Vector.__bases__ = (cls._VectorOps,)
        return cls

    @classmethod
    def __exit__(cls, exception_type, exception_value, traceback):
        # reset default operator mixins
        gb.Matrix.__bases__ = cls.__mat_base
        gb.Vector.__bases__ = cls.__vec_base

    # Matrix mixin to provide operator overloads
    class _MatrixOps(object):

        def __add__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.eWiseAdd(self.mat, other.mat)

        def __mul__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.eWiseMult(self.mat, other.mat)

        def __matmul__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.mxm(self.mat, other.mat)

        def __rmatmul__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.mxm(other.mat, self.mat)

    # Vector mixin to provide operator overloads
    class _VectorOps(object):

        def __add__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.eWiseAdd(self.vec, other.vec)

        def __mul__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.eWiseMult(self.vec, other.vec)

        def __matmul__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.vxm(self.vec, other.mat)

        def __rmatmul__(self, other):
            sr = self.__class__.sr(self, other)
            return sr.mxv(other.mat, self.vec)

class ArithmeticSemiring(SemiringBase):
    _add_binop = "Plus"
    _add_idnty = "0"
    _mul_binop = "Times"

class MinPlusSemiring(SemiringBase):
    _add_binop = "Min"
    _add_idnty = "MinIdentity"
    _mul_binop = "Plus"
                     
class MaxTimesSemiring(SemiringBase):
    # @todo The following identity only works for unsigned domains
    _add_binop = "Max"
    _add_idnty = "0"
    _mul_binop = "Times"

class MinSelect2ndSemiring(SemiringBase):
    _add_binop = "Min"
    _add_idnty = "MinIdentity"
    _mul_binop = "Second"

class MaxSelect2ndSemiring(SemiringBase):
    _add_binop = "Max"
    _add_idnty = "0"
    _mul_binop = "Second"

class MinSelect1stSemiring(SemiringBase):
    _add_binop = "Min"
    _add_idnty = "MinIdentity"
    _mul_binop = "First"

class MaxSelect1stSemiring(SemiringBase):
    _add_binop = "Max"
    _add_idnty = "0"
    _mul_binop = "First"

