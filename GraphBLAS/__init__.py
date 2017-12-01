# import Matrix and Vector classes
from GraphBLAS.containers import Matrix, Vector
from GraphBLAS.operators import ArithmeticSemiring, ArithmeticAccumulate

# set arithmetic semiring as default
ArithmeticSemiring.__enter__()
ArithmeticAccumulate.__enter__()

__all__ = ['matrix', 'vector', 'algorithms', 'operators']
