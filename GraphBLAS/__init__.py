# import Matrix and Vector classes
from .containers import Matrix, Vector
from .operators import ArithmeticSemiring, ArithmeticAccumulate

# set arithmetic semiring as default
ArithmeticSemiring.__enter__()
ArithmeticAccumulate.__enter__()

__all__ = ['Matrix', 'Vector', 'algorithms', 'operators']
