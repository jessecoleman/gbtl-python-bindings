from . import c_functions as c
from . import Vector, Matrix

@c.type_check
def diagonal(vector: Vector):
    length = vector.shape[0]
    matrix = Matrix(
            c.utilities("diagonal", vector=vector),
            shape = (length, length), 
            dtype = vector.dtype, 
            copy  = False
    )
    return matrix

@c.type_check
def scaled_identity(shape: int, scalar: (int, float)):

    return c.utilities(
            function    = "scaled_identity",
            kwargs      = {"a_type": c.types[type(scalar)]},
    )(mat_size=shape, val=scalar)

@c.type_check
def split(matrix: Matrix):

    lower = Matrix(shape=matrix.shape, dtype=matrix.dtype)
    upper = Matrix(shape=matrix.shape, dtype=matrix.dtype)
    c.utilities(
            "split",
            A = matrix,
            L = lower,
            U = upper
    )
    return lower, upper

@c.type_check
def normalize_rows(matrix: Matrix):

    c.utilities("normalize_rows", A=matrix)
    return matrix

@c.type_check
def normalize_cols(matrix: Matrix):

    c.utilities("normalize_cols", A=matrix)
    return matrix

