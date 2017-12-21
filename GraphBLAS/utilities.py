from . import c_functions as c
from . import Vector, Matrix

@c.tc(Vector)
def diagonal(vector):
    length = vector.shape[0]
    matrix = Matrix(
            c.utilities("diagonal", vector=vector)(), 
            shape=(length, length), 
            dtype=vector.dtype, 
            copy=False
    )
    return matrix

@c.typecheck
def scaled_identity(shape: int, scalar: (int, float)):

    return c.utilities(
            function    = "scaled_identity",
            kwargs      = [("a_type", c.types[type(scalar)])],
    )(mat_size=shape, val=scalar)

@c.typecheck
def split(matrix: Matrix):

    lower = matrix._out_container()
    upper = matrix._out_container()
    c.utilities(
            "split",
            A = matrix,
            L = lower,
            U = upper
    )()
    return lower, upper

@c.typecheck
def normalize_rows(matrix: Matrix):

    f = c.utilities("normalize_rows", A=matrix)
    return matrix

@c.typecheck
def normalize_cols(matrix: Matrix):

    c.utilities("normalize_cols", A=matrix)()
    return matrix

# TODO implement dimension checking
def check_dims(container):
    pass
