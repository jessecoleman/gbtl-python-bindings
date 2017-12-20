from . import c_functions as c
from . import Vector, Matrix

def diagonal(vector):
    length = vector.shape[0]
    matrix = Matrix(
            c.utilities("diagonal", vector=vector)(), 
            shape=(length, length), 
            dtype=vector.dtype, 
            copy=False
    )
    return matrix

def scaled_identity(shape, scalar):

    return c.utilities(
            function    = "scaled_identity",
            kwargs      = [("a_type", c.types[type(scalar)])],
    )(mat_size=shape, val=scalar)

def split(matrix):

    lower = matrix._out_container()
    upper = matrix._out_container()
    c.utilities(
            "split",
            A = matrix,
            L = lower,
            U = upper
    )
    return lower, upper

def normalize_rows(matrix):

    c.utilities("normalize_rows", matrix=matrix)
    return matrix

def normalize_cols(matrix):

    c.utilities("normalize_cols", matrix=matrix)
    return matrix

# TODO implement dimension checking
def check_dims(container):
    pass
