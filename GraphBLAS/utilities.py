from . import compile_c as c
from . import Vector, Matrix

def diagonal(vector):
    m = c.get_utilities("diagonal", vector)
    length = vector.shape[0]
    matrix = Matrix(
            m.diagonal(vector.vec), 
            shape=(length, length), 
            dtype=vector.dtype, 
            copy=False
    )
    return matrix

def scaled_identity(shape, scalar):
    m = c.get_utilities(
            "scaled_identity", 
            type=[
                ("a_Matrix", 1), 
                ("atype", c.types[type(scalar)])
            ]
    )
    matrix = m.scaled_identity(shape, scalar)
    return matrix

def split(matrix):
    m = c.get_utilities("split", matrix)
    lower = matrix._out_container()
    upper = matrix._out_container()
    m.split(matrix.mat, lower.mat, upper.mat)
    return lower, upper

def normalize_rows(matrix):
    m = c.get_utilities("normalize_rows", matrix)
    m.normalize_rows(matrix.mat)
    return matrix

def normalize_cols(matrix):
    m = c.get_utilities("normalize_cols", matrix)
    m.normalize_cols(matrix.mat)
    return matrix

# TODO implement dimension checking
def check_dims(container):
    pass
