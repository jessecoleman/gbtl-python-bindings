from GraphBLAS import *

def ops():

    A = Matrix(
        (
            [4213, 234, 242, 1123, 3342], 
            ([1, 2, 2, 3, 3], [0, 1, 2, 1, 3])
        ), shape=(4,4)
    )

    v = Vector([1, 2, 3, 4])


    # element-wise operators

    B = A + A  # eWiseAdd (matrix)

    u = v + v  # eWiseAdd (vector)

    C = A * A  # eWiseMult (matrix)

    w = v * v  # eWiseMult (vector)


    # matrix products

    u = A @ v  # mxv

    w = v @ A  # vxm

    C = A @ A  # mxm


    # with masking
    from scipy import sparse

    # construct empty matrix to assign into
    B = Matrix(shape=A.shape, dtype=A.dtype)

    # construct random matrix from scipy.sparse matrix
    M = Matrix(sparse.random(*A.shape, density=0.5, dtype='float')) 

    B[:] = A + A  # no mask

    B[M] = A * A  # matrix mask

    B[~M] = A @ A  # matrix complement mask

    B[M.T] = A @ A  # matrix transpose mask

    
    # with accumulators

    C = Matrix(shape=A.shape, dtype=A.dtype)
    C[:] = 1  # assign 1 to every element of C

    C[:] += A + A  # default accumulator (arithmetic)

    C[M] += A * A  # accumulator with mask

    C[:,True] += A @ A  # replace_flag = True

    # using non-default operators

    # built-in boolean accumulate
    from GraphBLAS.operators import BooleanAccumulate
    with BooleanAccumulate:
        C[:] += A + A

    # binary operator definition
    from GraphBLAS.operators import BinaryOp
    Max = BinaryOp("Max")
    with Max:
        B = A + A

    # built-in semiring operator
    from GraphBLAS.operators import MaxSelect2ndSemiring
    with MaxSelect2ndSemiring:
        C = A @ A
     
     # assign operator

    A[2, 3] = 3  # assign single element

    # assign with slice
    A[:,:] = B  # assign to entire matrix A

    D = Matrix(([1, 2, 3, 4], ([0, 1, 0, 1], [0, 1, 0, 1])))
    A[2:,:2] = D  # assign to upper-right 2x2 submatrix

    # assign with list/np.array
    i = list(range(A.shape[0]))
    j = list(range(A.shape[1]))
    A[i, j] = B

    ##### TODO #####

    # A[1,:] = v  # assign to row of A

    # A[:,2] = v  # assign to column of A

    ##### END TODO #####

    A[2:,3] = 4  # assign constant to submatrix

    

    # extract operator

    D = A[:]  # extract entire matrix A (essentially copies)
    
    D = A[:2,2:]  # extract lower left 2x2 submatrix of A

    # TODO D[:] = A[::2,::2]  # extract into existing matrix D (no mask)

    D = A[:,1]  # extract column of A


    # apply operator
    from GraphBLAS.operators import apply, UnaryOp

    A = apply(operators.AdditiveInverse, A)  # apply unary operator

    plus1 = UnaryOp("Plus", 1)

    A = apply(plus1, A)  # apply binary operator with bound constant

    A[M] = apply(plus1, A)  # apply with mask


    # reduce operator
    from GraphBLAS.operators import reduce, Monoid

    plus = Monoid("Plus", 0)

    c = reduce(plus, A)  # reduce to constant

    v = Vector(shape=A.shape[0], dtype=A.dtype)
    v[:] = reduce(plus, A)  # reduce to vector


ops()
