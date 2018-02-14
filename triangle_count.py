import GraphBLAS as gb
from GraphBLAS import algorithms
from GraphBLAS import Matrix, Vector
from GraphBLAS.operators import ArithmeticSemiring, BinaryOp, reduce, PlusMonoid
from GraphBLAS import utilities

def triangle_count(graph):

    L, U = utilities.split(graph)

    B = Matrix(shape=graph.shape, dtype=graph.dtype)

    with ArithmeticSemiring:
        B[None] = L @ L.T

    C = Matrix(shape=graph.shape, dtype=graph.dtype)

    with BinaryOp("Times"):
        C[None] = graph * B

    sum = reduce(PlusMonoid, C).eval(0)

    return sum / 2

def triangle_count_masked(graph):

    B = Matrix(shape=graph.shape, dtype=graph.dtype)

    with ArithmeticSemiring:
        B[graph] = graph @ graph.T

    sum = reduce(PlusMonoid, B).eval(0)

    return sum

if __name__ == "__main__":
    import networkx as nx 

    size = 16

    g = nx.gnp_random_graph(size, size**(-1/2))
    i, j = zip(*g.edges())
    i, j = list(i), list(j)
    i_temp = i[:]
    i.extend(j)
    j.extend(i_temp)

    v = [1] * len(i)

    m = Matrix((v, (i, j)), shape=(size, size))

    print("python masked:", triangle_count_masked(m))
    print("cpp masked:", algorithms.triangle_count_masked(m))

