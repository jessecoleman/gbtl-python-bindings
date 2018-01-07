import GraphBLAS as gb
from GraphBLAS import utilities, algorithms
from GraphBLAS.operators import *

def tri_count(graph):
    rows = graph.shape[0]
    cols = graph.shape[1]

    L, U = gb.utilities.split(graph)

    B = gb.Matrix(shape=graph.shape, dtype=graph.dtype)
    C = gb.Matrix(shape=graph.shape, dtype=graph.dtype)

    with ArithmeticSemiring:
        B[:] = L @ L.T

        C[:] = graph * B
    
    plus = Monoid(binary_ops.plus, identities.additive)

    sum = reduce(plus, C).eval() / 2
    return sum

iL = []
iU = []
iA = []

jL = []
jU = []
jA = []

with open("triangle_count_data_ca-HepTh.tsv", 'r') as f:

    rows = 0
    max_id = 0

    for line in f:

        src, dst = [int(i) for i in line.strip().split()]

        if src > max_id:
            max_id = src
        if dst > max_id:
            max_id = dst

        if src < dst:
            iA.append(src)
            jA.append(dst)

            iU.append(src)
            jU.append(dst)

        elif src > dst:
            iA.append(src)
            jA.append(dst)

            iL.append(src)
            jL.append(dst)
            
        rows += 1

num_nodes = max_id + 1

v = [1] * len(iA)

print(num_nodes)
print(len(iA), len(jA), len(v))

A = gb.Matrix((v, (iA, jA)), shape=(num_nodes, num_nodes))

print(A.shape, A.nvals)

#L = gb.Matrix((v, (iL, jL)), shape=(num_nodes, num_nodes))
#U = gb.Matrix((v, (iU, jU)), shape=(num_nodes, num_nodes))

print(tri_count(A))

print(gb.algorithms.triangle_count(A))
 
