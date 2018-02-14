import GraphBLAS as gb
from GraphBLAS import Matrix, Vector
from GraphBLAS.operators import MinPlusSemiring, Accumulator
from GraphBLAS import algorithms

def sssp(graph, path):

    if (graph.shape[0] != path.shape[0] 
            or graph.shape[1] != path.shape[0]):
        raise Error()

    with MinPlusSemiring, Accumulator("Min"):
        for i in range(graph.shape[0]):
            path[None] += path @ graph

