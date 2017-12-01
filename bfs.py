import GraphBLAS as gb
from GraphBLAS.operators import *

def index_of_1based(vector):
    i, j, v = [],[],[]
    
    for idx in range(len(vector)):
        i.append(idx)
        j.append(idx)
        v.append(idx+1)

    #result = gb.Vector([False] * 8)

    identity_ramp = gb.Matrix((v,(i,j)))

    with MinSelect2ndSemiring:
        result = vector * identity_ramp

def bfs(graph, wavefront):
    parent_list = gb.Vector([1] + [0] * 9)
    wavefront = gb.Vector([1] + [0] * 9)

    while wavefront.nvals > 0:
        index_of_1based(wavefront)

        with MinSelect1stSemiring:
            wavefront[~parent_list] = wavefront * graph

        with ArithmeticAccumulate:
            print("parent list accumulate")
            print(Identity(wavefront))
            parent_list += Identity(wavefront)
            print(parent_list)
            print("accumulated")

import numpy as np
import scipy

graph = np.random.rand(10,10)
graph[graph > .6] = 1
graph[graph <= .6] = 0
print(graph)
bfs(gb.Matrix(scipy.sparse.coo_matrix(graph.astype("int64"))), None)

