import GraphBLAS as gb
from GraphBLAS.operators import *

def index_of_1based(vector):
    i, j, v = [],[],[]
    
    for idx in range(len(vector)):
        i.append(idx)
        j.append(idx)
        v.append(idx+1)

    identity_ramp = gb.Matrix((v,(i,j)))

    with MinSelect2ndSemiring:
        vector[True] = vector @ identity_ramp

    return vector

def bfs(graph, wavefront):

    parent_list = gb.Vector([i for i in range(20)])
    wavefront = gb.Vector([1] + [0] * 19)

    print(wavefront)

    add_20 = Apply("Plus", 20)

    result = add_20(graph).eval()

    print(graph)
    print(result)

    subtract_1 = Apply("Minus", 1)
    print(wavefront[:])

    wavefront[:] = subtract_1(wavefront)

    print(wavefront)

    exit()

    wavefront = index_of_1based(wavefront)

    while wavefront.nvals > 0:

        wavefront = index_of_1based(wavefront)

        with MinSelect1stSemiring:
            wavefront[~parent_list,True] = wavefront @ graph

        with ArithmeticAccumulate:
            parent_list[:] += Identity(wavefront)

        print(parent_list)

    subtract_1 = Apply("Minus", 1)

    parent_list[:,True] = subtract_1(parent_list)

    print(parent_list)

import numpy as np
import scipy

i = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 8, 8]
j = [3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6, 2, 3, 8, 2, 1, 2, 3, 2, 4]
v = [1] * len(i)

graph = gb.Matrix((v, (i, j)))

print(graph)
bfs(graph, None)

