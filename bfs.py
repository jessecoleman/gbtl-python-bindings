import GraphBLAS as gb
from GraphBLAS.operators import *
from GraphBLAS import algorithms

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

    parents = algorithms.bfs(graph, 3)
    print(parents)

    exit()

    w = [0] * 9
    w[3] = 1
    wavefront = gb.Vector(w)
    parent_list = gb.Vector(w)

    parent_list = index_of_1based(parent_list)
    print(graph)

    while wavefront.nvals > 0:

        #wavefront = index_of_1based(wavefront)
        print("wavefront1:",wavefront)

        with MinSelect1stSemiring:
            wavefront[~parent_list,True] = wavefront @ graph

        print("wavefront2:",wavefront)
        
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

graph = gb.Matrix((v, (i, j)), shape=(9,9))

bfs(graph, None)

