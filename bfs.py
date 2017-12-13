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

def bfs(graph, wavefront, parent_list):

    #algorithms.bfs(graph, wavefront, parent_list)

    parent_list = index_of_1based(parent_list)

    while wavefront.nvals > 0:

        wavefront = index_of_1based(wavefront)

        with MinSelect1stSemiring:
            wavefront[~parent_list,True] = wavefront @ graph

        with ArithmeticAccumulate:
            parent_list[:] += wavefront

    subtract_1 = Apply("Minus", 1)

    parent_list[:,True] = subtract_1(parent_list)

    print("FINAL", parent_list)

import numpy as np
import scipy

i = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 8, 8]
j = [3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6, 2, 3, 8, 2, 1, 2, 3, 2, 4]
v = [1] * len(i)

i = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,
    4,4,4,
    5,5,5,5,
    6,6,6,6,
    7,7,7,7,
    8,8,8,8,8,
    9,9,
    10,10,10,
    11,
    12,12,
    13,13,13,13,13,
    14,14,
    15,15,
    16,16,
    17,17,
    18,18,
    19,19,19,
    20,20,
    21,21,
    22,22,
    22,23,23,23,23,
    24,24,24,
    25,25,25,
    26,26,
    27,27,27,27,
    28,28,28,
    29,29,29,29,
    30,30,30,30,
    31,31,31,31,31,31,
    32,32,32,32,32,32,32,32,32,32,32,32,
    33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33]

j = [
    1,2,3,4,5,6,7,8,10,11,12,13,19,21,23,31,
    0,2,3,7,13,17,19,21,30,
    0,1,3,7,8,9,13,27,28,32,
    0,1,2,7,12,13,
    0,6,10,
    0,6,10,16,
    0,4,5,16,
    0,1,2,3,
    0,2,30,32,33,
    1,33,
    0,4,5,
    0,
    0,3,
    0,1,2,3,33,
    32,33,
    32,33,
    5,6,
    0,1,
    32,33,
    0,1,33,
    32,33,
    0,1,
    32,33,
    25,27,29,32,33,
    25,27,31,
    23,24,31,
    29,33,
    2,23,24,33,
    2,31,33,
    23,26,32,33,
    1,8,32,33,
    1,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32]

v = [1] * len(i)

w = ([1], [30])

graph = gb.Matrix((v, (i, j)), shape=(34,34))
wavefront = gb.Vector(w, shape=(34,))
parent_list = gb.Vector(w, shape=(34,))

print(wavefront)

bfs(graph, wavefront, parent_list)

