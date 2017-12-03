import GraphBLAS as gb
from GraphBLAS.operators import *

def index_of_1based(vector):
    parent_list = gb.Vector([1] + [0] * 9)
    i, j, v = [],[],[]
    
    for idx in range(len(vector)):
        i.append(idx)
        j.append(idx)
        v.append(idx+2)

    identity_ramp = gb.Matrix((v,(i,j)))

    subtract_1 = Apply("Minus", 1)

    print(vector)
    print(identity_ramp)
    print(~parent_list)
    print("CLOSURE")
    print(subtract_1(parent_list)(parent_list))

    with MinSelect2ndSemiring:
        vector[~parent_list] = vector @ identity_ramp

    print(vector)

def bfs(graph, wavefront):
    

    parent_list = gb.Vector([i for i in range(10)])
    wavefront = gb.Vector([-10] + [0] * 9)
    wavefront1 = gb.Vector([1] * 10)


    Add2 = Apply("Plus", 2)
    print(wavefront1)
    wavefront[:] = wavefront1
    print(wavefront)


    Multiply2 = Apply("Times", 2)

    #print(graph)
    #graph[:] += Add2(graph)
    #print(graph)
    #graph[:] += Add2(graph)
    #print(graph)

    #print(Multiply2(graph))
    #print(Multiply2(graph)())

    exit()

    for i in range(2):
        graph[:] = Add2(graph)
        print(graph)

    graph[:] += graph

    print("masked")
    print(graph[::2,::,True]) 

    graph[::2,::,True] = Add2(graph)

    print(graph)

    exit()

    #wavefront1[2:3]
    #print(dir(wavefront1))
    #print(wavefront._mask)

    wavefront[::2, True] += wavefront1
    print(wavefront)

    exit()

    print(parent_list)
    print(wavefront)

    Subtract2 = Apply("Minus", 2)
    DivideBy2 = Apply("Times", 0.5)

    wavefront = Subtract2(wavefront)
    print("WAVEFRONT", wavefront)
    wavefront = wavefront()
    print("WAVEFRONT", wavefront)
    wavefront = DivideBy2(wavefront)
    print("WAVEFRONT", wavefront)
    wavefront = wavefront()
    print("WAVEFRONT", wavefront)

    exit()

    print((parent_list + wavefront + wavefront1)())

    exit()

    for i in range(5):
        wavefront[:] += wavefront1 + parent_list
        wavefront[:] = wavefront1
        print("WAVEFRONT")
        print(wavefront)

    exit()

    print((parent_list + wavefront)(wavefront1, ArithmeticAccumulate))
    
    exit()

    with ArithmeticAccumulate:
        wavefront[::2] += parent_list
        print(wavefront)
        (wavefront1 + parent_list)(wavefront)
        print(wavefront)
        (wavefront1 + parent_list)(wavefront)
        print(wavefront)

    while wavefront.nvals > 0:
        index_of_1based(wavefront)

        print(wavefront)
        with MinSelect1stSemiring:
            wavefront[~parent_list] = wavefront @ graph
        print(wavefront)

        with ArithmeticAccumulate:
            parent_list += Identity(wavefront)


import numpy as np
import scipy

graph = np.random.rand(10,10)
graph[graph > .6] = 1
graph[graph <= .6] = 0
bfs(gb.Matrix(scipy.sparse.coo_matrix(graph.astype("int64"))), None)

