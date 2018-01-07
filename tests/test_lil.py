import sys
import GraphBLAS as gb
from GraphBLAS import algorithms
from GraphBLAS.operators import *
import numpy as np
import scipy.sparse as sparse
import random


def bfs():
    i = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,
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
    23,23,23,23,23,
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

    j = [1,2,3,4,5,6,7,8,10,11,12,13,19,21,23,31,
    0,2,3,7,13,17,19,21,30,
    0,1,3,7,8,9,13,27,28,32,
    0,1,2,7,12,13,
    0,6,10,
    0,6,10,16,
    0,4,5,16,
    0,1,2,3,
    0,2,30,32,33,
    2,33,
    0,4,5,
    0,
    0,3,
    1,1,2,3,33,
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
    0,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32]

    v = [True] * len(i)

    csr = sparse.csr_matrix((v, (i,j)), dtype=np.uint32)

    m = gb.Matrix(csr)

    root = [0] * (max(i) + 1)
    root[30] = 1
    rootc = gb.Matrix(sparse.csr_matrix(root, dtype=np.uint32))

    levels = gb.bfs_level(m, rootc)
    print(levels)

def sssp():
    i = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 8, 8]
    j = [3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6, 2, 3, 8, 2, 1, 2, 3, 2, 4]

    v = [1] * len(i)

    csr = sparse.csr_matrix((v, (i,j)))
    print(type(csr))
    m = gb.Matrix(csr)

    path = gb.Vector(np.array([0] * 9))

    algorithms.sssp(m, path)

    print(path)

def tricount():
    iA = []
    jA = []
    iU = []
    jU = []
    iL = []
    jL = []

    max_id = 0

    with open("triangle_count_data_ca-HepTh.tsv") as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            if src > max_id: max_id = src
            if dst > max_id: max_id = dst

            iA.append(src)
            jA.append(dst)
            if src < dst:
                iU.append(src)
                jU.append(dst)
            elif src > dst:
                iL.append(src)
                jL.append(dst)

    num_nodes = max_id + 1

    vA = [1] * len(iA)
    A = sparse.csr_matrix((vA, (iA, jA)), shape=(num_nodes,num_nodes))
    vU = [1] * len(iU)
    U = sparse.csr_matrix((vU, (iU, jU)), shape=(num_nodes,num_nodes))
    vL = [1] * len(iL)
    L = sparse.csr_matrix((vL, (iL, jL)), shape=(num_nodes,num_nodes))

    print(L.shape)
    gL = gb.Matrix(L)
    gU = gb.Matrix(U)

    print(sys.getsizeof(gL))

    triangles = algorithms.triangle_count(gL, gU)
    print(triangles)

def test_mult():
    i_mA = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    j_mA = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    v_mA = [12, 7, 3, 4, 5, 6, 7, 8, 9]

    sA = sparse.coo_matrix((v_mA, (i_mA, j_mA)))

    print("initA")
    A = gb.Matrix((v_mA, i_mA, j_mA))
    print(A)

    i_mB = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    j_mB = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3])
    v_mB = np.array([5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1])

    sB = sparse.coo_matrix((v_mB, (i_mB, j_mB)))

    print("initB")
    B = gb.Matrix((v_mB, i_mB, j_mB))
    print(B)

    i_answer = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    j_answer = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    v_answer = [114, 160, 60, 27, 74, 97, 73, 14, 119, 157, 112, 23]
    print("initANS")
    ANS = gb.Matrix((v_answer, i_answer, j_answer))
    print(ANS)

    ANS2 = sA * sB
    print(ANS2.toarray())

    #C = A @ B

    print(C)

    A = sparse.random(8, 10, density=0.25)
    B = sparse.random(10, 6, density=0.25)

    print(A.toarray())
    print(B.toarray())

    #C = A @ B
    print(C.toarray())

    print("init A, B")
    #ANS = gb.Matrix(A) @ gb.Matrix(B)
    print(ANS)

import math
import random
def test_semiring():
    a = sparse.coo_matrix(np.array([float(math.ceil(5 * random.random())) for i in range(16)]).reshape(4,4))
    b = sparse.random(4, 4, density=0.5, format='coo', dtype='float')
    c = sparse.coo_matrix(np.array([1] * 16).reshape(4,4))
    d = sparse.coo_matrix(np.array([random.choice((True, False)) for i in range(16)]).reshape(4,4))

    A = gb.Matrix(a)
    B = gb.Matrix(b)
    C = gb.Matrix(c)
    D = gb.Matrix(d)

    print(A + B)

    with MaxSelect2ndSemiring, ArithmeticAccumulate:
        print("semiring")
        print(C)
        C += A + B
        print(C)
        D = A @ B
        print(D)
        exit()

        with ArithmeticAccumulate:
            print(C, C2)
            C += A + B
            C2 += A + B
            print(C, C2)

    with ArithmeticAccumulate: 
        C += A + B
        #G = A + B
        print(C)
        print(C1)
        #print(G)
        with MaxSelect2ndSemiring:
            print("max_select_second")
            F = A @ B
            G = A + B
            print(F)
            print(G)

        print("min_select_first_rev")
        F = B * A
        G = B + A
        print(F)
        print(G)

    
def bfs(graph, wavefront, parent_list):

        # Set the roots parents to themselves using one-based indices because
        # the mask is sensitive to stored zeros.
        parent_list = wavefront
        index_of_1based(parent_list)

        while (wavefront.nvals() > 0):
            # convert all stored values to their 1-based column index
            index_of_1based(wavefront)

            # Select1st because we are left multiplying wavefront rows
            # Masking out the parent list ensures wavefront values do not
            # overlap values already stored in the parent list
            #wavefront = wavefront @ graph
            #GraphBLAS::vxm(wavefront,
            #               GraphBLAS::complement(parent_list),
            #               GraphBLAS::NoAccumulate(),
            #               GraphBLAS::MinSelect1stSemiring<T>(),
            #               wavefront, graph, true);

            ## We don't need to mask here since we did it in mxm.
            ## Merges new parents in current wavefront with existing parents
            ## parent_list<!parent_list,merge> += wavefront
            #GraphBLAS::apply(parent_list,
            #                 GraphBLAS::NoMask(),
            #                 GraphBLAS::Plus<T>(),
            #                 GraphBLAS::Identity<T>(),
            #                 wavefront,
            #                 false);

        # Restore zero-based indices by subtracting 1 from all stored values
        #GraphBLAS::BinaryOp_Bind2nd<unsigned int,
        #                            GraphBLAS::Minus<unsigned int>>
        #    subtract_1(1);

        #GraphBLAS::apply(parent_list,
        #                 GraphBLAS::NoMask(),
        #                 GraphBLAS::NoAccumulate(),
        #                 subtract_1,
        #                 parent_list,
        #                 true);

def index_of_1based(vec):
    i = []
    j = []
    v = []
    for ix in range(vec.size()):
        i.append(ix)
        j.append(ix)
        v.append(ix + 1)

    identity_ramp = gb.Matrix((v, i, j))

    print(identity_ramp)

    #GraphBLAS::vxm(vec,
    #               GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
    #               GraphBLAS::MinSelect2ndSemiring<T>(),
    #               vec, identity_ramp, true);

class MyNumber(object):
    def __init__(self, i):
        self.i = i
        
    def __add__(self, other):
        if isinstance(other, MyExpression):
            return MyExpression([self, "+"] + other.expression)
        return MyExpression([self, "+", other])
        
    def __mul__(self, other):
        if isinstance(other, MyExpression):
            return MyExpression([self, "*"] + other.expression)
        return MyExpression([self, "*", other])

    def __iadd__(self, other):
        if isinstance(other, MyExpression):
            return MyExpression([self, "+="] + other.expression)
        return MyExpression([self, "+=", other])

    def __getitem__(self, other):
        if isinstance(other, MyExpression):
            return MyExpression([self, "[]"] +  other.expression)
        return MyExpression([self, "[]", other])



    def __repr__(self):
        return str(self.i)

    def __str__(self): 
        return str(self.i)

class MyExpression(object):

    def __iadd__(self, other):
        if isinstance(other, MyExpression):
            return MyExpression(self.expression + ["+="] + other.expression)
        return MyExpression([self, "+=", other])

    def __setitem__(self, i, val):
        if isinstance(val, MyExpression):
            return MyExpression([self, "[%s]" % i] +  val.expression)
        return MyExpression([self, "[%s]" %i, val])

    def __getitem__(self, other):
        if isinstance(other, MyExpression):
            return MyExpression([self, "[]"] +  other.expression)
        return MyExpression([self, "[]", other])

    def __init__(self, e): 
        self.expression = [e]

    def evaluate(self):
        return eval(" ".join(map(str, self.expression)))

    def __str__(self): return str(self.expression)

def order_of_operations():
    c = MyExpression(0)
    print(MyNumber(1) + MyNumber(2) * MyNumber(5))
    c[0] += MyNumber(1) + MyNumber(2) * MyNumber(5)
    print(c.expression)

#order_of_operations()
#sssp()
#test_mult()
test_semiring()
#tricount()
