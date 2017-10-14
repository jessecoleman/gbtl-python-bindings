import math
#import sys
#sys.path.append("~/graphpack/gbtl/src/demo/graphBLAS.cpython-35m-x86_64-linux-gnu.so")
import numpy as np
import graphBLAS as gb
from graphBLAS import algorithms

def main():
    NUM_NODES = 9
    INF = gb.max_val;
    m = gb.Matrix(NUM_NODES, NUM_NODES, INF)
    i = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 8, 8]
    j = [3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6, 2, 3, 8, 2, 1, 2, 3, 2, 4]
    v = [1.0] * len(i)

    gb.build_matrix(m, i, j, v, len(i))
    #print("m: zero = " + str(m.get_zero()))
    roots = gb.identity(NUM_NODES, INF, 0);

    m1 = gb.Matrix(NUM_NODES, NUM_NODES, INF)
    algorithms.bfs(m, roots, m1)

    print("Parents by rows: zero = " + str(m1.get_zero()))
    print(m1)

if __name__=='__main__':
    main()
